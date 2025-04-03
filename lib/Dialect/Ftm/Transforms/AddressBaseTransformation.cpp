//===--------- AddressBaseTransformation.cpp - address base transformation Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_ADDRESSBASETRANSFORMATION
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

void implAddressBaseTransformation(func::FuncOp funcOp) {
    // 遍历函数中所有的scf::forOp
    funcOp.walk([&](scf::ForOp forOp) {
        // 检查循环是否有迭代参数
        if (forOp.getInitArgs().empty())
            return;

        // 存储映射关系：offset迭代参数 -> (base值, increment值)
        DenseMap<Value, std::pair<Value, Value>> offsetToBaseMap;
        // 记录需要移除的加法操作
        SmallVector<Operation*, 4> addOpsToRemove;
        // 记录offset在迭代参数中的索引
        DenseMap<Value, unsigned> offsetToIndexMap;

        // 遍历迭代参数，判断是否被两个加法使用
        // 其中一个加法的另一个参数是变量，也就是base值，将该加法加入到addOpsToRemove
        // 另一个加法是常量加法，也就是increment值
        for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
            Value iterArg = forOp.getRegionIterArgs()[i];
            Value baseValue = nullptr;
            Value incrementValue = nullptr;
            Operation* addrAddOp = nullptr;
            
            // 遍历使用该迭代参数的操作
            for (Operation* user : iterArg.getUsers()) {
                // 检查是否是加法操作
                if (auto addOp = dyn_cast<arith::AddIOp>(user)) {
                    Value lhs = addOp.getLhs();
                    Value rhs = addOp.getRhs();
                    
                    // 检查是否是 base + offset 形式
                    if (lhs == iterArg && !rhs.getDefiningOp<arith::ConstantOp>()) {
                        baseValue = rhs;
                        addrAddOp = addOp;
                    } else if (rhs == iterArg && !lhs.getDefiningOp<arith::ConstantOp>()) {
                        baseValue = lhs;
                        addrAddOp = addOp;
                    }
                    
                    // 检查是否是 offset + increment 形式
                    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                    if (addOp.getResult() == yieldOp.getOperands()[i]) {
                        if (lhs == iterArg && rhs.getDefiningOp<arith::ConstantOp>()) {
                            incrementValue = rhs;
                        } else if (rhs == iterArg && lhs.getDefiningOp<arith::ConstantOp>()) {
                            incrementValue = lhs;
                        }
                    }
                }
            }
            
            // 如果找到了完整的模式，记录下来
            if (baseValue && incrementValue && addrAddOp) {
                offsetToBaseMap[iterArg] = {baseValue, incrementValue};
                offsetToIndexMap[iterArg] = i;
                addOpsToRemove.push_back(addrAddOp);
            }
        }

        // 如果没有找到可优化的模式，直接返回
        if (offsetToBaseMap.empty())
            return;
            
        OpBuilder builder(forOp);

        // 准备新的初始值
        // 传入的初始化参数需要首先进行一次base = base + offset，如果offset初始为常量0则跳过该加法
        SmallVector<Value, 4> newInitArgs;
        for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
            Value initArg = forOp.getInitArgs()[i];
            Value iterArg = forOp.getRegionIterArgs()[i];
            
            if (offsetToBaseMap.count(iterArg)) {
                // 这是一个offset参数，需要替换为base
                Value base = offsetToBaseMap[iterArg].first;
                
                // 检查初始offset是否为0
                if (auto constOp = initArg.getDefiningOp<arith::ConstantOp>()) {
                    auto constAttr = constOp.getValue().cast<IntegerAttr>();
                    if (constAttr.getInt() == 0) {
                        // offset初始为0，直接使用base
                        newInitArgs.push_back(base);
                    } else {
                        // offset初始不为0，需要base = base + offset
                        auto baseWithOffset = builder.create<arith::AddIOp>(
                            forOp.getLoc(), base, initArg);
                        newInitArgs.push_back(baseWithOffset);
                    }
                } else {
                    // offset初始不是常量，需要base = base + offset
                    auto baseWithOffset = builder.create<arith::AddIOp>(
                        forOp.getLoc(), base, initArg);
                    newInitArgs.push_back(baseWithOffset);
                }
            } else {
                // 保留原有参数
                newInitArgs.push_back(initArg);
            }
        }

        // 创建新的循环
        auto newForOp = builder.create<scf::ForOp>(
            forOp.getLoc(),
            forOp.getLowerBound(),
            forOp.getUpperBound(),
            forOp.getStep(),
            newInitArgs,
            [&](OpBuilder &nestedBuilder, Location loc, Value iv, ValueRange iterArgs) {
                // 创建映射
                IRMapping mapper;
                mapper.map(forOp.getInductionVar(), iv);
                
                // 映射迭代参数
                for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
                    Value oldArg = forOp.getRegionIterArgs()[i];
                    mapper.map(oldArg, iterArgs[i]);
                    
                    // 如果是offset参数，还需要映射对应的地址计算结果
                    if (offsetToBaseMap.count(oldArg)) {
                        // 找到对应的地址计算操作
                        for (Operation* user : oldArg.getUsers()) {
                            if (std::find(addOpsToRemove.begin(), addOpsToRemove.end(), user) != addOpsToRemove.end()) {
                                // 将地址计算结果映射为基址参数
                                mapper.map(user->getResult(0), iterArgs[i]);
                            }
                        }
                    }
                }
                
                // 克隆循环体中的操作，排除需要移除的加法操作
                for (auto &op : forOp.getRegion().front()) {
                    // 跳过要移除的加法操作
                    if (std::find(addOpsToRemove.begin(), addOpsToRemove.end(), &op) != addOpsToRemove.end()) {
                        continue;
                    }
                    
                    nestedBuilder.clone(op, mapper);
                }
            });
        
        // 替换旧循环的结果
        forOp->replaceAllUsesWith(newForOp);
        
        // 移除旧循环
        forOp.erase();
    });
}

} // namepsace

namespace mlir {
class AddressBaseTransformationPass : public impl::AddressBaseTransformationBase<AddressBaseTransformationPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implAddressBaseTransformation(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createAddressBaseTransformationPass() {
  return std::make_unique<AddressBaseTransformationPass>();
}