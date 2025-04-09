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
#include "llvm/ADT/SmallSet.h"

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

        // offset迭代参数 -> base + offset
        DenseMap<Value, arith::AddIOp> offsetToBaseAddMap;
        // offset迭代参数 -> cast
        DenseMap<Value, ftm::CastOp> offsetToCastMap;
        // offset迭代参数 -> offset + increment
        DenseMap<Value, arith::AddIOp> offsetToOffsetAddMap;
        llvm::SmallSet<Operation*, 8> opsToRemove;

        // 遍历迭代参数，判断是否被两个加法使用
        for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
            Value iterArg = forOp.getRegionIterArgs()[i];
            arith::AddIOp baseAddOp = nullptr;
            arith::AddIOp offsetAddOp = nullptr;
            ftm::CastOp castOp = nullptr;
            
            // 遍历使用该迭代参数的操作
            for (Operation* user : iterArg.getUsers()) {
                if (auto addOp = dyn_cast<arith::AddIOp>(user)) {
                    Value lhs = addOp.getLhs();
                    Value rhs = addOp.getRhs();
                    bool containsIterArg = (lhs == iterArg || rhs == iterArg);
                    Value otherOperand = (lhs == iterArg) ? rhs : lhs;

                    // Check for base + offset pattern
                    if (containsIterArg && !otherOperand.getDefiningOp<arith::ConstantOp>() &&
                        addOp.getResult().hasOneUse() &&
                        (castOp = dyn_cast<ftm::CastOp>(*addOp.getResult().user_begin())) &&
                        castOp.getType().isa<LLVM::LLVMPointerType>()) {
                        baseAddOp = addOp;
                    }

                    // Check for offset + increment pattern
                    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                    if (containsIterArg && otherOperand.getDefiningOp<arith::ConstantOp>() &&
                        addOp.getResult() == yieldOp.getOperands()[i]) {
                        offsetAddOp = addOp;
                    }
                }
            }
            
            // 如果找到了完整的模式，记录下来
            if (baseAddOp && castOp && offsetAddOp) {
                offsetToBaseAddMap[iterArg] = baseAddOp;
                offsetToCastMap[iterArg] = castOp;
                offsetToOffsetAddMap[iterArg] = offsetAddOp;
                // 移除base + offset和cast
                opsToRemove.insert(baseAddOp);
                opsToRemove.insert(castOp);
            }
        }

        // 如果没有找到可优化的模式，直接返回
        if (offsetToBaseAddMap.empty())
            return;
            
        OpBuilder builder(forOp);
        builder.setInsertionPoint(forOp);

        // 准备新的初始值
        // 传入的初始化参数需要首先进行一次basePtr = basePtr + offset，如果offset初始为常量0则跳过该加法
        SmallVector<Value, 4> newInitArgs;
        for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
            Value initArg = forOp.getInitArgs()[i];
            Value iterArg = forOp.getRegionIterArgs()[i];
            
            if (offsetToBaseAddMap.count(iterArg)) {
                auto baseAdd = offsetToBaseAddMap[iterArg];
                Value base = offsetToBaseAddMap.contains(baseAdd.getLhs()) ? 
                    baseAdd.getRhs() : baseAdd.getLhs();
                // 将base转换为!llvm.ptr类型
                Value basePtr = builder.create<ftm::CastOp>(forOp.getLoc(), 
                    LLVM::LLVMPointerType::get(builder.getContext()), 
                    base);
                
                // 检查初始offset是否为0
                if (auto constOp = initArg.getDefiningOp<arith::ConstantOp>()) {
                    auto constAttr = constOp.getValue().cast<IntegerAttr>();
                    if (constAttr.getInt() == 0) {
                        // offset初始为0，直接使用basePtr
                        newInitArgs.push_back(basePtr);
                    } else {
                        // offset初始不为0，需要basePtr = basePtr + offset
                        auto baseWithOffset = builder.create<ftm::AddaOp>(
                            forOp.getLoc(), basePtr, initArg);
                        newInitArgs.push_back(baseWithOffset);
                    }
                } else {
                    // offset初始不是常量，需要base = basePtr + offset
                    auto baseWithOffset = builder.create<ftm::AddaOp>(
                        forOp.getLoc(), basePtr, initArg);
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
                    Value newArg = iterArgs[i];
                    mapper.map(oldArg, newArg);
                    
                    // 如果是offset参数，映射cast后的指针到迭代参数
                    if (offsetToCastMap.count(oldArg)) {
                        auto ptrValue = offsetToCastMap[oldArg].getResult();
                        mapper.map(ptrValue, newArg);
                    }
                }
                
                // 克隆循环体中的操作
                for (auto &op : forOp.getRegion().front()) {
                    // 跳过要移除的操作
                    if (opsToRemove.contains(&op)){
                        continue;
                    }

                    // 如果是加法操作，检查是否是offset + increment形式
                    if(auto addOp = dyn_cast<arith::AddIOp>(op)){
                        ftm::AddaOp basePtrAddOp = nullptr;
                        for (auto &entry : offsetToOffsetAddMap) {
                            if (entry.second == addOp) {
                                // 使用Adda替换Addi
                                basePtrAddOp = nestedBuilder.create<ftm::AddaOp>(op.getLoc(), 
                                    mapper.lookup(entry.first), 
                                    entry.second.getRhs());
                                mapper.map(entry.second.getResult(), basePtrAddOp.getResult());
                                break;
                            }
                        }
                        if(basePtrAddOp)
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