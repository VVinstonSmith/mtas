//===--------- LoopStrengthReduce.cpp - loop strength reduce Pass ---------===//
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
#define GEN_PASS_DEF_LOOPSTRENGTHREDUCE
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

// 存储所有需要优化的乘法操作及其常量值
struct MulInfo {
  arith::MulIOp mulOp;
  Value constOperand;
  int64_t multiplier;
};

void implLoopStrengthReduce(func::FuncOp funcOp) {
  // 遍历函数中所有的scf::forOp
  funcOp.walk([&](scf::ForOp forOp) {
    // 获取forOp的迭代变量和步长
    OpBuilder builder(forOp);
    Value inductionVar = forOp.getInductionVar();
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();

    // 检查循环下界和步长是否是常量
    auto lowerBoundConstOp = lowerBound.getDefiningOp<arith::ConstantOp>();
    if (!lowerBoundConstOp)
      return;
    auto stepConstOp = step.getDefiningOp<arith::ConstantOp>();
    if (!stepConstOp)
      return;

    // 提取循环下界和步长值
    auto lowerBoundAttr = lowerBoundConstOp.getValue().cast<IntegerAttr>();
    int64_t lowerBoundVal = lowerBoundAttr.getInt();
    auto stepConstAttr = stepConstOp.getValue().cast<IntegerAttr>();
    int64_t stepVal = stepConstAttr.getInt();

    SmallVector<MulInfo, 4> mulOpsToOptimize;

    // 收集归纳变量及其转换后的值
    SmallVector<Value, 4> inductionVars;
    inductionVars.push_back(inductionVar);
    
    // 寻找归纳变量的所有类型转换
    for (Operation *user : inductionVar.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
        inductionVars.push_back(castOp.getResult(0));
      }
    }

    // 遍历所有归纳变量及其转换后值的使用者
    for (Value indVar : inductionVars) {
      for (Operation *user : indVar.getUsers()) {
        // 判断是否是arith.muli操作且另一个操作数是常量
        if (auto mulOp = dyn_cast<arith::MulIOp>(user)) {
          Value lhs = mulOp.getLhs();
          Value rhs = mulOp.getRhs();
          
          // 确定哪个是归纳变量，哪个是常量
          Value constOperand;
          if (lhs == indVar && rhs.getDefiningOp<arith::ConstantOp>()) {
            constOperand = rhs;
          } else if (rhs == indVar && lhs.getDefiningOp<arith::ConstantOp>()) {
            constOperand = lhs;
          } else {
            // 不是常量乘以归纳变量的形式，跳过
            continue;
          }
        
        // 检查常量类型是否为i64
        auto constDefOp = constOperand.getDefiningOp<arith::ConstantOp>();
        auto constAttr = constDefOp.getValue().cast<IntegerAttr>();
        if (constAttr.getType().isInteger(64)) {
          // 记录乘法信息
          mulOpsToOptimize.push_back({mulOp, constOperand, constAttr.getInt()});
        }
      }
    }
    }

    // 如果没有找到可优化的乘法操作，直接返回
    if (mulOpsToOptimize.empty())
      return;

    // 创建所有乘法操作的初始值和增量值
    builder.setInsertionPoint(forOp);
    SmallVector<Value, 4> iterArgs;

    // 为每个需要优化的乘法操作创建对应的初始值
    for (auto &mulInfo : mulOpsToOptimize) {
      // 计算初始值 = 下界 * 乘法常量
      int64_t initialVal = lowerBoundVal * mulInfo.multiplier;
      
      // 创建初始值常量
      auto constDefOp = mulInfo.constOperand.getDefiningOp<arith::ConstantOp>();
      auto constAttr = constDefOp.getValue().cast<IntegerAttr>();
      auto initConstant = builder.create<arith::ConstantOp>(
          forOp.getLoc(),
          constAttr.getType(),
          builder.getIntegerAttr(constAttr.getType(), initialVal));
      
      iterArgs.push_back(initConstant.getResult());
    }

    // 创建新的forOp，使用所有累加变量作为迭代参数
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        lowerBound,
        upperBound,
        step,
        iterArgs,
        [&](OpBuilder &nestedBuilder, Location loc, Value newIV, ValueRange loopVars) {
          // 建立旧值到新值的映射
          IRMapping mapping;
          mapping.map(inductionVar, newIV);
          
          // 将每个需要优化的乘法操作映射到对应的迭代参数
          for (size_t i = 0; i < mulOpsToOptimize.size(); ++i) {
            mapping.map(mulOpsToOptimize[i].mulOp.getResult(), loopVars[i]);
          }
          
          // 克隆循环体中的所有操作，除了terminator（yield）
          for (auto &op : forOp.getBody()->without_terminator()) {
            bool skipOp = false;
            
            // 检查是否是需要跳过的乘法操作
            for (auto &mulInfo : mulOpsToOptimize) {
              if (&op == mulInfo.mulOp.getOperation()) {
                skipOp = true;
                break;
              }
            }
            
            if (!skipOp) {
              nestedBuilder.clone(op, mapping);
            }
          }
          
          // 记录每个累加变量更新后的值
          SmallVector<Value, 4> newYieldValues;
          // 为每个累加变量创建一个更新操作
          for (size_t i = 0; i < mulOpsToOptimize.size(); ++i) {
            // 计算增量值 = 步长 * 乘法常量
            int64_t incrementVal = stepVal * mulOpsToOptimize[i].multiplier;
            
            // 创建增量常量
            auto constType = mulOpsToOptimize[i].constOperand.getType();
            auto incrementConstant = nestedBuilder.create<arith::ConstantOp>(
                loc,
                constType,
                nestedBuilder.getIntegerAttr(constType, incrementVal));
            
            // 创建加法操作，更新累加变量
            auto updatedValue = nestedBuilder.create<arith::AddIOp>(
                loc,
                loopVars[i],
                incrementConstant);
            
            newYieldValues.push_back(updatedValue);
          }
          
          // 创建新的yield操作
          nestedBuilder.create<scf::YieldOp>(loc, newYieldValues);
        });

    // 移除原循环
    forOp.erase();
  });
}

} // namepsace

namespace mlir {
class LoopStrengthReducePass : public impl::LoopStrengthReduceBase<LoopStrengthReducePass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implLoopStrengthReduce(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLoopStrengthReducePass() {
  return std::make_unique<LoopStrengthReducePass>();
}