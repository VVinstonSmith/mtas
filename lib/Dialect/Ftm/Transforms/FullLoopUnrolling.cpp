//===- FullLoopUnrolling.cpp - Full loop unrolling pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to fully unroll loops without unroll_factor attribute.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_FULLLOOPUNROLLING
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

bool applyFullLoopUnrolling(scf::ForOp loopOp) {
  auto loc = loopOp.getLoc();
  auto ctx = loopOp.getContext();
  OpBuilder builder(ctx);

  // 检查循环是否有 ftm.loop_id 属性
  if (!loopOp->hasAttr("ftm.loop_id")) {
    return false;
  }

  // 检查循环是否有 ftm.unroll_factor 属性，如果有则跳过
  if (loopOp->hasAttr(ftm::UnrollFactorAttr::name)) {
    return false;
  }

  // 获取循环的边界和步长，必须是编译时常量
  int64_t lowerBound, upperBound, loopStep;
  
  if (auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getLowerBound().getDefiningOp())) {
    lowerBound = cstOp.value();
  } else {
    return false;
  }
  
  if (auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getUpperBound().getDefiningOp())) {
    upperBound = cstOp.value();
  } else {
    return false;
  }
  
  if (auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getStep().getDefiningOp())) {
    loopStep = cstOp.value();
  } else {
    return false;
  }

  // 计算迭代次数，避免过度展开
  int64_t iterationCount = (upperBound - lowerBound + loopStep - 1) / loopStep;
  if (iterationCount <= 0) {
    // 如果迭代次数为0或负数，直接删除循环
    loopOp.erase();
    return true;
  }

  // 可选：限制最大展开次数以避免代码膨胀
  // if (iterationCount > 64) {
  //   return false;
  // }

  builder.setInsertionPoint(loopOp);

  // 对每个迭代进行展开
  for (auto pos = lowerBound; pos < upperBound; pos += loopStep) {
    // 创建当前迭代的归纳变量常量
    auto inductVar = builder.create<arith::ConstantIndexOp>(loc, pos);
    
    // 为当前迭代创建独立的映射
    IRMapping iterationMapping;
    iterationMapping.map(loopOp.getInductionVar(), inductVar);

    // 遍历循环体中的所有操作并复制
    loopOp.walk([&](Operation *op) {
      // 只处理直接属于当前循环的操作
      if (op->getParentOp() != loopOp) {
        return WalkResult::skip();
      }
      
      // 跳过 yield 操作
      if (isa<scf::YieldOp>(op)) {
        return WalkResult::interrupt();
      }
      
      // 克隆操作，使用当前迭代的映射
      auto newOp = builder.clone(*op, iterationMapping);
      
      // 更新映射：将原操作的结果映射到新操作的结果
      for (auto [oldResult, newResult] : 
           llvm::zip(op->getResults(), newOp->getResults())) {
        iterationMapping.map(oldResult, newResult);
      }
      
      return WalkResult::advance();
    });
  }

  // 删除原始循环
  loopOp.erase();
  return true;
}

} // namespace

namespace mlir {

class FullLoopUnrollingPass : 
    public impl::FullLoopUnrollingBase<FullLoopUnrollingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    // 记录是否有循环被展开
    bool hasUnrolled = false;
    
    // 使用迭代方式从内到外展开，确保最内层循环优先处理
    bool hasChanges = true;
    while (hasChanges) {
      hasChanges = false;
      
      // 收集所有符合条件的循环，按嵌套深度排序（内层优先）
      SmallVector<std::pair<scf::ForOp, unsigned>> loopsWithDepth;
      
      funcOp.walk([&](scf::ForOp loopOp) {
        // 检查是否符合展开条件
        if (loopOp->hasAttr("ftm.loop_id") && 
            !loopOp->hasAttr(ftm::UnrollFactorAttr::name)) {
          
          // 计算嵌套深度
          unsigned depth = 0;
          Operation *parent = loopOp->getParentOp();
          while (parent && !isa<func::FuncOp>(parent)) {
            if (isa<scf::ForOp>(parent)) {
              depth++;
            }
            parent = parent->getParentOp();
          }
          
          loopsWithDepth.push_back({loopOp, depth});
        }
        return WalkResult::advance();
      });
      
      // 按深度排序，深度大的（内层）优先处理
      llvm::sort(loopsWithDepth, [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      
      // 处理最内层的循环
      for (auto &[loopOp, depth] : loopsWithDepth) {
        // 验证循环仍然存在且有效
        if (!loopOp->getParentOp()) {
          continue;
        }
        
        if (applyFullLoopUnrolling(loopOp)) {
          hasUnrolled = true;
          hasChanges = true;
          break; // 处理一个循环后重新扫描
        }
      }
    }
    
    // 如果有循环被展开，运行清理和优化pass
    if (hasUnrolled) {
      auto module = funcOp->getParentOfType<ModuleOp>();
      mlir::PassManager pm(module.getContext());
      
      // 添加标准的清理和优化pass
      pm.addPass(createCSEPass());           // 公共子表达式消除
      pm.addPass(createCanonicalizerPass()); // 规范化
      pm.addPass(createCSEPass());           // 再次CSE
      
      if (failed(pm.run(module))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createFullLoopUnrollingPass() {
  return std::make_unique<FullLoopUnrollingPass>();
}