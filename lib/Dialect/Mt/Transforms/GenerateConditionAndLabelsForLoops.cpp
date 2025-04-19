//===- GenerateConditionAndLabelsForLoops.cpp - Lower Ftm to Mt dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to lower Ftm dialect operations to Mt dialect.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_GENERATECONDITIONANDLABELSFORLOOPS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class GenerateConditionAndLabelsForLoopsPass : 
    public impl::GenerateConditionAndLabelsForLoopsBase<GenerateConditionAndLabelsForLoopsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implGenerateConditionAndLabelsForLoops(funcOp);
  }
private:
    void implGenerateConditionAndLabelsForLoops(func::FuncOp funcOp){
      auto ctx = funcOp.getContext();
      OpBuilder builder(ctx);
      // 遍历ForOp
      funcOp.walk([&](scf::ForOp forOp) {
        // 在循环的开始插入Label
        builder.setInsertionPointToStart(forOp.getBody());
        builder.create<mt::LabelOp>(forOp.getLoc(), "loop_k");

        // 在函数开始插入寄存器分配操作
        builder.setInsertionPointToStart(&funcOp.getBody().front());
        mt::DeclareRegisterOp regOp = builder.create<mt::DeclareRegisterOp>(forOp.getLoc(), builder.getI64Type());
        regOp->setAttr(ftm::MemLevelAttr::name, 
            ftm::MemLevelAttr::get(builder.getContext(), ftm::Cache::ScalarRegister));
        regOp->setAttr(ftm::RegisterIdAttr::name, 
            ftm::RegisterIdAttr::get(builder.getContext(), 0));
        // 在循环的结束插入：
        //     SADD将下界的寄存器加上步长寄存器
        //     SLT判断下界寄存器是否小于上界寄存器
        //     SBR根据比较结果跳转到Label
        Operation *terminator = forOp.getBody()->getTerminator();
        builder.setInsertionPoint(terminator);
        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value step = forOp.getStep();

        builder.create<mt::SaddOp>(forOp.getLoc(), lb, step, lb);
        builder.create<mt::SltOp>(forOp.getLoc(), lb, ub, regOp.getResult());
        builder.create<mt::SbrLabelOp>(forOp.getLoc(), "loop_k", regOp.getResult());
      });
    }
};

}

std::unique_ptr<Pass> mlir::mt::createGenerateConditionAndLabelsForLoopsPass() {
  return std::make_unique<GenerateConditionAndLabelsForLoopsPass>();
}