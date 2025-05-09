//===- RegisterReuseOptimization.cpp - Optimize register reuse
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to optimize register usage by reusing registers
// that are no longer needed.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_REGISTERREUSEOPTIMIZATION
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class RegisterReuseOptimizationPass
    : public impl::RegisterReuseOptimizationBase<
          RegisterReuseOptimizationPass> {
 public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implRegisterReuseOptimization(funcOp);
  }

 private:
  void implRegisterReuseOptimization(func::FuncOp funcOp) {
    // 遍历所有的Vbale2h操作
    funcOp.walk([&](Vbale2hOp vbale2hOp) {
      auto src1 = vbale2hOp.getSrc1();
      auto src2 = vbale2hOp.getSrc2();
      auto dst = vbale2hOp.getDst();
      // 将其第三个操作数替换为第一个操作数
      OpBuilder builder(vbale2hOp);
      builder.setInsertionPoint(vbale2hOp);
      builder.create<Vbale2hOp>(vbale2hOp.getLoc(), src1, src2, src1);
      vbale2hOp.erase();
      // 将第三个操作数的所有使用替换为第一个操作数
      dst.replaceAllUsesWith(src1);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createRegisterReuseOptimizationPass() {
  return std::make_unique<RegisterReuseOptimizationPass>();
}