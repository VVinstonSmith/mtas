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
    // funcOp.walk([&](Vbale2hOp vbale2hOp) {
    //   auto src1 = vbale2hOp.getSrc1();
    //   auto dst = vbale2hOp.getDst();
    //   dst.replaceAllUsesWith(src1);
    // });
    funcOp.walk([&](SvbcastOp svbcastOp) {
      auto svbcastOpDst = svbcastOp.getDst();
      auto delRegOp = svbcastOpDst.getDefiningOp();
      Vbale2Op vbale2Op = nullptr;
      Vbale2hOp vbale2hOp = nullptr;
      for (auto user : delRegOp->getUsers()) {
        if (auto op = dyn_cast<Vbale2Op>(user)) {
          vbale2Op = op;
        } else if (auto op = dyn_cast<Vbale2hOp>(user)) {
          vbale2hOp = op;
        } else if (!isa<SvbcastOp>(user)){
          llvm_unreachable("SvbcastOp的结果存在其他使用");
        }
      }
      if (vbale2Op && vbale2hOp) {
        auto vbale2OpDst = vbale2Op.getDst();
        auto vbale2hOpDst = vbale2hOp.getDst();
        vbale2hOpDst.replaceAllUsesWith(vbale2OpDst);
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createRegisterReuseOptimizationPass() {
  return std::make_unique<RegisterReuseOptimizationPass>();
}