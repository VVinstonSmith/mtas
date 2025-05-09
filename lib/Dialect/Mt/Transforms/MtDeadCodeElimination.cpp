//===- MtDeadCodeElimination.cpp - Eliminate dead code in Mt dialect
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to eliminate dead code in the Mt dialect.
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
#define GEN_PASS_DEF_MTDEADCODEELIMINATION
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class MtDeadCodeEliminationPass
    : public impl::MtDeadCodeEliminationBase<MtDeadCodeEliminationPass> {
 public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implMtDeadCodeElimination(funcOp);
  }

 private:
  void implMtDeadCodeElimination(func::FuncOp funcOp) {
    // 遍历mt.declare_register
    funcOp.walk([&](mt::DeclareRegisterOp declareRegisterOp) {
      // 遍历它的使用者
      auto reg = declareRegisterOp.getResult();
      for (Operation *user : reg.getUsers()) {
        auto op = dyn_cast<InstructionSchedulingInterface>(user);
        if (op == nullptr) continue;

        auto readRegs = op.getReadRegisters();
        for (auto readReg : readRegs) {
          // 如果存在对reg的读操作，跳过
          if (readReg == reg) return;
        }
      }

      // 没有使用者或仅存在对reg的写操作
      // 删除mt.declare_register和它的所有使用者
      for (Operation *user : reg.getUsers()) {
        user->erase();
      }
      declareRegisterOp.erase();
    });
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createMtDeadCodeEliminationPass() {
  return std::make_unique<MtDeadCodeEliminationPass>();
}