//===----------------- LowerFillOps.cpp - lower fill ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOWERFILLOPS
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

// bool examineFillOp(linalg::FillOp fillOp) {
//   auto defSrcOp = fillOp.getInputs()[0].getDefiningOp();
//   if(!defSrcOp || !isa<arith::ConstantOp>(defSrcOp))
//     return false;
  
// }

} // namepsace

namespace mlir {
class LowerFillOpsPass : 
    public impl::LowerFillOpsBase<LowerFillOpsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](linalg::FillOp op) {
      
      

      return WalkResult::advance();
    });
    
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLowerFillOpsPass() {
  return std::make_unique<LowerFillOpsPass>();
}