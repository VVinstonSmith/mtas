//===----------------- SplitMatmul.cpp - Split matmul ops -----------------===//
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

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_SPLITMATMUL
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

std::pair<ftm::Matrix, ftm::Cache>
getOperandNameAndMemLevel(Value operand) {
  ftm::Matrix matrixName = ftm::Matrix::MatX;
  ftm::Cache memoryLevel = ftm::Cache::SM;
  if(auto defOp = operand.getDefiningOp()) {
    if(auto attr = defOp->getAttr(ftm::MatrixNameAttr::name))
      matrixName = attr.cast<ftm::MatrixNameAttr>().getMatrix();
    if(auto attr = defOp->getAttr(ftm::MemLevelAttr::name))
      memoryLevel = attr.cast<ftm::MemLevelAttr>().getLevel();
  } else if(auto blockArg = dyn_cast<BlockArgument>(operand)) {
    if(auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      for(auto funcArg : llvm::enumerate(funcOp.getArguments())) {
        if(funcArg.value() == blockArg) {
          if(auto attr = funcOp.getArgAttr(funcArg.index(), ftm::MatrixNameAttr::name))
            matrixName = attr.cast<ftm::MatrixNameAttr>().getMatrix();
          if(auto attr = funcOp.getArgAttr(funcArg.index(), ftm::MemLevelAttr::name))
            memoryLevel = attr.cast<ftm::MemLevelAttr>().getLevel(); 
          break;
        }
      }
    }
  }
  return {matrixName, memoryLevel};
}

bool implMatmulSpliting(linalg::MatmulOp matmulOp) {
  auto loc = matmulOp.getLoc();
  auto ctx = matmulOp.getContext();
  OpBuilder builder(ctx);

  SmallVector<mlir::ftm::Matrix, 3> operandMatNames;
  SmallVector<mlir::ftm::Cache, 3> operandMemLevels;
  for(auto operand : matmulOp.getOperands()) {
    auto [operandMatName, operandMemLevel] = getOperandNameAndMemLevel(operand);
    operandMatNames.push_back(operandMatName);
    operandMemLevels.push_back(operandMemLevel);
  }
  matmulOp->setAttr(ftm::OperandMatrixNameAttr::name,
      ftm::OperandMatrixNameAttr::get(ctx, operandMatNames));
  matmulOp->setAttr(ftm::OperandMemLevelAttr::name,
      ftm::OperandMemLevelAttr::get(ctx, operandMemLevels));

  builder.setInsertionPoint(matmulOp);

  auto matC = matmulOp.getDpsInits()[0];
  auto tmpMemC = builder.create<memref::AllocaOp>(loc, matC.getType().cast<MemRefType>());
  if(operandMemLevels[2] == ftm::Cache::AM) {
    operandMemLevels[2] = ftm::Cache::VectorRegister;
  } else if(operandMemLevels[2] == ftm::Cache::SM) {
    operandMemLevels[2] = ftm::Cache::ScalarRegister;
  }
  tmpMemC->setAttr(ftm::OperandMatrixNameAttr::name,
      ftm::OperandMatrixNameAttr::get(ctx, operandMatNames[2]));
  tmpMemC->setAttr(ftm::MemLevelAttr::name,
      ftm::MemLevelAttr::get(ctx, operandMemLevels[2]));
  matmulOp.setDpsInitOperand(0, tmpMemC);
  matmulOp->setAttr(ftm::OperandMemLevelAttr::name,
      ftm::OperandMemLevelAttr::get(ctx, operandMemLevels));

  auto c0_f32 = builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), builder.getF32Type());
  auto fillZeroOp = builder.create<linalg::FillOp>(loc, ValueRange{c0_f32}, tmpMemC.getResult());
  
  builder.setInsertionPointAfter(matmulOp);
  auto addtionOp = builder.create<linalg::AddOp>(loc, ValueRange{matC, tmpMemC}, matC);
  
}

} // namepsace

namespace mlir {
class SplitMatmulPass : public impl::SplitMatmulBase<SplitMatmulPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](linalg::MatmulOp op) {
      if(op->getParentOp() != funcOp)
        return WalkResult::skip();

      implMatmulSpliting(op);

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createSplitMatmulPass() {
  return std::make_unique<SplitMatmulPass>();
}