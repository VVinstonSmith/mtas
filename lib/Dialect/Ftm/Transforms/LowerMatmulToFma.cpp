//===----------------- LowerMatmulToFma.cpp - lower matmul to fma ops -----------------===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOWERMATMULTOFMA
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

bool implMatmulLoweringToFma(linalg::MatmulOp matmulOp) {
  auto loc = matmulOp.getLoc();
  auto ctx = matmulOp.getContext();
  OpBuilder builder(ctx);

  auto memLevelAttrs = matmulOp->getAttr(ftm::OperandMemLevelAttr::name);
  if(!memLevelAttrs) {
    llvm::errs() << "Operands of linalgOp need memory level attrs.\n";
    return false;
  }
  auto OperandMemLevels = 
      memLevelAttrs.cast<ftm::OperandMemLevelAttr>().getMemLevels();

  // ensure the length of each dimension of operands is one
  // for (auto operand : matmulOp.getOperands()) {
  //   auto memrefType = operand.getType().cast<MemRefType>();
  //   for (int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
  //     if (memrefType.getDimSize(dimIdx) != 1) {
  //       llvm::errs() << "matmul operand shape length must be 1\n";
  //       return false;
  //     }
  //   }
  // }
  // Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc,
  //       memrefDescriptor, ArrayRef<int64_t>{1});
  //   Value offset = rewriter.create<LLVM::ExtractValueOp>(loc,
  //       memrefDescriptor, ArrayRef<int64_t>{2});
  //   Value offsetPtr = rewriter.create<LLVM::GEPOp>(loc,
  //       rewriter.getType<LLVM::LLVMPointerType>(),  // ptr type
  //       rewriter.getF32Type(),                      // element type
  //       alignedPtr,                                 // base ptr
  //       offset,                                     // offset
  //       /*inbounds=*/true);                         // set inbounds tag
  
  SmallVector<Value, 3> subMats(3);
  SmallVector<memref::SubViewOp, 3> subviews(3);

  for(auto [idx, operand] : llvm::enumerate(matmulOp.getOperands())) {
    subMats[idx] = matmulOp.getOperand(idx);
    auto defOp = subMats[idx].getDefiningOp();
    if(!defOp) {
      llvm::errs() << "matmul operands must be the result of memref.subview";
      return false;
    }
    subviews[idx] = dyn_cast<memref::SubViewOp>(defOp);
    if(!subviews[idx]) {
      llvm::errs() << "matmul operands must be the result of memref.subview";
      return false;
    }
  }

  builder.setInsertionPoint(matmulOp);
  auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Type elem64BitTy = VectorType::get({2}, builder.getF32Type());
  Type elem1024BitTy = VectorType::get({32}, builder.getF32Type());

  SmallVector<Value, 3> fmaOperands(3);
  for(auto [idx, operand] : llvm::enumerate(matmulOp.getOperands())) {
    auto memLevel = OperandMemLevels[idx];
    if(memLevel == ftm::Cache::AM || memLevel == ftm::Cache::VectorRegister) {
      fmaOperands[idx] = builder.create<mlir::vector::LoadOp>(loc,
          elem1024BitTy, subviews[idx], ValueRange{c0, c0});
    } else {
      auto doubleF32Value = builder.create<mlir::vector::LoadOp>(loc,
          elem64BitTy, subviews[idx], ValueRange{c0, c0});
      auto bcastValue = builder.create<ftm::BroadcastOp>(loc, elem1024BitTy, doubleF32Value);
      fmaOperands[idx] = builder.create<ftm::Vbale2hOp>(loc, elem1024BitTy, bcastValue);
      // auto columIdx = subviews[idx].getOffsets()[1];
    }
  }
  auto vectorFMA = builder.create<mlir::vector::FMAOp>(loc, fmaOperands[0], fmaOperands[1], fmaOperands[2]);

  

  return true;
}

} // namepsace

namespace mlir {
class LowerMatmulToFmaPass : public impl::LowerMatmulToFmaBase<LowerMatmulToFmaPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](linalg::MatmulOp op) {
      if(op->getParentOfType<func::FuncOp>() != funcOp)
        return WalkResult::skip();

      implMatmulLoweringToFma(op);

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLowerMatmulToFmaPass() {
  return std::make_unique<LowerMatmulToFmaPass>();
}