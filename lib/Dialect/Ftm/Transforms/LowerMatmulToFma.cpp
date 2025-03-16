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

ftm::UnrollSegmentAttr
searchParentOpToGetUnrollSegmentAttr(Operation *op) {
  if(auto attr = op->getAttr(ftm::UnrollSegmentAttr::name))
    return attr.cast<ftm::UnrollSegmentAttr>();
  while(auto parentOp = op->getParentOp()) {
    if(auto attr = parentOp->getAttr(ftm::UnrollSegmentAttr::name))
      return attr.cast<ftm::UnrollSegmentAttr>();
    op = parentOp;
  }
  return nullptr;
}

memref::SubViewOp getOffsetSubviewFrom(
    Value Val, int64_t idx, int64_t offset, OpBuilder& builder) {
  if(auto defOp = Val.getDefiningOp()) {
    if(auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(subview);
      auto loc = defOp->getLoc();
      int64_t orgOffset = subview.getStaticOffsets()[idx];
      if(orgOffset == 0) {
        auto newPreSubview = getOffsetSubviewFrom(subview.getSource(), idx, offset, builder);
        if(!newPreSubview)
          return nullptr;
        return builder.create<memref::SubViewOp>(loc, newPreSubview,
            subview.getMixedOffsets(), subview.getMixedSizes(), subview.getMixedStrides());  
      }
      // create memref.subview with new offset
      auto mixedOffsets = subview.getMixedOffsets();
      if(orgOffset == ShapedType::kDynamic) {
        mixedOffsets[idx] = builder.create<arith::SubIOp>(loc,
            subview.getDynamicOffset(subview.getIndexOfDynamicOffset(idx)),
            builder.create<arith::ConstantIndexOp>(loc, -offset)).getResult();
      } else {
        mixedOffsets[idx] =
            builder.create<arith::ConstantIndexOp>(loc, orgOffset + offset).getResult();
      }
      return builder.create<memref::SubViewOp>(loc, subview.getSource(),
          mixedOffsets, subview.getMixedSizes(), subview.getMixedStrides());
    }
  }
  return nullptr;
}

bool implMatmulLoweringToFma(linalg::MatmulOp matmulOp) {
  auto loc = matmulOp.getLoc();
  auto ctx = matmulOp.getContext();
  OpBuilder builder(ctx);

  auto unrollSegmentAttr = searchParentOpToGetUnrollSegmentAttr(matmulOp);
  if(!unrollSegmentAttr) {
    llvm::errs() << "linalg.matmul needs belong to an unrolling segment.\n";
    return false;
  }
  int64_t unrollSegmentId = unrollSegmentAttr.getSegmentId();

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
  Type elem64BitTy = VectorType::get({2}, builder.getF32Type());
  Type elem1024BitTy = VectorType::get({32}, builder.getF32Type());

  SmallVector<Value, 3> fmaOperands(3);
  for(auto [idx, operand] : llvm::enumerate(matmulOp.getOperands())) {
    auto memLevel = OperandMemLevels[idx];
    if(memLevel == ftm::Cache::AM || memLevel == ftm::Cache::VectorRegister) {
      fmaOperands[idx] = builder.create<ftm::MemRefLoadOp>(loc,
          elem1024BitTy, subviews[idx]);
      
    } else {
      if(unrollSegmentId % 2 == 1) {
        // auto oddMixedOffsets = subviews[idx].getMixedOffsets();
        // assert(oddMixedOffsets[1].dyn_cast<Value>());
        // auto evenColOffset = builder.create<arith::SubIOp>(loc,
        //     cast<Value>(oddMixedOffsets[1]),
        //     builder.create<arith::ConstantIndexOp>(loc, 1));
        // SmallVector<OpFoldResult, 2> evenMixedOffsets{oddMixedOffsets[0], 
        //                                               evenColOffset.getResult()};
        // auto evenSubview = builder.create<memref::SubViewOp>(loc,
        //     subviews[idx].getSource(), evenMixedOffsets,
        //     subviews[idx].getMixedSizes(), subviews[idx].getMixedStrides());
        
        auto evenSubview = 
            getOffsetSubviewFrom(subviews[idx], /*idx*/1, /*offset*/-1, builder);
        
        subviews[idx].replaceAllUsesWith(evenSubview.getOperation());
        subviews[idx] = evenSubview;
      }
      auto doubleF32Value = builder.create<ftm::MemRefLoadOp>(loc, elem64BitTy, subviews[idx]);
      auto bcastValue = builder.create<ftm::BroadcastOp>(loc, elem1024BitTy, doubleF32Value);
      if(unrollSegmentId % 2 == 1)
        fmaOperands[idx] = builder.create<ftm::Vbale2hOp>(loc, elem1024BitTy, bcastValue);
      else
        fmaOperands[idx] = builder.create<ftm::Vbale2lOp>(loc, elem1024BitTy, bcastValue);
    }
    fmaOperands[idx].getDefiningOp()->setAttr(ftm::MemLevelAttr::name,
      MemLevelAttr::get(ctx, OperandMemLevels[idx]));
  }

  auto vectorFMA = builder.create<ftm::FMAOp>(loc,
      elem1024BitTy, fmaOperands[0], fmaOperands[1], fmaOperands[2]);

  auto storeValue = builder.create<ftm::MemRefStoreOp>(loc, vectorFMA, subviews[2]);
  storeValue->setAttr(ftm::MemLevelAttr::name,
      MemLevelAttr::get(ctx, OperandMemLevels[2]));
  
  matmulOp.erase();

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