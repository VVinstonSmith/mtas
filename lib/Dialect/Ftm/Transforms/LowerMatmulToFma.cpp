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
    Value Val, int64_t idx, int64_t offset, int64_t newSize, OpBuilder& builder) {
  if(auto defOp = Val.getDefiningOp()) {
    if(auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(subview);
      auto loc = defOp->getLoc();
      auto newSizes = subview.getMixedSizes();
      newSizes[idx] = builder.create<arith::ConstantIndexOp>(loc, newSize).getResult();
      int64_t orgOffset = subview.getStaticOffsets()[idx];
      if(orgOffset == 0) {
        auto newPreSubview = getOffsetSubviewFrom(subview.getSource(), idx, offset, newSize, builder);
        if(!newPreSubview)
          return nullptr;
        return builder.create<memref::SubViewOp>(loc, newPreSubview,
            subview.getMixedOffsets(), newSizes, subview.getMixedStrides());  
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
          mixedOffsets, newSizes, subview.getMixedStrides());
    }
  }
  return nullptr;
}

bool implMatmulLowering(Operation* op) {
  auto matmulOp = cast<linalg::MatmulOp>(op);
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
  auto operandMemLevels = 
      memLevelAttrs.cast<ftm::OperandMemLevelAttr>().getMemLevels();

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
    auto memLevel = operandMemLevels[idx];
    if(memLevel == ftm::Cache::AM || memLevel == ftm::Cache::VectorRegister) {
      auto loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem1024BitTy, subviews[idx]);
      loadOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, operandMemLevels[idx]));
      fmaOperands[idx] = loadOp.getResult();
    } else {
      int offset = (unrollSegmentId % 2 == 1) ? -1 : 0;
      auto evenSubview = getOffsetSubviewFrom(subviews[idx], /*idx*/1, offset, /*newSize*/2, builder);
      subviews[idx].replaceAllUsesWith(evenSubview.getOperation());
      subviews[idx] = evenSubview;
      auto loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem64BitTy, subviews[idx]);
      loadOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, operandMemLevels[idx]));
      auto bcastValue = builder.create<ftm::BroadcastOp>(loc, elem1024BitTy, loadOp);
      if(unrollSegmentId % 2 == 1)
        fmaOperands[idx] = builder.create<ftm::Vbale2hOp>(loc, elem1024BitTy, bcastValue);
      else
        fmaOperands[idx] = builder.create<ftm::Vbale2lOp>(loc, elem1024BitTy, bcastValue);
    }
  }

  auto vectorFMA = builder.create<ftm::FMAOp>(loc,
      elem1024BitTy, fmaOperands[0], fmaOperands[1], fmaOperands[2]);

  auto storeValue = builder.create<ftm::MemRefStoreOp>(loc, vectorFMA, subviews[2]);
  storeValue->setAttr(ftm::MemLevelAttr::name,
      MemLevelAttr::get(ctx, operandMemLevels[2]));
  
  matmulOp.erase();
  return true;
}

bool implAddOpLowering(Operation *op) {
  auto addOp = cast<linalg::AddOp>(op);
  auto loc = addOp.getLoc();
  auto ctx = addOp.getContext();
  auto funcOp = addOp->getParentOfType<func::FuncOp>();
  OpBuilder builder(ctx);

  auto memLevelAttrs = addOp->getAttr(ftm::OperandMemLevelAttr::name);
  if(!memLevelAttrs) {
    llvm::errs() << "Operands of linalgOp need memory level attrs.\n";
    return false;
  }
  auto operandMemLevels = 
      memLevelAttrs.cast<ftm::OperandMemLevelAttr>().getMemLevels();

  SmallVector<Value, 3> subMats(3);
  SmallVector<memref::SubViewOp, 3> subviews(3);

  for(auto [idx, operand] : llvm::enumerate(addOp.getOperands())) {
    subMats[idx] = addOp.getOperand(idx);
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

  builder.setInsertionPoint(addOp);
  Type elem64BitTy = VectorType::get({2}, builder.getF32Type());
  Type elem1024BitTy = VectorType::get({32}, builder.getF32Type());

  SmallVector<Value, 2> addOpOperands(2);
  for(auto [idx, operand] : llvm::enumerate(addOp.getInputs())) {
    auto memLevel = operandMemLevels[idx];
    if(memLevel == ftm::Cache::AM || memLevel == ftm::Cache::VectorRegister) {
      auto loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem1024BitTy, subviews[idx]);
      loadOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, operandMemLevels[idx]));
      addOpOperands[idx] = loadOp.getResult();
    } else { // SM or ScalarRegister
      auto evenSubview = getOffsetSubviewFrom(
          subviews[idx], /*idx*/1, /*offset*/0, /*newSize*/2, builder);
      subviews[idx].replaceAllUsesWith(evenSubview.getOperation());
      subviews[idx] = evenSubview;
      auto loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem64BitTy, subviews[idx]);
      loadOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, operandMemLevels[idx]));
      addOpOperands[idx] = loadOp.getResult();
    }
  }

  Type decOutputType;
  ftm::Cache registerLevel;
  int64_t registerId;
  if(operandMemLevels[2] == ftm::Cache::AM ||
      operandMemLevels[2] == ftm::Cache::VectorRegister) {
    decOutputType = elem1024BitTy;
    registerLevel = ftm::Cache::VectorRegister;
    registerId = 63;
  } else {
    decOutputType = elem64BitTy;
    registerLevel = ftm::Cache::ScalarRegister;
    registerId = 61;
  }
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  auto declareOp = builder.create<ftm::DeclareRegisterOp>(loc, decOutputType);
  declareOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, registerLevel));
  declareOp->setAttr(ftm::RegisterIdAttr::name, RegisterIdAttr::get(ctx, registerId)); 
  
  Operation* moviOp = (registerLevel == ftm::Cache::VectorRegister) ?
      builder.create<ftm::MoviOp>(loc, decOutputType,
          builder.getI64IntegerAttr(0x3F8000003F800000), declareOp) :
      builder.create<ftm::SmoviOp>(loc, decOutputType,
          builder.getI64IntegerAttr(0x3F8000003F800000), declareOp);    
  Value cstOneOperand = moviOp->getResult(0);
  
  builder.setInsertionPoint(addOp);
  auto fmaOp = builder.create<ftm::FMAOp>(loc,
      decOutputType, addOpOperands[0], addOpOperands[1], cstOneOperand);

  auto storeValue = builder.create<ftm::MemRefStoreOp>(loc, 
      fmaOp.getResult(), subviews[2]);
  storeValue->setAttr(ftm::MemLevelAttr::name,
      MemLevelAttr::get(ctx, operandMemLevels[2]));

  addOp.erase();
  return true;
}

bool implFillOpLowering(Operation *op) {
  auto fillOp = cast<linalg::FillOp>(op);
  auto loc = fillOp.getLoc();
  auto ctx = fillOp.getContext();
  auto funcOp = fillOp->getParentOfType<func::FuncOp>();
  OpBuilder builder(ctx);

  auto memLevelAttrs = fillOp->getAttr(ftm::OperandMemLevelAttr::name);
  if(!memLevelAttrs) {
    llvm::errs() << "Operands of linalgOp need memory level attrs.\n";
    return false;
  }
  auto memLevel = memLevelAttrs.cast<
      ftm::OperandMemLevelAttr>().getMemLevels()[1];
  
  union {
    float constF32[2];
    int64_t constI64;
  } constSrc;

  if(auto defSrcOp = fillOp.getInputs()[0].getDefiningOp()) {
    if(auto constOp = dyn_cast<arith::ConstantOp>(defSrcOp)) {
      constSrc.constF32[0] = constSrc.constF32[1] =
          constOp.getValue().cast<FloatAttr>().getValueAsDouble();
    }
  }

  Value dstMat = fillOp.getOutputs()[0];
  memref::SubViewOp subview;
  if(auto defDstOp = dstMat.getDefiningOp()) {
    if(!defDstOp) {
      llvm::errs() << "fill destiny must be the result of memref.subview";
      return false;
    }
    subview = dyn_cast<memref::SubViewOp>(defDstOp);
    if(!subview) {
      llvm::errs() << "matmul operands must be the result of memref.subview";
      return false;
    }
  }

  builder.setInsertionPoint(fillOp);
  Type elem64BitTy = VectorType::get({2}, builder.getF32Type());
  Type elem1024BitTy = VectorType::get({32}, builder.getF32Type());

  ftm::MemRefLoadOp loadOp;
  Operation* moviOp;
  if(memLevel == ftm::Cache::AM || memLevel == ftm::Cache::VectorRegister) {
    loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem1024BitTy, subview);
    moviOp =  builder.create<ftm::MoviOp>(loc, elem1024BitTy,
        builder.getI64IntegerAttr(constSrc.constI64), loadOp.getResult());
  } else { // SM or ScalarRegister
    auto evenSubview = getOffsetSubviewFrom(
        subview, /*idx*/1, /*offset*/0, /*newSize*/2, builder);
    subview.replaceAllUsesWith(evenSubview.getOperation());
    subview = evenSubview;
    loadOp = builder.create<ftm::MemRefLoadOp>(loc, elem64BitTy, subview);
    moviOp =  builder.create<ftm::SmoviOp>(loc, elem1024BitTy,
        builder.getI64IntegerAttr(constSrc.constI64), loadOp.getResult());
  }
  loadOp->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, memLevel));

  builder.setInsertionPoint(fillOp);
  auto storeValue = builder.create<ftm::MemRefStoreOp>(loc, moviOp->getResult(0), subview);
  storeValue->setAttr(ftm::MemLevelAttr::name, MemLevelAttr::get(ctx, memLevel));

  fillOp.erase();
  return true;
}

} // namepsace

namespace mlir {
class LowerMatmulToFmaPass : public impl::LowerMatmulToFmaBase<LowerMatmulToFmaPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation *op) {
      if(op->getParentOfType<func::FuncOp>() != funcOp)
        return WalkResult::skip();
      
      if(isa<linalg::MatmulOp>(op)) {
        implMatmulLowering(op);
      } else if(isa<linalg::AddOp>(op)) {
        implAddOpLowering(op);
      } else if(isa<linalg::FillOp>(op)) {
        implFillOpLowering(op);
      }

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLowerMatmulToFmaPass() {
  return std::make_unique<LowerMatmulToFmaPass>();
}