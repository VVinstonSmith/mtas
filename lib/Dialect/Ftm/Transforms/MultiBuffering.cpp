//===--------- MultiBuffering.cpp - MultiBuffering Pass -----------------===//
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
#define GEN_PASS_DEF_MULTIBUFFERING
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

static memref::AllocOp traceAllocOp(Value operand) {
  if(auto parentOp = operand.getDefiningOp()) {
    if(auto allocOp = dyn_cast<memref::AllocOp>(parentOp)) {
      if(auto dstAttr = allocOp->getAttr(ftm::MemLevelAttr::name)) {
        if(cast<ftm::MemLevelAttr>(dstAttr).getLevel() != Cache::DDR)
          return allocOp;
      }
    } else if(auto subviewOp = dyn_cast<memref::SubViewOp>(parentOp)) {
      return traceAllocOp(subviewOp.getSource());
    }
  }
  return nullptr;
}

static memref::AllocOp liftUpAllocOp(OpBuilder& builder, memref::AllocOp allocOp);

static memref::AllocOp createMultiBufferAlloc(OpBuilder &builder,
    memref::AllocOp oldAllocOp, int64_t nBuffers) {
  auto oldType = oldAllocOp.getType();
  auto oldShape = oldType.getShape();
  SmallVector<int64_t> newShape = {nBuffers};
  newShape.append(oldShape.begin(), oldShape.end());
  auto newType = MemRefType::get(newShape, oldType.getElementType());
  auto dynSizes = oldAllocOp.getDynamicSizes();
  auto alignAttr = oldAllocOp.getAlignmentAttr();
  auto newAllocOp = liftUpAllocOp(builder, 
      builder.create<memref::AllocOp>(oldAllocOp.getLoc(), 
          newType, dynSizes, alignAttr));
  newAllocOp->setAttrs(oldAllocOp->getAttrs());
  return newAllocOp;
}

static memref::SubViewOp createMultiBufferSubview(OpBuilder &builder,
    Value newSrc, memref::SubViewOp oldSubViewOp, int64_t nBuffers){
  /// new subview params
  SmallVector<OpFoldResult> newOffsets = {builder.getIndexAttr(0)};
  newOffsets.append(oldSubViewOp.getMixedOffsets());
  SmallVector<OpFoldResult> newSizes = {builder.getIndexAttr(nBuffers)};
  newSizes.append(oldSubViewOp.getMixedSizes());
  SmallVector<OpFoldResult> newStrides = {builder.getIndexAttr(1)};
  newStrides.append(oldSubViewOp.getMixedStrides());
  /// create new memref.subview
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPointAfter(oldSubViewOp);
  auto newSubviewOp = builder.create<memref::SubViewOp>(oldSubViewOp.getLoc(), 
      newSrc, newOffsets, newSizes, newStrides);
  return newSubviewOp;
}

static Operation* traceAllocOpAndCreateNew(OpBuilder &builder,
    Value operand, int64_t nBuffers,
    DenseMap<memref::AllocOp, memref::AllocOp> alloc2D_to_alloc3D,
    SetVector<Operation*>& opsToBeErased) {
  if(auto parentOp = operand.getDefiningOp()) {
    opsToBeErased.insert(parentOp);
    if(auto alloc = dyn_cast<memref::AllocOp>(parentOp)) {
      if(alloc.getType().getRank() == 3)
        return alloc;
      if(alloc2D_to_alloc3D.count(alloc))
        return alloc2D_to_alloc3D[alloc];
      auto alloc3d = createMultiBufferAlloc(builder, alloc, nBuffers);
      alloc2D_to_alloc3D[alloc] = alloc3d;
      return alloc3d;
    } else if(auto subviewOp = dyn_cast<memref::SubViewOp>(parentOp)) {
      auto newParentOp = traceAllocOpAndCreateNew(builder, 
          subviewOp.getSource(), nBuffers, alloc2D_to_alloc3D, opsToBeErased);
      auto newSubviewOp = createMultiBufferSubview(builder,
          newParentOp->getResult(0), subviewOp, nBuffers);
      return newSubviewOp;
    }
  }
  return nullptr;
}

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
  strides.back() = 1;
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic, strides));
}

static memref::SubViewOp createDimCutSubview(OpBuilder& builder,
        Operation* op, Value BufferId) {
  if(!isa<memref::AllocOp>(op) && !isa<memref::SubViewOp>(op))
    return nullptr;
  auto memref3dType = op->getResult(0).getType().cast<MemRefType>();
  auto oldShape = memref3dType.getShape().drop_front(1);
  auto oldType = MemRefType::get(oldShape, memref3dType.getElementType());
  /// get offsets, sizes, strides of memref.subview
  SmallVector<OpFoldResult, 2> oldOffsets(oldType.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult, 3> newOffsets = {BufferId};
  newOffsets.append(oldOffsets);
  SmallVector<OpFoldResult, 3> newSizes;
  if(auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
    newSizes = subviewOp.getMixedSizes();
  } else if(auto allocOp = dyn_cast<memref::AllocOp>(op)){
    newSizes = allocOp.getMixedSizes();
  }
  newSizes[0] = builder.getIndexAttr(1);
  SmallVector<OpFoldResult, 3> newStrides(oldType.getRank() + 1, builder.getIndexAttr(1));
  /// create dim-cut memref.subview from op
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPointAfter(op);
  auto dimCutSubview = builder.create<memref::SubViewOp>(
      op->getLoc(), makeStridedLayoutDynamic(oldType), 
      op->getResult(0), newOffsets, newSizes, newStrides);
 return dimCutSubview;
}

static memref::AllocOp liftUpAllocOp(OpBuilder& builder, memref::AllocOp allocOp) {
  Operation* lastProducer = allocOp.getOperand(0).getDefiningOp();
  for(auto operand : allocOp.getOperands()) {
    auto producer = operand.getDefiningOp();
    if(lastProducer->isBeforeInBlock(producer)) {
      lastProducer = producer;
    }
  }
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPointAfter(lastProducer);
  auto newAllocOp = cast<memref::AllocOp>(builder.clone(*allocOp));
  allocOp.getResult().replaceAllUsesWith(newAllocOp.getResult());
  allocOp.erase();
  return newAllocOp;
}

static Value createRemSIOp(Location loc, 
    OpBuilder& builder, Value target, int64_t factor) {
  auto cst = builder.create<arith::ConstantIndexOp>(loc, factor);
  if(factor != 3)
    return builder.create<arith::RemSIOp>(loc, target, cst);
  // target - (target / 3) * 3
  auto tmp1 = builder.create<arith::DivSIOp>(loc, target, cst);
  auto tmp2 = builder.create<arith::MulIOp>(loc, tmp1, cst);
  return builder.create<arith::SubIOp>(loc, target, tmp2);
}

static bool isOperationInStage(Operation* op, ftm::Stage stage) {
  if(auto attr = op->getAttr(ftm::MultiStageAttr::name)) {
    if(attr.cast<ftm::MultiStageAttr>().getStage() == stage)
      return true;
  } return false;
}

scf::ForOp multiBufferize(scf::ForOp loopOp, int level) {
  auto loc = loopOp.getLoc();
  auto ctx = loopOp.getContext();
  OpBuilder builder(ctx);

  DenseMap<Value, int64_t> bufferOpr2nBuffers;

  { // Find memref.alloc from linalg.copy
    SetVector<linalg::CopyOp> readCpyOps, writeCpyOps;

    loopOp.walk([&](linalg::CopyOp copyOp) {
      if(copyOp->getParentOp() != loopOp)
        return WalkResult::skip();
      if(isOperationInStage(copyOp, Stage::Prelogue))
        return WalkResult::skip();
      if(copyOp->getAttr("xxx"))
        return WalkResult::skip();
      auto copySrc = copyOp.getDpsInputs()[0];
      auto copyDst = copyOp.getDpsInits()[0];
      auto srcAlloc = traceAllocOp(copySrc);
      auto dstAlloc = traceAllocOp(copyDst);
      if(srcAlloc && dstAlloc) {
        auto srcMemLevel = cast<ftm::MemLevelAttr>(
            srcAlloc->getAttr(ftm::MemLevelAttr::name)).getLevel();
        auto dstMemLevel = cast<ftm::MemLevelAttr>(
            dstAlloc->getAttr(ftm::MemLevelAttr::name)).getLevel();
        if(srcMemLevel < dstMemLevel) { // gsm -> am,sm
          srcAlloc = nullptr;
        } else { // am,sm -> gsm
          dstAlloc = nullptr;
        }
      }
      if(dstAlloc) { // ddr,gsm -> am,sm
        readCpyOps.insert(copyOp);
        if(!bufferOpr2nBuffers.contains(copyDst))
          bufferOpr2nBuffers[copyDst] = 2;
      } else if(srcAlloc) { // am,sm -> gsm,ddr
        writeCpyOps.insert(copyOp);
        bufferOpr2nBuffers[copySrc] = 3;
      }
      return WalkResult::advance();
    });
    if(readCpyOps.empty() && writeCpyOps.empty())
      return loopOp;
  }

  { // transfrom 2d memref.alloc and memref.subview to 3d version
    DenseMap<memref::AllocOp, memref::AllocOp> alloc2D_to_alloc3D;
    SetVector<Operation*> opsToBeErased;

    builder.setInsertionPointToStart(loopOp.getBody());
    auto absPos = builder.create<arith::SubIOp>(loc,
        loopOp.getInductionVar(), loopOp.getLowerBound());
    auto iterIdx = builder.create<arith::DivSIOp>(loc, absPos, loopOp.getStep());

    for(auto [copyOpr, nBuffers] : bufferOpr2nBuffers) {
      auto bufferId = builder.create<arith::RemSIOp>(loc, iterIdx,
          builder.create<arith::ConstantIndexOp>(loc, nBuffers));
      // auto bufferId = createRemSIOp(loc, builder, iterIdx, nBuffers);
      auto subview3d = traceAllocOpAndCreateNew(builder,
          copyOpr, nBuffers, alloc2D_to_alloc3D, opsToBeErased);
      /// create new memmref.copy
      auto newCopyOpr = createDimCutSubview(builder, subview3d, bufferId);
      copyOpr.replaceAllUsesWith(newCopyOpr);
    }
    for(auto op : opsToBeErased)
      op->erase();
  }
  
  { // create prelogue
    IRMapping prelogueMapping;
    prelogueMapping.map(loopOp.getInductionVar(), loopOp.getLowerBound());
    
    builder.setInsertionPoint(loopOp);
    builder.create<ftm::AnnotateOp>(loc)->setAttr(
        ftm::MultiStageAttr::name,
        MultiStageAttr::get(ctx, Stage::Prelogue));
    
    loopOp.walk([&](Operation *op) {
      if(op->getParentOp() != loopOp)
        return WalkResult::skip();
      if(isOperationInStage(op, Stage::Prelogue))
        return WalkResult::interrupt();
      if(isa<scf::ForOp>(op) ||
          (isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op)))
        return WalkResult::interrupt();
      
      Operation *newOp = builder.clone(*op, prelogueMapping);
      newOp->setAttr(ftm::MultiStageAttr::name, 
          ftm::MultiStageAttr::get(ctx, ftm::Stage::Prelogue));
      return WalkResult::advance();
    });
  }

  /// After create the prelogue, create the main loop.
  auto newLoopOp = builder.create<scf::ForOp>(loc, 
      loopOp.getLowerBound(), loopOp.getUpperBound(), loopOp.getStep());
  newLoopOp->setAttrs(loopOp->getAttrs());
  
  { // create prefetch at the begining of the main loop
    builder.setInsertionPointToStart(newLoopOp.getBody());
    auto posNext = builder.create<arith::AddIOp>(loc,
        newLoopOp.getInductionVar(), loopOp.getStep());
    auto not_last_iter = builder.create<arith::CmpIOp>(loc,
        mlir::arith::CmpIPredicate::slt, posNext, loopOp.getUpperBound());
    auto prefetchIfOp = builder.create<scf::IfOp>(loc, not_last_iter);
    builder.setInsertionPointToStart(prefetchIfOp.getBody());

    IRMapping prefetchMapping;
    prefetchMapping.map(loopOp.getInductionVar(), posNext);

    loopOp.walk([&](Operation *op) {
      if(op->getParentOp() != loopOp)
        return WalkResult::skip();
      if(isOperationInStage(op, Stage::Prelogue))
        return WalkResult::interrupt();
      if(isa<scf::ForOp>(op) ||
          (isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op)))
        return WalkResult::interrupt();

      Operation *newOp = builder.clone(*op, prefetchMapping);
      newOp->setAttr(ftm::MultiStageAttr::name,
          MultiStageAttr::get(ctx, Stage::Prefetch));
      return WalkResult::advance();
    });
    builder.setInsertionPointAfter(prefetchIfOp);
  }

  { // create operations in the main body
    IRMapping bodyMapping;
    bodyMapping.map(loopOp.getInductionVar(), newLoopOp.getInductionVar());

    bool activate = false;
    loopOp.walk([&](Operation *op) {
      if(op->getParentOp() != loopOp || isa<scf::YieldOp>(op))
        return WalkResult::skip();
      if(isa<scf::ForOp>(op) ||
          (isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op)) ||
          isOperationInStage(op, Stage::Prelogue)) {
        activate = true;
      }
      if(isa<linalg::CopyOp>(op)) {
        if(!isOperationInStage(op, Stage::Prelogue) &&
            !op->getAttr("xxx")) {
          if(!activate)
            return WalkResult::skip();
          Operation *newOp = builder.clone(*op, bodyMapping);
          newOp->setAttr(ftm::MultiStageAttr::name,
              MultiStageAttr::get(ctx, Stage::PostStore));
          return WalkResult::skip();
        }
      }
      Operation *newOp = builder.clone(*op, bodyMapping);
      return WalkResult::advance();
    });
  }
  
  loopOp.erase();
  return newLoopOp;
}

void recursiveTraverseForOps(scf::ForOp loopOp, int level){
  scf::ForOp newLoopOp = loopOp;

  newLoopOp = multiBufferize(loopOp, level);

  newLoopOp.walk([&](scf::ForOp loopOp){
    if(loopOp->getParentOp() != newLoopOp)
      return WalkResult::skip();
    recursiveTraverseForOps(loopOp, level + 1);
    return WalkResult::advance();
  });
}

} // namepsace

namespace mlir {
class MultiBufferingPass : public impl::MultiBufferingBase<MultiBufferingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](scf::ForOp loopOp) {
      if(loopOp->getParentOp() != funcOp)
        return WalkResult::skip();
      recursiveTraverseForOps(loopOp, 0 /*level*/);
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createMultiBufferingPass() {
  return std::make_unique<MultiBufferingPass>();
}