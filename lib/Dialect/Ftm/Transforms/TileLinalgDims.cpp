//===--------- TileLinalgDims.cpp - Tile Linalg Dimensions Pass ---------===//
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
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_TILELINALGDIMS
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

static SmallVector<OpFoldResult> getMixedSizes(OpBuilder& builder,
    SmallVector<int64_t>& staticSizes, SmallVector<Value>& dynamicSizes) {
  SmallVector<OpFoldResult> results;
  results.reserve(staticSizes.size());
  unsigned dynamicPos = 0;
  for (int64_t size : staticSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamicSizes[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

std::pair<SmallVector<Operation*>, SmallVector<Operation*>>
applyTiling(Operation* targetOp,
    SmallVector<int64_t> staticSizes, SmallVector<Value> dynamicSizes) {
  { // discard entirely zero staticSizes
    int noneZeroCnt = 0;
    for(int64_t dimIdx = 0; dimIdx < staticSizes.size(); dimIdx++) {
      if(staticSizes[dimIdx] != 0)
        noneZeroCnt++;
    }
    if(noneZeroCnt == 0)
      return {{targetOp}, {}}; 
  }

  auto ctx = targetOp->getContext();
  IRRewriter rewriter(ctx);
  
  auto tilingInterface = dyn_cast<TilingInterface>(targetOp);
  if (!tilingInterface) {
    llvm::errs() << "only ops implementing TilingInterface are supported";
    return {};
  }
  if (staticSizes.size() > tilingInterface.getLoopIteratorTypes().size()) {
    llvm::errs()
        << "too many tiles provided, expected at most "
        << tilingInterface.getLoopIteratorTypes().size() << " found "
        << staticSizes.size();
    return {};
  }

  scf::SCFTilingOptions tilingOptions;
  if (staticSizes.empty()) {
    tilingOptions.setTileSizeComputationFunction(
        [](OpBuilder &, Operation *) -> SmallVector<OpFoldResult> {
          return {};
        });
  } else {
    tilingOptions.setTileSizeComputationFunction([&](OpBuilder &b, Operation *) {
      return getMixedSizes(b, staticSizes, dynamicSizes);
    });
  }

  FailureOr<scf::SCFTilingResult> maybeTilingResult =
      tileUsingSCFForOp(rewriter, tilingInterface, tilingOptions);
  if (failed(maybeTilingResult)) {
    llvm::errs() << "fail to tile the target op";
    return {};
  }

  rewriter.replaceOp(targetOp, maybeTilingResult->replacements);

  SmallVector<Operation *> tiled = maybeTilingResult->tiledOps;
  SmallVector<Operation *, 4> loops;
  for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
    loops.push_back(en2.value());

  return std::make_pair(tiled, loops);
}

bool implLinalgOpTiling(Operation* op) {
  auto ctx = op->getContext();
  auto linalgOp = cast<linalg::LinalgOp>(op);

  auto memLevelAttrs = op->getAttr(ftm::OperandMemLevelAttr::name);
  if(!memLevelAttrs) {
    llvm::errs() << "Operands of linalgOp need memory level attrs.";
    return false;
  }
  auto OperandMemLevels = 
      memLevelAttrs.cast<ftm::OperandMemLevelAttr>().getMemLevels();

  auto nLoops = linalgOp.getNumLoops();
  SmallVector<int64_t> tileSizes(nLoops, 1);

  // find out vector dimensions
  for(auto [oprIdx, oprIdxMap] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
    if(OperandMemLevels[oprIdx] == ftm::Cache::Unknown)
      continue;
    auto operand = linalgOp->getOperand(oprIdx);
    auto memrefType = operand.getType().cast<MemRefType>();
    if(OperandMemLevels[oprIdx] == Cache::AM ||
        OperandMemLevels[oprIdx] == Cache::VectorRegister)
      tileSizes[oprIdxMap.getDimPosition(memrefType.getRank() - 1)] = 32;
  }

  SmallVector<int64_t> outerTileSizes(nLoops, 0);
  SmallVector<int64_t> innerTileSizes(nLoops, 0);

  // classify outer loops and inner loops
  for(auto [oprIdx, oprIdxMap] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
    if(OperandMemLevels[oprIdx] == ftm::Cache::Unknown)
      continue;
    auto operand = linalgOp->getOperand(oprIdx);
    auto memrefType = operand.getType().cast<MemRefType>();
    for(int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
      auto linalgDimIdx = oprIdxMap.getDimPosition(dimIdx);
      if(memrefType.getDimSize(dimIdx) / tileSizes[linalgDimIdx] < 16) {
        innerTileSizes[linalgDimIdx] = tileSizes[linalgDimIdx];
      } else {
        outerTileSizes[linalgDimIdx] = tileSizes[linalgDimIdx];
      }
    }
  }

  auto [outerTiledOp, outerLoops] = applyTiling(op, outerTileSizes, {});
  auto [innerTiledOp, innerLoops] = applyTiling(outerTiledOp.back(), innerTileSizes, {});

  unsigned loopId = 0;
  for(auto loop : outerLoops) {
    loop->setAttr(ftm::LoopIdAttr::name, ftm::LoopIdAttr::get(ctx, loopId++));
  }
  for(auto loop : innerLoops) {
    loop->setAttr(ftm::LoopIdAttr::name, ftm::LoopIdAttr::get(ctx, loopId++));
  }

  if(auto attr = linalgOp->getAttr(ftm::UnrollLoopNumberAttr::name)) {
    int64_t unrollLoopNumber = attr.cast<ftm::UnrollLoopNumberAttr>().getNumber();
    if(unrollLoopNumber < innerLoops.size()) {
      innerLoops[unrollLoopNumber]->setAttr(ftm::UnrollFactorAttr::name, 
          ftm::UnrollFactorAttr::get(ctx, 2));
    } else if(unrollLoopNumber < innerLoops.size() + outerLoops.size()) {
      outerLoops[unrollLoopNumber - innerLoops.size()]->setAttr(
        ftm::UnrollFactorAttr::name, ftm::UnrollFactorAttr::get(ctx, 2));
    } else {
      llvm::errs() << "unroll loop number is wrong\n";
      return false;
    }
  }
  return true;
}

} // namepsace

namespace mlir {
class TileLinalgDimsPass : public impl::TileLinalgDimsBase<TileLinalgDimsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation* op) {
      if(op->getParentOp() != funcOp)
        return WalkResult::skip();
      
      if(isa<linalg::LinalgOp>(op)) {
        if(isa<linalg::MatmulOp>(op))
          implLinalgOpTiling(op);
        else if(isa<linalg::AddOp>(op))
          implLinalgOpTiling(op);
        else if(isa<linalg::FillOp>(op))
          implLinalgOpTiling(op);
      }

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createTileLinalgDimsPass() {
  return std::make_unique<TileLinalgDimsPass>();
}