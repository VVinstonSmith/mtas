//===--------- TileDynamicDims.cpp - Tile Dynamic Dimensions Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mtasm/IR/Mtasm.h"
#include "mtas/Dialect/Mtasm/Transforms/Passes.h"

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
#define GEN_PASS_DEF_TILEDYNAMICDIMS
#include "mtas/Dialect/Mtasm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtasm;

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
    SmallVector<int64_t>& staticSizes, SmallVector<Value>& dynamicSizes) {
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

void implDynamicDimsTiling(Operation* op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);

  auto nLoops = linalgOp.getNumLoops();
  SmallVector<int64_t> staticTileSizes(nLoops, 0);

  for(auto [oprIdx, oprIdxMap] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
    auto operand = linalgOp->getOperand(oprIdx);
    auto memrefType = operand.getType().cast<MemRefType>();
    for(int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
      if(!memrefType.isDynamicDim(dimIdx))
        continue;
      auto linalgDimIdx = oprIdxMap.getDimPosition(dimIdx);
      staticTileSizes[linalgDimIdx] = 1;
    }
  }

  SmallVector<Value> dynamicSizes;
  auto [tiledOp, loops] = applyTiling(op, staticTileSizes, dynamicSizes);
}

} // namepsace

namespace mlir {
class TileDynamicDimsPass : public impl::TileDynamicDimsBase<TileDynamicDimsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](Operation* op) {
      if(op->getParentOp() != funcOp)
        return WalkResult::skip();

      if(isa<linalg::LinalgOp>(op)) {
        implDynamicDimsTiling(op);
      }
      
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtasm::createTileDynamicDimsPass() {
  return std::make_unique<TileDynamicDimsPass>();
}