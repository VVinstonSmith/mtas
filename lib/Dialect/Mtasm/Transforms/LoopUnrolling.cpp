//===--------- LoopUnrolling.cpp - Unroll the innermost loop Pass ---------===//
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

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOOPUNROLLING
#include "mtas/Dialect/Mtasm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtasm;

namespace {

scf::ForOp applyUnrolling(scf::ForOp loopOp, unsigned factor) {
  auto loc = loopOp.getLoc();
  auto ctx = loopOp.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(loopOp);
  auto step_x_factor = builder.create<arith::MulIOp>(loc, 
      loopOp.getStep(), 
      builder.create<arith::ConstantIndexOp>(loc, factor));
  auto newLoopOp = builder.create<scf::ForOp>(loc, 
      loopOp.getLowerBound(), loopOp.getUpperBound(), 
      step_x_factor, loopOp.getInitArgs());
  if(auto yieldOp = newLoopOp.getBody()->getTerminator())
    yieldOp->erase();
  auto inductVar = newLoopOp.getInductionVar();
  
  IRMapping mapping;
  for(auto [idx, orgRegionArg] : llvm::enumerate(loopOp.getRegionIterArgs())) {
    mapping.map(orgRegionArg, newLoopOp.getRegionIterArg(idx));
  }
  builder.setInsertionPointToStart(newLoopOp.getBody()); 
  builder.create<mtasm::AnnotateOp>(loc)->setAttr(
        mtasm::UnrollSegmentAttr::name,
        UnrollSegmentAttr::get(newLoopOp->getContext(), 0));
  for(uint64_t nf = 0; nf < factor; nf++) {
    mapping.map(loopOp.getInductionVar(), inductVar);
    loopOp.walk([&](Operation* op){
      if(op->getParentOp() != loopOp)
        return WalkResult::skip();
      if(auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        auto newYieldOp = builder.clone(*op, mapping);
        if(nf == factor - 1)
          return WalkResult::interrupt();
        mapping.clear();
        for(auto [idx, orgRegionArg] : llvm::enumerate(loopOp.getRegionIterArgs())) {
          mapping.map(orgRegionArg, newYieldOp->getOperand(idx));
        }
        newYieldOp->erase();
        builder.create<mtasm::AnnotateOp>(loc)->setAttr(
            mtasm::UnrollSegmentAttr::name,
            UnrollSegmentAttr::get(newLoopOp->getContext(), nf + 1));
        inductVar = builder.create<arith::AddIOp>(loc, inductVar, loopOp.getStep());
        return WalkResult::interrupt();
      }
      auto newOp = builder.clone(*op, mapping);
      newOp->setAttr(mtasm::UnrollSegmentAttr::name,
          UnrollSegmentAttr::get(newLoopOp->getContext(), nf));
      return WalkResult::advance();
    });
  }
  for(auto [idx, retVal] : llvm::enumerate(loopOp.getResults())) {
    retVal.replaceAllUsesWith(newLoopOp.getResult(idx));
  }
  
  newLoopOp->setAttrs(loopOp->getAttrs());
  newLoopOp->setAttr(mtasm::UnrollFactorAttr::name, 
      UnrollFactorAttr::get(newLoopOp->getContext(), factor));
  loopOp.erase();
  return newLoopOp;
}

void unrollInnermostLoop(scf::ForOp loopOp, unsigned unrollingFactor) {
  bool thisIsInnermost = true;
  loopOp.walk([&](scf::ForOp op) {
    if(op->getParentOp() != loopOp)
      return WalkResult::skip();
    thisIsInnermost = false;
    unrollInnermostLoop(op, unrollingFactor);
    return WalkResult::advance();
  });
  if(thisIsInnermost)
    auto unrolledLoop = applyUnrolling(loopOp, unrollingFactor);
}

} // namepsace

namespace mlir {
class LoopUnrollingPass : public impl::LoopUnrollingBase<LoopUnrollingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    unsigned unrollingFactor = 2;
    unrollingFactor = this->unrollingFactor;

    funcOp.walk([&](scf::ForOp loopOp) {
      if(loopOp->getParentOp() != funcOp)
        return WalkResult::skip();

      unrollInnermostLoop(loopOp, unrollingFactor);

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtasm::createLoopUnrollingPass() {
  return std::make_unique<LoopUnrollingPass>();
}