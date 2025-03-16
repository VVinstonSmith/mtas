//===--------- LoopFolding.cpp - fold loops with static parameters Pass ---------===//
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
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOOPFOLDING
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

bool applyLoopFolding(scf::ForOp loopOp) {
  auto loc = loopOp.getLoc();
  auto ctx = loopOp.getContext();
  OpBuilder builder(ctx);

  int64_t lowerBound, upperBound, loopStep; 
  if(auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getLowerBound().getDefiningOp()))
    lowerBound = cstOp.value();
  else return false;
  if(auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getUpperBound().getDefiningOp()))
    upperBound = cstOp.value();
  else return false;
  if(auto cstOp = dyn_cast<arith::ConstantIndexOp>(
      loopOp.getStep().getDefiningOp()))
    loopStep = cstOp.value();
  else return false;

  if((upperBound - lowerBound) / loopStep > 16)
    return false;

  builder.setInsertionPoint(loopOp);
  IRMapping mapping;

  for(int pos = lowerBound; pos < upperBound; pos += loopStep) {
    auto inductVar = builder.create<arith::ConstantIndexOp>(loc, pos);
    mapping.map(loopOp.getInductionVar(), inductVar);
    loopOp.walk([&](Operation *op) {
      if(op->getParentOp() != loopOp)
        return WalkResult::skip();
      if(isa<scf::YieldOp>(op))
        return WalkResult::interrupt();
      auto newOp = builder.clone(*op, mapping);
      return WalkResult::advance();
    });
  }

  loopOp.erase();
  return true;
}

void eliminateConstantCast(func::FuncOp funcOp) {
  auto loc = funcOp.getLoc();
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  funcOp.walk([&](UnrealizedConversionCastOp op) {
    auto defOp = op.getInputs()[0].getDefiningOp();
    if(!defOp)
      return WalkResult::skip();
    
    int64_t cst;
    if(auto constOp = dyn_cast<arith::ConstantIndexOp>(defOp))
      cst = constOp.value();
    else if(auto constOp = dyn_cast<arith::ConstantIntOp>(defOp))
      cst = constOp.value();
    else 
      return WalkResult::skip();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);

    Value resVal;
    if(op.getResult(0).getType() == builder.getI64Type())
      resVal = builder.create<arith::ConstantIntOp>(loc, cst, builder.getI64Type());
    else if(op.getResult(0).getType() == builder.getIndexType())
      resVal = builder.create<arith::ConstantIndexOp>(loc, cst);
    else
      return WalkResult::skip();
    
    op.getResult(0).replaceAllUsesWith(resVal);

    return WalkResult::advance();
  });
}

/// (a + b) * c -> (a * c) + (b * c) 
void simplifyMuliOp(arith::MulIOp muliOp) {
  auto defOp_lhs = muliOp.getLhs().getDefiningOp();
  auto defOp_rhs = muliOp.getRhs().getDefiningOp();
  if((!defOp_lhs || !isa<arith::AddIOp>(defOp_lhs)) && 
      (!defOp_rhs || !isa<arith::AddIOp>(defOp_rhs)))
    return;
  if(defOp_lhs && isa<arith::AddIOp>(defOp_lhs) && 
      defOp_rhs && isa<arith::AddIOp>(defOp_rhs))
    return;
  Value addi_lhs, addi_rhs, muli_opr;
  if(defOp_lhs)
    if(auto addiOp = dyn_cast<arith::AddIOp>(defOp_lhs)) {
      addi_lhs = addiOp.getLhs();
      addi_rhs = addiOp.getRhs();
      muli_opr = muliOp.getRhs();
    } 
  if(defOp_rhs)
    if(auto addiOp = dyn_cast<arith::AddIOp>(defOp_rhs)) {
      addi_lhs = addiOp.getLhs();
      addi_rhs = addiOp.getRhs();
      muli_opr = muliOp.getLhs();
    }

  auto loc = muliOp.getLoc();
  auto ctx = muliOp.getContext();
  OpBuilder builder(ctx);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(muliOp);

  auto new_lhs = builder.create<arith::MulIOp>(loc, addi_lhs, muli_opr);
  auto new_rhs = builder.create<arith::MulIOp>(loc, addi_rhs, muli_opr);
  auto new_addi = builder.create<arith::AddIOp>(loc, new_lhs, new_rhs);
  muliOp.getResult().replaceAllUsesWith(new_addi.getResult());
}

/// b = cst + a -> b = a + cst
void exchangeConstToLeft(arith::AddIOp op) {
  auto loc = op.getLoc();
  auto ctx = op.getContext();
  OpBuilder builder(ctx);
  if(auto defOp = op.getLhs().getDefiningOp()) {
    if(auto constOp = dyn_cast<arith::ConstantIntOp>(defOp)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(op);
      auto newOp = builder.create<arith::AddIOp>(loc, op.getRhs(), op.getLhs());
      op.getResult().replaceAllUsesWith(newOp.getResult());
    }
  }
}

/// b = a + cst
/// e = b + d
/// -->
/// b1 = a + d
/// e = b1 + cst
void simplifyAddConstant(arith::AddIOp op) {
  auto loc = op.getLoc();
  auto ctx = op.getContext();
  OpBuilder builder(ctx);
  auto implSimplifyConstant = [&](Value operand_1, Value operand_2) {
    if(auto defOp = operand_1.getDefiningOp()) {
      if(auto parentAddOp = dyn_cast<arith::AddIOp>(defOp)) {
        if(auto defDefOp = parentAddOp.getRhs().getDefiningOp()) {
          if(auto constOp = dyn_cast<arith::ConstantIntOp>(defDefOp)) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(op);
            auto newOp_1 = builder.create<arith::AddIOp>(loc,
                parentAddOp.getLhs(), operand_2);
            auto newOp_2 = builder.create<arith::AddIOp>(loc,
                newOp_1.getResult(), parentAddOp.getRhs());
            op.getResult().replaceAllUsesWith(newOp_2.getResult());
          }
        }
      }
    }
  };
  implSimplifyConstant(op.getLhs(), op.getRhs());
  implSimplifyConstant(op.getRhs(), op.getLhs());
}

bool moveUpCastIndexToI64(UnrealizedConversionCastOp castOp) {
  auto loc = castOp.getLoc();
  auto ctx = castOp.getContext();
  OpBuilder builder(ctx);
  if(castOp.getOperandTypes()[0] != builder.getIndexType() ||
      castOp.getResultTypes()[0] != builder.getI64Type())
  return false;
  auto castSrc = castOp.getInputs()[0];
  if(auto defOp = castSrc.getDefiningOp()) {
    /// castSrc is the result of another castOp
    if(auto parentCastOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(parentCastOp);
      Value newRes;
      if(parentCastOp.getOperandTypes()[0] == builder.getI64Type() &&
          parentCastOp.getResultTypes()[0] == builder.getIndexType()) {
        newRes = parentCastOp.getInputs()[0];
      } else {
        newRes = builder.create<UnrealizedConversionCastOp>(loc,
            builder.getI64Type(), parentCastOp.getInputs()[0]).getResult(0);
      }
      castOp.getResult(0).replaceAllUsesWith(newRes);
      castOp.erase();
      return true;
    }
    /// castSrc is the result of a arith op
    SmallVector<Value> newOperands;
    for(auto operand : defOp->getOperands()) {
      if(auto defDefOp = operand.getDefiningOp()) {
        if(auto constOp = dyn_cast<arith::ConstantIndexOp>(defDefOp)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(constOp);
          auto newConstOp = builder.create<arith::ConstantIntOp>(loc,
              constOp.value(), builder.getI64Type());
          newOperands.push_back(newConstOp.getResult());
          continue;
        }
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(defOp);
      auto newCastOp = builder.create<UnrealizedConversionCastOp>(loc,
          builder.getI64Type(), operand);
      newOperands.push_back(newCastOp.getResult(0));
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(defOp);
    Value newRes;
    if(isa<arith::AddIOp>(defOp)) {
      newRes = builder.create<arith::AddIOp>(loc,
          builder.getI64Type(), newOperands);
    } else if(isa<arith::MulIOp>(defOp)) {
      newRes = builder.create<arith::MulIOp>(loc,
        builder.getI64Type(), newOperands);
    }
    castOp.getResult(0).replaceAllUsesWith(newRes);
    castOp.erase();
    return true;
  }
  return false;
}

void runMoveUpCastIndexToI64(func::FuncOp funcOp) {
  int cnt = 1;
  while(cnt != 0) {
    cnt = 0;
    funcOp.walk([&](UnrealizedConversionCastOp op) {
      if(moveUpCastIndexToI64(op))
        cnt++;
      return WalkResult::advance();
    });
  }
}

} // namepsace

namespace mlir {
class LoopFoldingPass : public impl::LoopFoldingBase<LoopFoldingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](scf::ForOp loopOp) {
      applyLoopFolding(loopOp);
      return WalkResult::advance();
    });

    auto module = funcOp->getParentOfType<ModuleOp>();
    mlir::PassManager pm(module.getContext());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.run(module);

    eliminateConstantCast(funcOp);
    pm.run(module);

    funcOp.walk([&](arith::MulIOp muliOp) {
      simplifyMuliOp(muliOp);
      return WalkResult::advance();
    });
    pm.run(module);

    runMoveUpCastIndexToI64(funcOp);

    funcOp.walk([&](arith::AddIOp op) {
      exchangeConstToLeft(op);
      return WalkResult::advance();
    });

    funcOp.walk([&](arith::AddIOp op) {
      simplifyAddConstant(op);
      return WalkResult::advance();
    });
    pm.run(module);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLoopFoldingPass() {
  return std::make_unique<LoopFoldingPass>();
}