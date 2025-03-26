//===--------- CastPtrToInt64.cpp - cast ptr to int63 Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_CASTPTRTOINT64
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

void castFuncArgPtrToI64(func::FuncOp funcOp, int64_t argIdx) {
  auto loc = funcOp.getLoc();
  auto ctx = funcOp.getContext();
  auto argVal_ptr = funcOp.getArgument(argIdx);
  OpBuilder builder(ctx);
  
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  auto argVal_i64 = 
      builder.create<ftm::CastOp>(loc, builder.getI64Type(), argVal_ptr);

  for(auto userOp : argVal_ptr.getUsers()) {
    if(auto gepOp = dyn_cast<LLVM::GEPOp>(userOp)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(gepOp);
      auto offsetBytes = builder.create<arith::MulIOp>(loc,
          gepOp.getOperand(1),
          builder.create<arith::ConstantIntOp>(loc, 4, builder.getI64Type()));
      auto AddOp = builder.create<arith::AddIOp>(loc, argVal_i64, offsetBytes);

      // auto AddOp = builder.create<arith::AddIOp>(loc, argVal_i64, gepOp.getOperand(1));
      auto result_ptr = builder.create<ftm::CastOp>(loc, 
          builder.getType<LLVM::LLVMPointerType>(), AddOp.getResult());
      gepOp.getResult().replaceAllUsesWith(result_ptr.getResult());
      gepOp.erase();
    }
  }
}

void implCastPtrToInt64(func::FuncOp funcOp) {
  for(int64_t argIdx = 0; argIdx < funcOp.getNumArguments(); argIdx++) {
    if(funcOp.getArgument(argIdx).getType().isa<LLVM::LLVMPointerType>()) {
      castFuncArgPtrToI64(funcOp, argIdx);
    }
  }
}

} // namepsace

namespace mlir {
class CastPtrToInt64Pass : public impl::CastPtrToInt64Base<CastPtrToInt64Pass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implCastPtrToInt64(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createCastPtrToInt64Pass() {
  return std::make_unique<CastPtrToInt64Pass>();
}