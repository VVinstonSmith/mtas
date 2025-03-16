//===--------- LowerLoadAndStoreMemRefToPtr.cpp - LowerLoadAndStoreMemRefToPtr Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOWERLOADANDSTOREMEMREFTOPTR
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

Value getPtrFromSubview(Value src, OpBuilder& builder) {
  auto subview = src.getDefiningOp();
  auto loc = subview->getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(subview);
  Value alignedPtr = builder.create<LLVM::ExtractValueOp>(loc,
      src, ArrayRef<int64_t>{1});
  alignedPtr.dump();

  Value offset = builder.create<LLVM::ExtractValueOp>(loc,
      src, ArrayRef<int64_t>{2});
  offset.dump();
  Value offsetPtr = builder.create<LLVM::GEPOp>(loc,
      builder.getType<LLVM::LLVMPointerType>(),  // ptr type
      builder.getF32Type(),                      // element type
      alignedPtr,                                 // base ptr
      offset,                                     // offset
      /*inbounds=*/true);                         // set inbounds tag
  return offsetPtr;
}

bool implMemRefLowering(func::FuncOp funcOp) {
  auto loc = funcOp.getLoc();
  auto module = funcOp->getParentOfType<ModuleOp>();
  auto ctx = module.getContext();
  OpBuilder builder(ctx);

  funcOp.walk([&](ftm::MemRefLoadOp loadOp) {
    
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(loadOp);
    
    getPtrFromSubview(loadOp.getBase(), builder);

    return WalkResult::advance();
  });

  return true;
}

} // namepsace

namespace mlir {
class LowerLoadAndStoreMemRefToPtrPass
    : public impl::LowerLoadAndStoreMemRefToPtrBase<LowerLoadAndStoreMemRefToPtrPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    implMemRefLowering(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLowerLoadAndStoreMemRefToPtrPass() {
  return std::make_unique<LowerLoadAndStoreMemRefToPtrPass>();
}
