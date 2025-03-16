//===- MemRefToPtr.cpp - memref to ptr conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the MtFusion to the MtIVM ops.
//
//===----------------------------------------------------------------------===//

#include "mtas/Conversion/MemRefToPtr/MemRefToPtr.h"
#include "mtas/Conversion/Passes.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOPTR
#include "mtas/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

} // namespace

//===----------------------------------------------------------------------===//
// MemRefLoadOp to LoadOp
//===----------------------------------------------------------------------===//

class MemRefLoadToLoad : public OpConversionPattern<ftm::MemRefLoadOp> {
  using OpConversionPattern<ftm::MemRefLoadOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(ftm::MemRefLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = module.getContext();
    auto src = adaptor.getBase();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    
    Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc,
        src, ArrayRef<int64_t>{1});
    Value offset = rewriter.create<LLVM::ExtractValueOp>(loc,
        src, ArrayRef<int64_t>{2});
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(loc,
        rewriter.getType<LLVM::LLVMPointerType>(), // ptr type
        rewriter.getF32Type(),                     // element type
        alignedPtr,                                // base ptr
        offset,                                    // offset
        /*inbounds=*/true);                        // set inbounds tag
    auto loadOp = rewriter.create<ftm::LoadOp>(loc, op->getResultTypes(), offsetPtr);
    loadOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, loadOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MemRefStoreOp to StoreOp
//===----------------------------------------------------------------------===//

class MemRefStoreToStore : public OpConversionPattern<ftm::MemRefStoreOp> {
  using OpConversionPattern<ftm::MemRefStoreOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(ftm::MemRefStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = module.getContext();
    auto value = adaptor.getValueToStore();
    auto dst = adaptor.getBase();
    
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    
    Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc,
        dst, ArrayRef<int64_t>{1});
    Value offset = rewriter.create<LLVM::ExtractValueOp>(loc,
        dst, ArrayRef<int64_t>{2});
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(loc,
        rewriter.getType<LLVM::LLVMPointerType>(), // ptr type
        rewriter.getF32Type(),                     // element type
        alignedPtr,                                // base ptr
        offset,                                    // offset
        /*inbounds=*/true);                        // set inbounds tag
    auto storeOp = rewriter.create<ftm::StoreOp>(loc, value, offsetPtr);
    storeOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, storeOp);
    return success();
  }
};

void populateLowerMemRefToPtrPattern(
      RewritePatternSet &patterns,
      LLVMTypeConverter &typeConverter) {
  patterns.add<
    MemRefLoadToLoad,
    MemRefStoreToStore
  >(typeConverter, patterns.getContext());
  return;
}

namespace {
struct ConvertMemRefToPtrPass
    : public impl::ConvertMemRefToPtrBase<ConvertMemRefToPtrPass> {
public:
  ConvertMemRefToPtrPass() = default;
  void runOnOperation() override;
};
} // namespace

void ConvertMemRefToPtrPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  OpBuilder builder(ctx);
  auto module = cast<ModuleOp>(getOperation());
  auto loc = module->getLoc();
  target.addLegalDialect<
      BuiltinDialect,
      func::FuncDialect,
      scf::SCFDialect,
      arith::ArithDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect, 
      LLVM::LLVMDialect,
      ftm::FtmDialect>(); 
  target.addIllegalOp<
      ftm::MemRefLoadOp,
      ftm::MemRefStoreOp>();

  RewritePatternSet patterns(ctx);
  LowerToLLVMOptions options(ctx);
  LLVMTypeConverter typeConverter(ctx, options);
  populateLowerMemRefToPtrPattern(patterns, typeConverter);

  /// apply conversion
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createMemRefToPtrConversionPass() {
  return std::make_unique<ConvertMemRefToPtrPass>();
}