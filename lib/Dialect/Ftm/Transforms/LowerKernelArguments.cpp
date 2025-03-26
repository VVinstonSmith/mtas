//===--------- LowerKernelArguments.cpp - LowerKernelArguments Pass ---------===//
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
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include <iostream>
#include <set>
#include <map>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_LOWERKERNELARGUMENTS
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

int64_t searchFuncArgIdxFromOpOperand(Value operand) {
  if(auto defOp = operand.getDefiningOp()) {
    if(auto subview = dyn_cast<memref::SubViewOp>(defOp))
      return searchFuncArgIdxFromOpOperand(subview.getSource());
  } else if(auto blockArg = dyn_cast<BlockArgument>(operand)) {
    if(auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      for(int64_t argIdx = 0; argIdx < funcOp.getNumArguments(); argIdx++) {
        if(blockArg == funcOp.getArgument(argIdx))
          return argIdx;
      }
    }
  }
  return -1;
}

int64_t dimLenFuncArgNum = 0;
std::map<std::pair<unsigned, unsigned>, unsigned>
dynFuncArgDim_to_newArgIdx; // {oldFuncArgIdx, dimIdx} -> newFuncArgIdx

func::FuncOp implKernelArgumentsLowering(func::FuncOp oldFuncOp) {
  auto loc = oldFuncOp.getLoc();
  auto module = oldFuncOp->getParentOfType<ModuleOp>();
  auto ctx = oldFuncOp.getContext();
  OpBuilder builder(ctx);
  LLVMTypeConverter typeConverter(ctx);
  builder.setInsertionPoint(oldFuncOp);

  oldFuncOp.walk([&](linalg::MatmulOp linalgOp) {

    std::map<unsigned, unsigned> linalgDimIdx_to_newArgIdx;
    std::set<unsigned> dynLinalgDimSet;

    for(auto [oprIdx, oprIdxMap] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
      auto operand = linalgOp->getOperand(oprIdx);
      // find relative function argument
      int64_t funcArgIdx = searchFuncArgIdxFromOpOperand(operand);
      if(funcArgIdx == -1)
        continue;
      // map linalg op dim idx to new funcArg
      auto memrefType = operand.getType().cast<MemRefType>();
      for(int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
        if(memrefType.getDimSize(dimIdx) != ShapedType::kDynamic)
          continue;
        auto linalgDimIdx = oprIdxMap.getDimPosition(dimIdx);
        dynLinalgDimSet.insert(linalgDimIdx);
        if(dynFuncArgDim_to_newArgIdx.count({funcArgIdx, dimIdx})) {
          linalgDimIdx_to_newArgIdx[linalgDimIdx] = 
              dynFuncArgDim_to_newArgIdx.at({funcArgIdx, dimIdx});
        }
      }
    }
    
    for(auto linalgDimIdx : dynLinalgDimSet) {
      if(!linalgDimIdx_to_newArgIdx.count(linalgDimIdx)) {
        linalgDimIdx_to_newArgIdx[linalgDimIdx] = dimLenFuncArgNum++;
      }
    }

    for(auto [oprIdx, oprIdxMap] : llvm::enumerate(linalgOp.getIndexingMapsArray())) {
      auto operand = linalgOp->getOperand(oprIdx);
      // find relative function argument
      int64_t funcArgIdx = searchFuncArgIdxFromOpOperand(operand);
      if(funcArgIdx == -1)
        continue;
      // map linalg op dim idx to new funcArg
      auto memrefType = operand.getType().cast<MemRefType>();
      for(int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
        if(memrefType.getDimSize(dimIdx) != ShapedType::kDynamic)
          continue;
        auto linalgDimIdx = oprIdxMap.getDimPosition(dimIdx);
        auto newArgIdx = linalgDimIdx_to_newArgIdx.at(linalgDimIdx);
        dynFuncArgDim_to_newArgIdx[{funcArgIdx, dimIdx}] = newArgIdx;
      }
    }
    return WalkResult::advance();
  });

  // std::cout<< "dimLenFuncArgNum = "<<dimLenFuncArgNum<<std::endl;
  // for(auto it : dynFuncArgDim_to_newArgIdx) {
  //   auto [oprIdx, dimIdx] = it.first;
  //   std::cout<< "oprIdx=" <<  oprIdx << " dimIdx=" << dimIdx << " newArgIdx=" << it.second << std::endl;
  // }

  /// create input types for new func.func
  auto oldFuncArgs = oldFuncOp.getArguments();
  SmallVector<Type> newInputTypes(dimLenFuncArgNum, builder.getI64Type());
  for(auto [argIdx, oldArg] : llvm::enumerate(oldFuncArgs)) {
    if(oldArg.getType().isa<MemRefType>()) {
      newInputTypes.push_back(builder.getType<LLVM::LLVMPointerType>());
    } else {
      newInputTypes.push_back(oldFuncOp.getArgument(argIdx).getType());
    }
  }

  /// create argument attributes for new func.func
  SmallVector<DictionaryAttr> newArgAttrs(dimLenFuncArgNum);
  for(auto [argIdx, oldArg] : llvm::enumerate(oldFuncArgs)) {
    newArgAttrs.push_back(oldFuncOp.getArgAttrDict(argIdx));
  }

  /// create new func.func
  auto newFuncType = FunctionType::get(ctx, 
      newInputTypes, oldFuncOp.getResultTypes());
  builder.setInsertionPoint(oldFuncOp);
  auto newFuncOp = builder.create<func::FuncOp>(
      loc, oldFuncOp.getSymName().str() + "_pointerized", newFuncType);
  auto entryBlock = newFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  auto newFuncArgs = newFuncOp.getArguments();

  /// add old attrs to the new func.func
  newFuncOp.setAllArgAttrs(newArgAttrs);

  IRMapping funcArgsMapping;
  /// create memref for ptr arguments in new func.func
  for(auto [oldIdx, oldArg] : llvm::enumerate(oldFuncArgs)) {
    auto newIdx = oldIdx + newFuncArgs.size() - oldFuncArgs.size();
    auto newArg = newFuncArgs[newIdx];
    if(!oldArg.getType().isa<MemRefType>()) {
      funcArgsMapping.map(oldArg, newArg);
      continue;
    }
    /// create struct for building memref.
    auto memrefType = oldArg.getType().cast<MemRefType>();
    auto structType = typeConverter.convertType(memrefType);
    auto descriptor = MemRefDescriptor::undef(builder, loc, structType);
    descriptor.setAllocatedPtr(builder, loc, newArg);
    descriptor.setAlignedPtr(builder, loc, newArg);
    descriptor.setOffset(builder, loc, 
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0)));
    /// get the sizes for memref.
    SmallVector<Value, 2> memrefSizes;
    for(int64_t dimIdx = 0; dimIdx < memrefType.getRank(); dimIdx++) {
      auto dimSize = memrefType.getDimSize(dimIdx);
      if(dimSize != ShapedType::kDynamic) {
        memrefSizes.emplace_back(
            builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(dimSize)));
        continue;
      }
      auto lenArgIdx = dynFuncArgDim_to_newArgIdx.at({oldIdx, dimIdx});
      memrefSizes.emplace_back(newFuncOp.getArgument(lenArgIdx));
    }
    /// get the strides for memref.
    SmallVector<Value, 2> memrefStrides = {memrefSizes[1],
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(1))};
    /// set the sizes and strides for memref.
    for(size_t idx = 0; idx < memrefSizes.size(); idx++) {
      descriptor.setSize(builder, loc, idx, memrefSizes[idx]);
      descriptor.setStride(builder, loc, idx, memrefStrides[idx]);
    }
    /// create memref from struct.
    auto newMemref = builder.create<UnrealizedConversionCastOp>(loc,
        memrefType, ValueRange{descriptor}).getResult(0);
    funcArgsMapping.map(oldArg, newMemref);
  }

  /// clone the function body of the old old func.func
  oldFuncOp.walk([&](Operation* op) {
    if(op->getParentOp() != oldFuncOp)
      return WalkResult::skip();
    builder.clone(*op, funcArgsMapping);
    return WalkResult::advance();
  });

  /// erase old device function
  oldFuncOp.setPrivate();
  oldFuncOp.erase();
  return newFuncOp;
}

} // namepsace

namespace mlir {
class LowerKernelArgumentsPass
    : public impl::LowerKernelArgumentsBase<LowerKernelArgumentsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    implKernelArgumentsLowering(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createLowerKernelArgumentsPass() {
  return std::make_unique<LowerKernelArgumentsPass>();
}
