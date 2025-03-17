//===----------------- AllocateOffsetRegisters.cpp - allocate offset registers -----------------===//
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_ALLOCATEOFFSETREGISTERS
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

int64_t vectorOffsetRegisterIdx = 0;
int64_t scalarOffsetRegisterIdx = 8;

ftm::Cache analyzeTipConstantAddOp(arith::AddIOp addiOp) {
  ftm::Cache memLevel = ftm::Cache::Unknown;
  if(auto defOp = addiOp.getRhs().getDefiningOp()) {
    if(!isa<arith::ConstantIntOp>(defOp))
      return ftm::Cache::Unknown;
  } else
    return ftm::Cache::Unknown;
  for(auto userOp : addiOp.getResult().getUsers()) {
    if(!isa<ftm::CastOp>(userOp))
      return ftm::Cache::Unknown;
    auto castOp = cast<ftm::CastOp>(userOp);
    if(!castOp.getType().isa<LLVM::LLVMPointerType>())
      return ftm::Cache::Unknown;
    for(auto ptrUserOp : castOp.getResult().getUsers()) {
      if(!isa<ftm::LoadOp>(ptrUserOp))
        continue;
      auto loadOp = cast<ftm::LoadOp>(ptrUserOp);
      if(!loadOp->getAttr(ftm::MemLevelAttr::name)) {
        llvm::errs() << "there is a ftm.load without memory level\n";
        return ftm::Cache::Unknown;
      }
      auto loadOpMemLevel = loadOp->getAttr(
          ftm::MemLevelAttr::name).cast<ftm::MemLevelAttr>().getLevel();
      if(memLevel != ftm::Cache::Unknown && memLevel != loadOpMemLevel) {
        llvm::errs() << "this ptr has multiple memory level\n";
        return ftm::Cache::Unknown;
      }
      memLevel = loadOpMemLevel;
    }
  }
  return memLevel;
}

bool implOffsetRegisterAllocating(arith::AddIOp addiOp, ftm::Cache memLevel) {
  if(memLevel == ftm::Cache::SM && scalarOffsetRegisterIdx == 16)
    return false;
  if(memLevel == ftm::Cache::AM && vectorOffsetRegisterIdx == 8)
    return false;

  auto loc = addiOp.getLoc();
  auto ctx = addiOp.getContext();
  OpBuilder builder(ctx);

  auto constOp = cast<arith::ConstantIntOp>(addiOp.getRhs().getDefiningOp());
  int64_t constInt = constOp.getValue().cast<IntegerAttr>().getInt();
  int64_t imm = constInt / 2;

  OpBuilder::InsertionGuard guard(builder);
  auto funcOp = addiOp->getParentOfType<func::FuncOp>();
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  auto declareOR = builder.create<ftm::DeclareRegisterOp>(loc, builder.getI64Type());
  declareOR->setAttr(ftm::MemLevelAttr::name,
      ftm::MemLevelAttr::get(ctx, ftm::Cache::OffsetRegister));
  if(memLevel == ftm::Cache::SM) {
    declareOR->setAttr(ftm::RegisterIdAttr::name, 
        ftm::RegisterIdAttr::get(ctx, scalarOffsetRegisterIdx++));
  } else if(memLevel == ftm::Cache::AM) {
    declareOR->setAttr(ftm::RegisterIdAttr::name, 
      ftm::RegisterIdAttr::get(ctx, vectorOffsetRegisterIdx++));
  }
  auto smovi = builder.create<ftm::SmoviOp>(loc,
      builder.getI64Type(), builder.getI64IntegerAttr(imm), declareOR);
  
  for(auto userOp : addiOp.getResult().getUsers()) {
    auto castOp = cast<ftm::CastOp>(userOp);
    for(auto ptrUserOp : castOp.getResult().getUsers()) {
      if(!isa<ftm::LoadOp>(ptrUserOp))
        continue;
      auto loadOp = cast<ftm::LoadOp>(ptrUserOp);
      builder.setInsertionPointAfter(castOp);
      auto newCastOp = builder.create<ftm::CastOp>(loc,
          builder.getType<LLVM::LLVMPointerType>(), addiOp.getLhs());
      auto newLoadOp = builder.create<ftm::LoadOp>(loc,
          loadOp.getType(), newCastOp, smovi.getResult());
      newLoadOp->setAttrs(loadOp->getAttrs());
      loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
    }
  }
  return true;
}

} // namepsace

namespace mlir {
class AllocateOffsetRegistersPass : 
    public impl::AllocateOffsetRegistersBase<AllocateOffsetRegistersPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](arith::AddIOp addiOp) {
      auto memLevel = analyzeTipConstantAddOp(addiOp);
      if(memLevel != ftm::Cache::Unknown)
        implOffsetRegisterAllocating(addiOp, memLevel);
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createAllocateOffsetRegistersPass() {
  return std::make_unique<AllocateOffsetRegistersPass>();
}