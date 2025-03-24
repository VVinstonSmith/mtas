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

int64_t vectorOffsetStartId = 0;
int64_t vectorOffsetEndId = 8;
int64_t scalarOffsetStartId = 8;
int64_t scalarOffsetEndId = 16;

ftm::SmoviOp scanOffsetRegisterInitializerWith(
    func::FuncOp funcOp, ftm::Cache memLevel, unsigned value) {
  ftm::SmoviOp targetOp;
  funcOp.walk([&](ftm::SmoviOp smoviOp) {
    if(auto defOp = smoviOp.getReg().getDefiningOp()) {
      if(auto declareOp = dyn_cast<ftm::DeclareRegisterOp>(defOp)) {
        auto regMemLevel = declareOp->getAttr(
            ftm::MemLevelAttr::name).cast<ftm::MemLevelAttr>().getLevel();
        auto registerId = declareOp->getAttr(
            ftm::RegisterIdAttr::name).cast<ftm::RegisterIdAttr>().getId();
        if(regMemLevel != ftm::Cache::OffsetRegister || smoviOp.getImm() != value)
          return WalkResult::skip();
        if((memLevel == ftm::Cache::AM && 
              registerId >= vectorOffsetStartId && registerId < vectorOffsetEndId) ||
           (memLevel == ftm::Cache::SM && 
              registerId >= scalarOffsetStartId && registerId < scalarOffsetEndId)) {
          targetOp = smoviOp;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return targetOp;
}

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
      if(!isa<ftm::LoadOp>(ptrUserOp) && !isa<ftm::StoreOp>(ptrUserOp))
        continue;
      if(!ptrUserOp->getAttr(ftm::MemLevelAttr::name)) {
        llvm::errs() << "there is a ftm.load without memory level\n";
        return ftm::Cache::Unknown;
      }
      auto loadOpMemLevel = ptrUserOp->getAttr(
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

bool implOffsetRegisterAllocating(arith::AddIOp addiOp, ftm::Cache memLevel,
    int64_t& scalarOffsetRegisterIdx, int64_t& vectorOffsetRegisterIdx) {
  if(memLevel == ftm::Cache::SM && scalarOffsetRegisterIdx >= scalarOffsetEndId)
    return false;
  if(memLevel == ftm::Cache::AM && vectorOffsetRegisterIdx >= vectorOffsetEndId)
    return false;

  auto loc = addiOp.getLoc();
  auto ctx = addiOp.getContext();
  OpBuilder builder(ctx);
  auto funcOp = addiOp->getParentOfType<func::FuncOp>();
  auto constOp = cast<arith::ConstantIntOp>(addiOp.getRhs().getDefiningOp());
  int64_t constInt = constOp.getValue().cast<IntegerAttr>().getInt();
  int64_t imm = constInt / 2;
  
  ftm::SmoviOp smovi = scanOffsetRegisterInitializerWith(funcOp, memLevel, imm);
  if(!smovi) {
    OpBuilder::InsertionGuard guard(builder);
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
    smovi = builder.create<ftm::SmoviOp>(loc,
        builder.getI64Type(), builder.getI64IntegerAttr(imm), declareOR);
  }

  for(auto userOp : addiOp.getResult().getUsers()) {
    auto castOp = cast<ftm::CastOp>(userOp);
    for(auto ptrUserOp : castOp.getResult().getUsers()) {
      if(!isa<ftm::LoadOp>(ptrUserOp) && !isa<ftm::StoreOp>(ptrUserOp))
        continue;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castOp);
      auto newCastOp = builder.create<ftm::CastOp>(loc,
          builder.getType<LLVM::LLVMPointerType>(), addiOp.getLhs());
      builder.setInsertionPointAfter(ptrUserOp);
      if(auto loadOp = dyn_cast<ftm::LoadOp>(ptrUserOp)) {
        auto newLoadOp = builder.create<ftm::LoadOp>(loc,
            loadOp.getType(), newCastOp, smovi.getResult());
        newLoadOp->setAttrs(loadOp->getAttrs());
        loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
        loadOp.erase();
      } else if(auto storeOp = dyn_cast<ftm::StoreOp>(ptrUserOp)) {
        auto newStoreOp = builder.create<ftm::StoreOp>(loc,
            storeOp.getValue(), newCastOp, smovi.getResult());
        newStoreOp->setAttrs(storeOp->getAttrs());
        storeOp.erase();
      }
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
    int64_t scalarOffsetRegisterIdx = scalarOffsetStartId;
    int64_t vectorOffsetRegisterIdx = vectorOffsetStartId;
    funcOp.walk([&](arith::AddIOp addiOp) {
      auto memLevel = analyzeTipConstantAddOp(addiOp);
      if(memLevel != ftm::Cache::Unknown)
        implOffsetRegisterAllocating(addiOp, memLevel,
            scalarOffsetRegisterIdx, vectorOffsetRegisterIdx);
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createAllocateOffsetRegistersPass() {
  return std::make_unique<AllocateOffsetRegistersPass>();
}