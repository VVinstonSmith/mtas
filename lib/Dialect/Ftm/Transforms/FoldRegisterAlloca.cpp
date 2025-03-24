//===----------------- FoldRegisterAlloca.cpp - fold register alloca -----------------===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_FOLDREGISTERALLOCA
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ftm;

namespace {

DenseMap<ftm::Cache, int64_t> unitLengthOf = {
  {ftm::Cache::ScalarRegister, 2},
  {ftm::Cache::VectorRegister, 32}
};

LLVM::AllocaOp searchAllocaFromPtr(Value ptr) {
  if(auto defOp = ptr.getDefiningOp()) {
    if(auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
      return searchAllocaFromPtr(gepOp.getBase());
    } else if(auto allocaOp = dyn_cast<LLVM::AllocaOp>(defOp)) {
      return allocaOp;
    }
  }
  return nullptr;
}

DenseSet<unsigned> scanAllocatedRegisters(func::FuncOp funcOp, ftm::Cache memLevel) {
  DenseSet<unsigned> allocated;
  funcOp.walk([&](ftm::DeclareRegisterOp op) {
    if(auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
      if(attr.cast<ftm::MemLevelAttr>().getLevel() != memLevel)
        return WalkResult::skip();
    } else return WalkResult::skip();
    if(auto attr = op->getAttr(ftm::RegisterIdAttr::name)) {
      unsigned id = attr.cast<ftm::RegisterIdAttr>().getId();
      allocated.insert(id);
    }
    return WalkResult::advance();
  });
  return allocated;
}

void implRegisterFolding(LLVM::AllocaOp allocaOp) {
  auto loc = allocaOp.getLoc();
  auto ctx = allocaOp.getContext();
  OpBuilder builder(ctx);

  auto memLevelAttr = allocaOp->getAttr(ftm::MemLevelAttr::name);
  ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

  if(!unitLengthOf.count(memLevel)) {
    llvm::errs() << "memory level must be scalar/vector regist\n";
    return;
  }
  int64_t elemSize = unitLengthOf.at(memLevel);

  int64_t n_regs = 0;
  if(auto defOp = allocaOp.getArraySize().getDefiningOp()) {
    if(auto constOp = dyn_cast<LLVM::ConstantOp>(defOp)) {
      n_regs = constOp.getValue().cast<IntegerAttr>().getInt() / elemSize;
    }
  }
  if(n_regs == 0)
    return;

  DenseSet<unsigned> allocatedIds = scanAllocatedRegisters(
      allocaOp->getParentOfType<func::FuncOp>(), memLevel); 

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(allocaOp);

  SmallVector<unsigned> registerIndices;
  Type outputType = VectorType::get({elemSize}, builder.getF32Type());

  for(int64_t idx = 0; n_regs != 0 ; idx++) {
    if(allocatedIds.count(idx))
      continue;
    auto declareOp = builder.create<ftm::DeclareRegisterOp>(loc, outputType);
    declareOp->setAttr(ftm::MemLevelAttr::name, memLevelAttr);
    declareOp->setAttr(ftm::RegisterIdAttr::name,
        ftm::RegisterIdAttr::get(ctx, idx)); 
    registerIndices.push_back(idx);
    n_regs--;
  }
  allocaOp->setAttr(ftm::RegisterIndicesAttr::name,
      ftm::RegisterIndicesAttr::get(ctx, registerIndices));
}

std::pair<LLVM::AllocaOp, int64_t>
searchAllocaAndOffsetFromPtr(Value ptr) {
  auto defOp = ptr.getDefiningOp();
  if(!defOp || 
      (!isa<LLVM::AllocaOp>(defOp) && !isa<LLVM::GEPOp>(defOp))) {
    llvm::errs() << "addr must be defined by alloca or gep\n";
    return {};
  }
  if(auto alloca = dyn_cast<LLVM::AllocaOp>(defOp)) {
    return {alloca, 0};
  }
  if(auto gep = dyn_cast<LLVM::GEPOp>(defOp)) {
    if(auto attr = gep.getIndices()[0].dyn_cast<IntegerAttr>()) {
      auto [alloca, offset] = searchAllocaAndOffsetFromPtr(gep.getBase());
      if(alloca) {
        return {alloca, offset + attr.getInt()};
      }
    } else {
      llvm::errs() << "gep must have constant index\n";
      return {};
    }
  }
  return {};
}

Value searchRegisterDeclareWith(
    func::FuncOp funcOp, Attribute memLevelAttr, int64_t registerId) {
  Value declareRegisterValue;
  funcOp.walk([&](ftm::DeclareRegisterOp op) {
    if(auto memLevelAttr_1 = op->getAttr(ftm::MemLevelAttr::name)) {
      if(memLevelAttr_1 != memLevelAttr)
        return WalkResult::skip();
      if(auto attr = op->getAttr(ftm::RegisterIdAttr::name)) {
        int64_t regId = attr.cast<ftm::RegisterIdAttr>().getId();
        if(regId == registerId) {
          declareRegisterValue = op.getResult();
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if(!declareRegisterValue) {
    llvm::errs() << "fail to find correct register declaration.\n";
    return nullptr;
  }
  return declareRegisterValue;
}

bool replaceLoadAndStoreWithRegister(Operation* op) {
  auto loc = op->getLoc();
  auto ctx = op->getContext();
  OpBuilder builder(ctx);

  Value addr;
  if(auto loadOp = dyn_cast<ftm::LoadOp>(op)) {
    addr = loadOp.getAddr();
  } else if(auto storeOp = dyn_cast<ftm::StoreOp>(op)) {
    addr = storeOp.getAddr();
  } else
    return false;
  auto [alloca, offset] = searchAllocaAndOffsetFromPtr(addr);
  if(!alloca)
    return false;
  auto memLevelAttr = alloca->getAttr(ftm::MemLevelAttr::name);
  if(!memLevelAttr) {
    llvm::errs() << "alloca must have memory level attr\n";
    return false;
  }
  ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();
  auto registerIndices = alloca->getAttr(
      ftm::RegisterIndicesAttr::name).cast<ftm::RegisterIndicesAttr>().getIndices();
  auto funcOp = op->getParentOfType<func::FuncOp>();
  auto regDecVal = searchRegisterDeclareWith(
      funcOp, memLevelAttr, registerIndices[offset / unitLengthOf.at(memLevel)]);
  if(!regDecVal)
    return false;
  if(auto loadOp = dyn_cast<ftm::LoadOp>(op)) {
    loadOp.getResult().replaceAllUsesWith(regDecVal);
    loadOp.erase();
  } else if(auto storeOp = dyn_cast<ftm::StoreOp>(op)) {
    auto defOp = storeOp.getValue().getDefiningOp();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(defOp);
    if(auto fma = dyn_cast<ftm::FMAOp>(defOp)) {
      builder.create<ftm::VFMAOp>(loc,
          fma.getLhs(), fma.getRhs(), fma.getAcc(), regDecVal);
    } else if(auto movi = dyn_cast<ftm::MoviOp>(defOp)) {
      builder.create<ftm::VmoviOp>(loc, movi.getImm(), movi.getReg());
    }
    storeOp.erase();
  }
  return true;
}

} // namepsace

namespace mlir {
class FoldRegisterAllocaPass : 
    public impl::FoldRegisterAllocaBase<FoldRegisterAllocaPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](ftm::LoadOp op) {
      if(auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
        if(auto alloca = searchAllocaFromPtr(op.getAddr())) {
          alloca->setAttr(ftm::MemLevelAttr::name, attr);
        }
      }
      return WalkResult::advance();
    });
    funcOp.walk([&](LLVM::AllocaOp op) {
      if(op->getAttr(ftm::RegisterIndicesAttr::name))
        return WalkResult::skip();
      if(auto attr = op->getAttr(ftm::MemLevelAttr::name)) 
        implRegisterFolding(op);
      return WalkResult::advance();
    });
    funcOp.walk([&](Operation *op) {
      if(!isa<ftm::LoadOp>(op) && !isa<ftm::StoreOp>(op))
        return WalkResult::skip();
      if(auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
        auto memLevel = attr.cast<ftm::MemLevelAttr>().getLevel();
        if(memLevel == ftm::Cache::ScalarRegister ||
            memLevel == ftm::Cache::VectorRegister) {
          replaceLoadAndStoreWithRegister(op);
        }
      }
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createFoldRegisterAllocaPass() {
  return std::make_unique<FoldRegisterAllocaPass>();
}