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
  // 检查值是否由操作定义
  if(auto defOp = ptr.getDefiningOp()) {
    // 如果是GEP操作，则递归检查基址指针
    if(auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
      return searchAllocaFromPtr(gepOp.getBase());
    } else if(auto allocaOp = dyn_cast<LLVM::AllocaOp>(defOp)) {
      // 如果是Alloca操作，则直接返回
      return allocaOp;
    }
  }
  // 如果不是由操作定义，或者不是GEP或Alloca操作，则返回nullptr
  return nullptr;
}

void implRegisterFolding(LLVM::AllocaOp allocaOp, 
                        DenseMap<std::pair<Value, int64_t>, Value>& memToRegMap) {
  auto loc = allocaOp.getLoc();
  auto ctx = allocaOp.getContext();
  OpBuilder builder(ctx);

  // 获取内存级别属性
  auto memLevelAttr = allocaOp->getAttr(ftm::MemLevelAttr::name);
  if (!memLevelAttr) {
    llvm::errs() << "alloca must have memory level attr\n";
    return;
  }
  
  ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

  // 验证内存级别是否为寄存器
  if (!unitLengthOf.count(memLevel)) {
    llvm::errs() << "memory level must be scalar/vector register\n";
    return;
  }
  int64_t elemSize = unitLengthOf.at(memLevel);

  // 计算需要的寄存器数量
  int64_t n_regs = 0;
  if (auto defOp = allocaOp.getArraySize().getDefiningOp()) {
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(defOp)) {
      n_regs = constOp.getValue().cast<IntegerAttr>().getInt() / elemSize;
    }
  }
  if (n_regs == 0)
    return;

  // 设置插入点 - 在AllocaOp之后
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(allocaOp);

  // 获取输出类型
  Type outputType = VectorType::get({elemSize}, builder.getF32Type());

  // 为每个需要的寄存器创建声明
  for (int64_t i = 0; i < n_regs; i++) {
    auto declareOp = builder.create<ftm::DeclareRegisterOp>(loc, outputType);
    declareOp->setAttr(ftm::MemLevelAttr::name, memLevelAttr);
    
    // 计算偏移量
    int64_t offset = i * elemSize;
    
    // 创建键值对：(分配操作, 偏移量) -> 寄存器值
    auto key = std::make_pair(allocaOp.getResult(), offset);
    memToRegMap[key] = declareOp.getResult();
  }
}

std::pair<LLVM::AllocaOp, int64_t>
searchAllocaAndOffsetFromPtr(Value ptr) {
  auto defOp = ptr.getDefiningOp();
  
  // 验证输入是否有效
  if(!defOp || 
      (!isa<LLVM::AllocaOp>(defOp) && !isa<LLVM::GEPOp>(defOp))) {
    llvm::errs() << "addr must be defined by alloca or gep\n";
    return {};
  }
  
  // 如果是Alloca操作，偏移量为0
  if(auto alloca = dyn_cast<LLVM::AllocaOp>(defOp)) {
    return {alloca, 0};
  }
  
  // 如果是GEP操作，需要计算偏移量
  if(auto gep = dyn_cast<LLVM::GEPOp>(defOp)) {
    // 检查索引是否为常量
    if(auto attr = gep.getIndices()[0].dyn_cast<IntegerAttr>()) {
      // 递归检查基址指针并获取累积偏移量
      auto [alloca, offset] = searchAllocaAndOffsetFromPtr(gep.getBase());
      if(alloca) {
        // 返回分配操作和累积偏移量
        return {alloca, offset + attr.getInt()};
      }
    } else {
      llvm::errs() << "gep must have constant index\n";
      return {};
    }
  }
  
  // 其他情况返回空
  return {};
}

bool replaceLoadAndStoreWithRegister(
    Operation* op, 
    DenseMap<std::pair<Value, int64_t>, Value>& memToRegMap) {
  auto loc = op->getLoc();
  auto ctx = op->getContext();
  OpBuilder builder(ctx);

  // 获取内存地址
  Value addr;
  if (auto loadOp = dyn_cast<ftm::LoadOp>(op)) {
    addr = loadOp.getAddr();
  } else if (auto storeOp = dyn_cast<ftm::StoreOp>(op)) {
    addr = storeOp.getAddr();
  } else {
    // 不是load或store操作，直接返回
    return false;
  }
  
  // 查找分配操作和偏移量
  auto [alloca, offset] = searchAllocaAndOffsetFromPtr(addr);
  if (!alloca) {
    // 无法找到分配操作，返回
    return false;
  }
  
  // 检查内存级别属性
  auto memLevelAttr = op->getAttr(ftm::MemLevelAttr::name);
  if (!memLevelAttr) {
    llvm::errs() << "operation must have memory level attr\n";
    return false;
  }
  
  ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();
  if (memLevel != ftm::Cache::ScalarRegister && 
      memLevel != ftm::Cache::VectorRegister) {
    // 非寄存器内存级别，返回
    return false;
  }
  
  // 查找内存位置对应的寄存器值
  auto key = std::make_pair(alloca.getResult(), offset);
  if (!memToRegMap.count(key)) {
    llvm::errs() << "cannot find register for memory location\n";
    return false;
  }
  
  Value regValue = memToRegMap[key];
  
  // 根据操作类型处理
  if (auto loadOp = dyn_cast<ftm::LoadOp>(op)) {
    // 对于加载操作，直接用寄存器值替换所有使用
    loadOp.getResult().replaceAllUsesWith(regValue);
    loadOp.erase();
    return true;
  } else if (auto storeOp = dyn_cast<ftm::StoreOp>(op)) {
    // 对于存储操作，需要处理更复杂的情况
    auto defOp = storeOp.getValue().getDefiningOp();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(defOp);
    if (auto fma = dyn_cast<ftm::FMAOp>(defOp)) {
      // 如果是FMA操作，则创建VFMA操作
      auto vfmaOp = builder.create<ftm::VFMAOp>(
          loc,
          fma.getLhs(), fma.getRhs(), fma.getAcc(), regValue);
      // 复制原FMA操作的所有属性
      for (auto namedAttr : fma->getAttrs()) {
        vfmaOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
    }
    // else if (auto movi = dyn_cast<ftm::MoviOp>(defOp)) {
    //   builder.create<ftm::VmoviOp>(
    //       loc, movi.getImm(), regValue);
    // }
    // 删除原Store操作
    storeOp.erase();
    return true;
  }
  
  return false;
}

} // namepsace

namespace mlir {
class FoldRegisterAllocaPass : 
    public impl::FoldRegisterAllocaBase<FoldRegisterAllocaPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    // 用于存储内存位置到寄存器值的映射
    DenseMap<std::pair<Value, int64_t>, Value> memToRegMap;
    
    // 第一步：将内存级别属性从加载操作传播到分配操作
    funcOp.walk([&](ftm::LoadOp op) {
      if (auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
        if (auto alloca = searchAllocaFromPtr(op.getAddr())) {
          alloca->setAttr(ftm::MemLevelAttr::name, attr);
        }
      }
      return WalkResult::advance();
    });
    
    // 第二步：收集所有需要处理的分配操作并执行寄存器折叠
    SmallVector<LLVM::AllocaOp, 8> allocaOps;
    funcOp.walk([&](LLVM::AllocaOp op) {
      if (auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
        allocaOps.push_back(op);
      }
      return WalkResult::advance();
    });
    // 对收集到的分配操作执行寄存器折叠
    for (auto op : allocaOps) {
      implRegisterFolding(op, memToRegMap);
    }
    
    // 第三步：将加载/存储操作替换为直接寄存器访问
    funcOp.walk([&](Operation *op) {
      if (!isa<ftm::LoadOp>(op) && !isa<ftm::StoreOp>(op))
        return WalkResult::skip();
        
      if (auto attr = op->getAttr(ftm::MemLevelAttr::name)) {
        auto memLevel = attr.cast<ftm::MemLevelAttr>().getLevel();
        if (memLevel == ftm::Cache::ScalarRegister ||
            memLevel == ftm::Cache::VectorRegister) {
          replaceLoadAndStoreWithRegister(op, memToRegMap);
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