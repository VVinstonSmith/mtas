//===- MatMulInPlaceAccumulation.cpp - Optimize MatMul register usage
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to optimize MatMul by accumulating directly on
// destination registers instead of using separate accumulator registers.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_MATMULINPLACEACCUMULATION
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mt;

namespace {
struct OpGroup {
  Operation* initOp;    // VMOVI/VMOV 初始化操作
  Operation* loadOp;    // VLDW/VLDWI 加载操作
  Operation* fmulasOp;  // VFMULAS32  累加操作
  Operation* storeOp;   // VSTW/VSTWI 存储操作
};

class MatMulInPlaceAccumulationPass
    : public impl::MatMulInPlaceAccumulationBase<
          MatMulInPlaceAccumulationPass> {
 public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implMatMulInPlaceAccumulation(funcOp);
  }

 private:
void implMatMulInPlaceAccumulation(func::FuncOp funcOp) {
  // 存储匹配到的操作组
  SmallVector<OpGroup, 8> matchedGroups;
  
  // 遍历函数中的所有块
  for (auto &block : funcOp.getBody()) {
    // 首先找到所有的VSTW操作
    for (auto &op : block) {
      // 跳过非VSTW操作
      if (!isa<VstwOp, VstwiOp>(&op))
        continue;
      
      // 找到一个VSTW或VSTWI操作
      Operation *storeOp = &op;
      Value storeSourceReg = storeOp->getOperand(0); // 存储的源寄存器
      Value storeDestPtr = storeOp->getOperand(1);   // 存储的目标地址
      
      // 获取源寄存器的定义操作
      Operation *regDefOp = storeSourceReg.getDefiningOp();
      if (!regDefOp || !isa<DeclareRegisterOp>(regDefOp))
        continue; // 确保是由declare_register定义的寄存器
      
      // 从使用者中识别VLDW和VFMULAS32操作
      Operation *loadOp = nullptr;
      Operation *fmulasOp = nullptr;
      
      for (Operation *user : storeSourceReg.getUsers()) {
        if (user == storeOp)
          continue; // 跳过VSTW操作
        
        if (isa<VldwOp, VldwiOp>(user)) {
          loadOp = user;
        } else if (auto fmulas = dyn_cast<Vfmulas32Op>(user)) {
          // 检查VFMULAS32的第三和第四操作数是否都是该寄存器
          if (fmulas.getSrc3() == storeSourceReg && 
              fmulas.getDst() == storeSourceReg) {
            fmulasOp = user;
          }
        } else {
          loadOp = nullptr;
          fmulasOp = nullptr;
          break;
        }
      }
      
      // 确保找到了所有需要的操作
      if (!loadOp || !fmulasOp)
        continue;
      
      // 获取VFMULAS32的第一个操作数（累加器寄存器）
      Value accumulatorReg = fmulasOp->getOperand(0);
      
      // 寻找初始化累加器的VMOVI操作
      Operation *initOp = nullptr;
      for (Operation *user : accumulatorReg.getUsers()) {
        if (user == fmulasOp)
          continue;
        
        if (auto movi = dyn_cast<VmoviOp>(user)) {
          if (movi.getReg() == accumulatorReg && 
              movi.getImm() == 0) { // 确保是初始化为0
            initOp = movi;
            break;
          }
        } else if (auto mov = dyn_cast<VmovOp>(user)) {
          if (mov.getDst() == accumulatorReg) {
            // 这里需要额外检查源操作数是否为0
            initOp = mov;
            break;
          }
        }
      }
      
      // 如果没有找到初始化操作，继续下一个
      if (!initOp)
        continue;
      
      // 检查地址是否匹配
      bool addressMatches = false;
      if (auto vldw = dyn_cast<VldwOp>(loadOp)) {
        addressMatches = (vldw.getBase() == storeDestPtr);
      } else if (auto vldwi = dyn_cast<VldwiOp>(loadOp)) {
        if (auto vstwi = dyn_cast<VstwiOp>(storeOp)) {
          addressMatches = (vldwi.getBase() == storeDestPtr && 
                           vldwi.getImmOffset() == vstwi.getImmOffset());
        } else {
          addressMatches = (vldwi.getBase() == storeDestPtr && 
                           vldwi.getImmOffset() == 0);
        }
      }
      
      if (!addressMatches)
        continue;
      
      // 找到了完整的匹配模式
      matchedGroups.push_back({initOp, loadOp, fmulasOp, storeOp});
      // llvm::outs() << "Found matching op group:\n"
      //            << "  Init: " << *initOp << "\n"
      //            << "  Load: " << *loadOp << "\n"
      //            << "  FMULAS: " << *fmulasOp << "\n"
      //            << "  Store: " << *storeOp << "\n";
    }
  }

  // 2. 对代码进行修改：
  // 将VMOVI或VMOV删除，在其位置插入新的VLDW：
  // VLDW %69, %65 : !llvm.ptr, vector<32xf32>
  // 将VLDW、VFMULAS32、VSTW删除，并在VSTW的位置插入新的VSTW：
  // VSTW %65, %69 : vector<32xf32>, !llvm.ptr
  // 执行代码修改，重写匹配到的操作组
  for (const auto &group : matchedGroups) {
    OpBuilder builder(funcOp.getContext());
    
    // 获取需要的值和位置信息
    Value accumulatorReg = cast<Vfmulas32Op>(group.fmulasOp).getSrc1(); // 累加器寄存器
    Value storeSourceReg = group.storeOp->getOperand(0); // 原始存储源寄存器
    Value storeDestPtr = group.storeOp->getOperand(1);   // 存储目标地址

    // 将累加器寄存器映射到要加载到的寄存器
    IRMapping valueMapper;
    valueMapper.map(storeSourceReg, accumulatorReg);
    
    // 1. 在VMOVI/VMOV的位置创建新的VLDW操作
    builder.setInsertionPoint(group.initOp);
    
    // 根据原始的VLDW/VLDWI操作类型，创建相应的新操作
    Operation *newLoadOp = builder.clone(*group.loadOp, valueMapper);
    
    // 2. 在VSTW位置创建新的VSTW操作
    builder.setInsertionPoint(group.storeOp);
    
    // 克隆原始的VSTW/VSTWI操作
    Operation *newStoreOp = builder.clone(*group.storeOp, valueMapper);
    
    // // 输出调试信息
    // llvm::outs() << "Created new operations:\n"
    //             << "  New Load: " << *newLoadOp << "\n"
    //             << "  New Store: " << *newStoreOp << "\n";
    
    // 3. 删除原有的操作
    group.initOp->erase();     // 删除VMOVI/VMOV
    group.loadOp->erase();     // 删除VLDW/VLDWI
    group.fmulasOp->erase();   // 删除VFMULAS32
    group.storeOp->erase();    // 删除VSTW/VSTWI
  }
}
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createMatMulInPlaceAccumulationPass() {
  return std::make_unique<MatMulInPlaceAccumulationPass>();
}