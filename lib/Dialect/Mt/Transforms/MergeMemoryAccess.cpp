//===- MergeMemoryAccess.cpp - Merge memory access operations
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to merge adjacent memory access operations.
// It combines VLDW pairs into VLDDW and SLDW pairs into SLDDW when they access
// consecutive memory locations, reducing instruction count and improving
// memory bandwidth utilization.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_MERGEMEMORYACCESS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class MergeMemoryAccessPass
    : public impl::MergeMemoryAccessBase<MergeMemoryAccessPass> {
 public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implMergeMemoryAccess(funcOp);
  }

 private:
  // 融合两个操作
  void merge(Operation* a, Operation* b, OpBuilder& builder) {
    if(auto vldwOpA = dyn_cast<VldwOp>(a)){
      auto vldwiOpB = dyn_cast<VldwiOp>(b);

      auto base = vldwOpA.getBase();
      assert(base == vldwiOpB.getBase());

      auto dstA = vldwOpA.getDst();
      auto dstB = vldwiOpB.getDst();

      auto offset = vldwOpA.getOffset();
      if(offset)
        llvm_unreachable("还未考虑带offset寄存器情况");

      // llvm::outs() << "merge:" << "\n" << vldwOpA << "\n" << vldwiOpB << "\n";

      builder.setInsertionPoint(vldwOpA);
      auto vlddwOp = builder.create<VlddwOp>(vldwOpA.getLoc(), base, dstB, dstA, nullptr);
      for (auto namedAttr : vldwOpA->getAttrs()) {
        if (namedAttr.getName() == "immOffset") continue;
        vlddwOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      vldwOpA.erase();
      vldwiOpB.erase();
      // llvm::outs() << vlddwOp << "\n\n";
      return;
    }
    
    if(auto vldwiOpA = dyn_cast<VldwiOp>(a)){
      auto vldwiOpB = dyn_cast<VldwiOp>(b);

      auto base = vldwiOpA.getBase();
      assert(base == vldwiOpB.getBase());

      auto dstA = vldwiOpA.getDst();
      auto dstB = vldwiOpB.getDst();

      auto immOffset = vldwiOpA.getImmOffset();

      // llvm::outs() << "merge:" << "\n" << vldwiOpA << "\n" << vldwiOpB << "\n";

      builder.setInsertionPoint(vldwiOpA);
      auto vlddwiOp = builder.create<VlddwiOp>(vldwiOpA.getLoc(), base, dstB, dstA, immOffset/2);
      for (auto namedAttr : vldwiOpA->getAttrs()) {
        if (namedAttr.getName() == "immOffset") continue;
        vlddwiOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      vldwiOpA.erase();
      vldwiOpB.erase();
      // llvm::outs() << vlddwiOp << "\n\n";
      return;
    }

    if(auto vstwOpA = dyn_cast<VstwOp>(a)){
      auto vstwiOpB = dyn_cast<VstwiOp>(b);

      auto base = vstwOpA.getBase();
      assert(base == vstwiOpB.getBase());

      auto srcA = vstwOpA.getSrc();
      auto srcB = vstwiOpB.getSrc();

      auto offset = vstwOpA.getOffset();
      if(offset)
        llvm_unreachable("还未考虑带offset寄存器情况");

      // llvm::outs() << "merge:" << "\n" << vstwOpA << "\n" << vstwiOpB << "\n";

      builder.setInsertionPoint(vstwOpA);
      auto vstdwOp = builder.create<VstdwOp>(vstwOpA.getLoc(), srcB, srcA, base, nullptr);
      for (auto namedAttr : vstwOpA->getAttrs()) {
        if (namedAttr.getName() == "immOffset") continue;
        vstdwOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      vstwOpA.erase();
      vstwiOpB.erase();
      // llvm::outs() << vstdwOp << "\n\n";
      return;
    }

    if(auto vstwiOpA = dyn_cast<VstwiOp>(a)){
      auto vstwiOpB = dyn_cast<VstwiOp>(b);

      auto base = vstwiOpA.getBase();
      assert(base == vstwiOpB.getBase());

      auto srcA = vstwiOpA.getSrc();
      auto srcB = vstwiOpB.getSrc();

      auto immOffset = vstwiOpA.getImmOffset();

      // llvm::outs() << "merge:" << "\n" << vstwiOpA << "\n" << vstwiOpB << "\n";

      builder.setInsertionPoint(vstwiOpA);
      auto vstdwiOp = builder.create<VstdwiOp>(vstwiOpA.getLoc(), srcB, srcA, base, immOffset/2);
      for (auto namedAttr : vstwiOpA->getAttrs()) {
        if (namedAttr.getName() == "immOffset") continue;
        vstdwiOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      vstwiOpA.erase();
      vstwiOpB.erase();
      // llvm::outs() << vstdwiOp << "\n\n";
      return;
    }
  }

  // 检查并合并连续的内存访问操作
  void mergeAdjacentMemoryAccesses(std::vector<std::vector<Operation*>>& opArray,
                               OpBuilder& builder, bool allowCrossRow = false) {
    int rowSize = opArray.size();
    int colSize = opArray[0].size();

    int index = 0;
    for (int i = 0; i < rowSize; i++) {
      for (int j = 0; j < colSize; j++) {
        // 如果当前不是DW的对齐位置，跳过
        if (index % 2) {
          index++;
          continue;
        }

        Operation* cur = opArray[i][j];
        Operation* next = nullptr;
        if (j < colSize - 1) {
          next = opArray[i][j + 1];
          merge(cur, next, builder);
        } else if (allowCrossRow && i < rowSize - 1) {
          next = opArray[i + 1][0];
          merge(cur, next, builder);
        }

        index++;
      }
    }
  }

  void implMergeMemoryAccess(func::FuncOp funcOp) {
    OpBuilder builder(funcOp);

    int M = 0;
    int N = 0;
    int K = 2;

    // 第一次遍历：找出最大的M和N
    funcOp.walk([&](mt::Vfmulas32Op vfmulas32Op) {
      auto mAttr = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.m");
      auto nAttr = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.n");
      if (!mAttr || !nAttr) return;

      int m = mAttr.getInt();
      int n = nAttr.getInt();
      if (m > M) M = m;
      if (n > N) N = n;
    });
    M++;
    N++;

    // 创建数组保存C的ld和st，B的ld
    std::vector<std::vector<Operation*>> cLoad(M, std::vector<Operation*>(N));
    std::vector<std::vector<Operation*>> cStore(M, std::vector<Operation*>(N));
    std::vector<std::vector<Operation*>> bLoad(K, std::vector<Operation*>(N));

    // 第二次遍历：添加属性并同时保存到数组
    funcOp.walk([&](mt::Vfmulas32Op vfmulas32Op) {
      // 获取属性
      auto mAttr = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.m");
      auto nAttr = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.n");
      if (!mAttr || !nAttr) return;

      int m = mAttr.getInt();
      int n = nAttr.getInt();

      // 为C生成属性并保存load/store操作
      Value dst = vfmulas32Op.getDst();
      for (auto user : dst.getUsers()) {
        if (isa<mt::VldwOp>(user) || isa<mt::VldwiOp>(user)) {
          user->setAttr("matmul.m", mAttr);
          user->setAttr("matmul.n", nAttr);
          cLoad[m][n] = user;
        } else if (isa<mt::VstwOp>(user) || isa<mt::VstwiOp>(user)) {
          user->setAttr("matmul.m", mAttr);
          user->setAttr("matmul.n", nAttr);
          cStore[m][n] = user;
        }
      }

      // 获取k属性
      auto kAttr = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.k");
      if (!kAttr) return;

      int k = kAttr.getInt();

      // 为B生成属性并保存load操作
      Value src2 = vfmulas32Op.getSrc2();
      for (auto user : src2.getUsers()) {
        if (isa<mt::VldwOp>(user) || isa<mt::VldwiOp>(user)) {
          user->setAttr("matmul.k", kAttr);
          user->setAttr("matmul.n", nAttr);
          if (k < K) bLoad[k][n] = user;
        }
      }
    });

    // // 输出cLoad、cStore、bLoad
    // llvm::errs() << "============= Memory Access Operations =============\n";

    // // 输出cLoad
    // llvm::errs() << "C Load Operations:\n";
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     if (cLoad[i][j]) {
    //       llvm::errs() << "cLoad[" << i << "][" << j << "]: ";
    //       cLoad[i][j]->print(llvm::errs());
    //       llvm::errs() << "\n";
    //     }
    //   }
    // }
    // llvm::errs() << "\n";

    // // 输出cStore
    // llvm::errs() << "C Store Operations:\n";
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     if (cStore[i][j]) {
    //       llvm::errs() << "cStore[" << i << "][" << j << "]: ";
    //       cStore[i][j]->print(llvm::errs());
    //       llvm::errs() << "\n";
    //     }
    //   }
    // }
    // llvm::errs() << "\n";

    // // 输出bLoad
    // llvm::errs() << "B Load Operations:\n";
    // for (int k = 0; k < K; k++) {
    //   for (int j = 0; j < N; j++) {
    //     if (bLoad[k][j]) {
    //       llvm::errs() << "bLoad[" << k << "][" << j << "]: ";
    //       bLoad[k][j]->print(llvm::errs());
    //       llvm::errs() << "\n";
    //     }
    //   }
    // }
    // llvm::errs() << "\n";

    // 检查它们是否访问连续的内存位置并进行合并
    // llvm::errs() << "Merging cLoad" << " Operations:\n";
    mergeAdjacentMemoryAccesses(cLoad, builder, true);
    mergeAdjacentMemoryAccesses(cStore, builder, true);
    mergeAdjacentMemoryAccesses(bLoad, builder);
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createMergeMemoryAccessPass() {
  return std::make_unique<MergeMemoryAccessPass>();
}