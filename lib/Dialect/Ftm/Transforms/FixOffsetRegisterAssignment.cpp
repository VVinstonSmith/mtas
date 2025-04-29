//===--------- FixOffsetRegisterAssignment.cpp - Fix Register Assignment Pass ---------===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include <iostream>
#include <vector>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_FIXOFFSETREGISTERASSIGNMENT
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

// 检查寄存器是否为offset_register类型
bool isOffsetRegister(Value reg) {
  if (auto defOp = reg.getDefiningOp()) {
    if (auto declareOp = dyn_cast<ftm::DeclareRegisterOp>(defOp)) {
      if (auto attr = defOp->getAttr(ftm::MemLevelAttr::name)) {
        auto memLevel = attr.cast<ftm::MemLevelAttr>().getLevel();
        return memLevel == ftm::Cache::OffsetRegister;
      }
    }
  }
  return false;
}

void implFixOffsetRegisterAssignment(func::FuncOp funcOp) {
  // 找到所有使用offset_register的ftm.smovi操作并转换它们
  funcOp.walk([&](ftm::SmoviOp smoviOp) {
    // 检查此smovi操作是否使用了offset_register
    Value reg = smoviOp.getReg();
    
    if (isOffsetRegister(reg)) {
      // 创建一个builder来插入新操作
      OpBuilder builder(smoviOp);
      
      // 获取immediate值
      int64_t immValue = smoviOp.getImm();
      
      // 创建arith.constant操作
      auto constOp = builder.create<arith::ConstantOp>(
          smoviOp.getLoc(), 
          builder.getI64IntegerAttr(immValue)
      );
      
      // 创建ftm.smvaga操作
      builder.create<ftm::SmvagaOp>(
          smoviOp.getLoc(),
          constOp.getResult(),
          reg
      );
      
      // 删除原来的ftm.smovi操作
      smoviOp.erase();
    }
  });
}

} // namespace

namespace mlir {
class FixOffsetRegisterAssignmentPass 
    : public impl::FixOffsetRegisterAssignmentBase<FixOffsetRegisterAssignmentPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implFixOffsetRegisterAssignment(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createFixOffsetRegisterAssignmentPass() {
  return std::make_unique<FixOffsetRegisterAssignmentPass>();
}