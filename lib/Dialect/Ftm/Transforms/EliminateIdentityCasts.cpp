//===--------- EliminateIdentityCasts.cpp - eliminate identity cast Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Ftm/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_ELIMINATEIDENTITYCASTS
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace ftm;

namespace {

void implEliminateIdentityCasts(func::FuncOp funcOp) {
  // 存储需要替换的映射关系: 第二个cast操作结果 -> 原始值
  DenseMap<Value, Value> replacements;
  
  // 先收集所有p->i64->p形式的cast对
  funcOp.walk([&](ftm::CastOp secondCastOp) {
    // 检查第二个cast是否将i64转换为ptr
    if (!secondCastOp.getResult().getType().isa<LLVM::LLVMPointerType>())
      return;
      
    // 检查输入是否来自另一个CastOp
    auto firstCastOp = secondCastOp.getOperand().getDefiningOp<ftm::CastOp>();
    if (!firstCastOp)
      return;
      
    // 检查第一个cast是否将ptr转换为i64
    if (!firstCastOp.getOperand().getType().isa<LLVM::LLVMPointerType>() ||
        !firstCastOp.getResult().getType().isInteger(64))
      return;
      
    // 记录替换关系
    replacements[secondCastOp.getResult()] = firstCastOp.getOperand();
  });
  
  // 应用所有替换
  for (auto &[result, originalValue] : replacements) {
    result.replaceAllUsesWith(originalValue);
  }
  
  // 清理无用的操作
  // 这里不直接删除，因为在遍历时删除可能会导致迭代器失效
  // 让死代码消除pass处理这些无用操作
}

} // namespace

namespace mlir {
class EliminateIdentityCastsPass : 
    public impl::EliminateIdentityCastsBase<EliminateIdentityCastsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implEliminateIdentityCasts(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::ftm::createEliminateIdentityCastsPass() {
  return std::make_unique<EliminateIdentityCastsPass>();
}