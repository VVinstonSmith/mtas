//===- InstructionFormatter.cpp - Instruction Formatting Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InstructionFormatter class for instruction formatting.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/InstructionFormatter.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include <llvm/Support/Format.h>

using namespace mlir;
using namespace mt;

std::string InstructionFormatter::formatInstructionName(StringRef baseName, StringRef functionalUnit) {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  os << baseName;
  
  // 如果指定了功能单元，则添加到指令名称中
  if (!functionalUnit.empty()) {
    os << "." << functionalUnit;
  }
  
  return os.str();
}

std::string InstructionFormatter::formatBranchWithCondition(
    StringRef opName,        // 操作名称（如"SBR"）
    Value cond){             // 条件寄存器（可以为nullptr表示无条件）
  std::ostringstream oss;
  
  // 如果有条件寄存器，则先输出[cond]
  if (cond) {
    oss << "[" << InstructionFormatter::formatRegisterName(cond) << "] ";
  }
  
  // 添加操作名称
  oss << opName.str();
  
  return oss.str();
}

std::string InstructionFormatter::formatRegisterName(Value value) {
  std::string regName;
  
  // 检查是否为操作结果
  if (value.isa<OpResult>()) {
    // 值来自操作
    Operation *definingOp = value.getDefiningOp();
    
    // 尝试将操作转换为DeclareRegisterOp
    if (auto regOp = dyn_cast<DeclareRegisterOp>(definingOp)) {
      regName = regOp.formatRegisterOutput();
    } else {
      // 如果不是DeclareRegisterOp，返回未知寄存器
      regName = "<Unknown>";
    }
  }
  
  // 检查是否为块参数（可能是函数参数）
  else if (value.isa<BlockArgument>()) {
    BlockArgument blockArg = value.cast<BlockArgument>();
    Block *block = blockArg.getOwner();
    
    // 判断是否为函数参数
    if (block->isEntryBlock() && block->getParent() && 
        isa<func::FuncOp>(block->getParent()->getParentOp())) {
      // 获取参数索引
      unsigned paramIndex = blockArg.getArgNumber();
      // regName = "R" + std::to_string(10 + 2*paramIndex);
      regName = "%" + std::to_string(paramIndex);
    } else {
      // 如果不是函数参数，可能是其他块参数
      regName = "<Block parameters>";
    }
  }
  
  // 其他情况
  else {
    regName = "<Invalid value>";
  }
  
  return regName;
}

std::string InstructionFormatter::formatOperandList(ArrayRef<Value> operands) {
  if (operands.empty()) {
    return "";
  }
  
  std::string result;
  llvm::raw_string_ostream os(result);
  
  // 添加第一个操作数
  os << formatRegisterName(operands[0]);
  
  // 添加其余操作数，用逗号分隔
  for (size_t i = 1; i < operands.size(); ++i) {
    os << ", " << formatRegisterName(operands[i]);
  }
  
  return os.str();
}

std::string InstructionFormatter::formatHexImmediate(uint64_t value){
  std::string result;
  llvm::raw_string_ostream os(result);
  
  // 使用十六进制格式化
  os << llvm::format_hex(value, /*Width=*/0, /*Lower=*/true);
  
  return os.str();
}

std::string InstructionFormatter::formatDecimalImmediate(uint64_t value){
  std::string result;
  llvm::raw_string_ostream os(result);
  
  // 使用十进制格式化
  os << value;
  
  return os.str();
}

std::string InstructionFormatter::formatMemoryOperand(Value base, Value offset) {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  // 添加内存访问符号"*"
  os << "*";
  
  // 检查是否有偏移量
  if (offset) {
    // 有偏移量，添加"+"符号
    os << "+";
  }
  
  // 添加基址寄存器
  os << formatRegisterName(base);
  
  // 如果有偏移量，添加偏移寄存器
  if (offset) {
    os << "[" << formatRegisterName(offset) << "]";
  }
  
  return os.str();
}

std::string InstructionFormatter::formatMemoryOperandWithImm(Value base, int64_t immOffset) {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  // 添加内存访问符号"*"
  os << "*";
  
  // 检查是否有立即数偏移量
  if (immOffset != 0) {
    // 有偏移量，添加"+"符号
    os << "+";
  }
  
  // 添加基址寄存器
  os << formatRegisterName(base);
  
  // 如果有立即数偏移量，添加立即数
  if (immOffset != 0) {
    os << "[" << immOffset << "]";
  }
  
  return os.str();
}