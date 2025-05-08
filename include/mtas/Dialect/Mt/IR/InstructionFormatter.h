//===- InstructionFormatter.h - Instruction Formatting Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstructionFormatter class for operation instruction formatting.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_IR_INSTRUCTIONFORMATTER_H
#define MTAS_DIALECT_MT_IR_INSTRUCTIONFORMATTER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir {
namespace mt {

// 前向声明
class DeclareRegisterOp;

class InstructionFormatter {
public:
  // 格式化指令名称（带功能单元）
  static std::string formatInstructionName(StringRef baseName, StringRef functionalUnit);

  // 格式化指令名称（带条件）
  static std::string formatBranchWithCondition(StringRef opName, Value cond);

  // 获取寄存器名称
  static std::string formatRegisterName(Value value);
  
  // 格式化操作数列表
  static std::string formatOperandList(ArrayRef<Value> operands);

  // 获取十六进制格式的立即数
  static std::string formatHexImmediate(uint64_t value);

  // 获取十进制格式的立即数
  static std::string formatDecimalImmediate(uint64_t value);
  
  // 格式化特殊操作数（如带偏移的访存指令）
  static std::string formatMemoryOperand(Value base, Value offset = nullptr);

  static std::string formatMemoryOperandWithImm(Value base, int64_t immOffset);
};

} // namespace mt
} // namespace mlir

#endif // MTAS_DIALECT_MT_IR_INSTRUCTIONFORMATTER_H