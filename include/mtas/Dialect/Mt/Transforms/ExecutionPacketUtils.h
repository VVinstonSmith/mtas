//===- ExecutionPacketUtils.h - Execution Packet Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExecutionPacket structure and related utility functions
// for instruction scheduling and packing.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_TRANSFORMS_EXECUTIONPACKETUTILS_H
#define MTAS_DIALECT_MT_TRANSFORMS_EXECUTIONPACKETUTILS_H

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/IR/InstructionSchedulingInterface.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>
#include <vector>

namespace mlir {
namespace mt {

// 定义执行包结构体，包含11个功能单元的使用情况
struct ExecutionPacket {
  // 标量单元
  InstructionSchedulingInterface SMAC[2] = {nullptr, nullptr};   // 2个标量乘加单元
  InstructionSchedulingInterface SIEU = nullptr;                 // 1个标量指令执行单元
  InstructionSchedulingInterface SLDST = nullptr;                // 1个标量加载/存储单元
  InstructionSchedulingInterface SBR = nullptr;                  // 1个标量分支单元
  
  // 向量单元
  InstructionSchedulingInterface VMAC[3] = {nullptr, nullptr, nullptr};  // 3个向量乘加单元
  InstructionSchedulingInterface VIEU = nullptr;                         // 1个向量指令执行单元
  InstructionSchedulingInterface VLDST[2] = {nullptr, nullptr};          // 2个向量加载/存储单元

  // 查看一个包是否为空
  bool isFree();

  // 向执行包中的指定单元添加操作
  bool addOperation(mt::FunctionalUnit unit, InstructionSchedulingInterface op);

  // 根据操作的功能单元方法向执行包中添加操作
  bool addOperation(InstructionSchedulingInterface op);

  // 创建一个基于条件过滤后的执行包副本
  ExecutionPacket createFilteredCopy(const std::function<bool(InstructionSchedulingInterface)>& filterCondition) const;

  // 格式化输出一个执行包
  void print(raw_ostream &os);
};

// 获取操作的简化类型名称
std::string getSimplifiedOpName(InstructionSchedulingInterface op);

// 固定宽度的字符串，左对齐
std::string padLeft(const std::string &str, unsigned width);

// 以表格形式输出调度窗口，支持正序或逆序打印
void printSchedulingWindowAsTable(
    const std::vector<ExecutionPacket>& window, 
    bool reverseOrder = false,
    int printStartPos = 0,
    raw_ostream &os = llvm::outs());

// 以最终格式输出调度后的指令
void printExecutionPackets(
    const std::vector<ExecutionPacket>& preLoopOps, 
    const std::vector<ExecutionPacket>& loopOps,
    const std::vector<ExecutionPacket>& postLoopOps, 
    raw_ostream &os = llvm::outs());

} // namespace mt
} // namespace mlir

#endif // MTAS_DIALECT_MT_TRANSFORMS_EXECUTIONPACKETUTILS_H