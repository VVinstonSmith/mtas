//===- DependencyGraph.h - Instruction Dependency Graph ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DependencyGraph class for instruction scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYGRAPH_H
#define MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYGRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace mt {

class InstructionSchedulingInterface;

class DependencyGraph {
public:
  // 使用位掩码定义依赖类型
  enum DependencyType {
    // 依赖性质维度（使用低3位）
    TrueDependency = 0x01,     // 真依赖（写后读，RAW）：001
    AntiDependency = 0x02,     // 反依赖（读后写，WAR）：010
    OutputDependency = 0x04,   // 输出依赖（写后写，WAW）：100
    
    // 循环关系维度（使用第5-6位）
    IntraLoop = 0x10,          // 循环内：10000
    InterLoop = 0x20,          // 循环间：100000
    
    // 组合类型（便于使用）
    IntraLoopTrueDependency = TrueDependency | IntraLoop,         // 10001
    IntraLoopAntiDependency = AntiDependency | IntraLoop,         // 10010
    IntraLoopOutputDependency = OutputDependency | IntraLoop,     // 10100
    InterLoopTrueDependency = TrueDependency | InterLoop,         // 100001
    InterLoopAntiDependency = AntiDependency | InterLoop,         // 100010
    InterLoopOutputDependency = OutputDependency | InterLoop      // 100100
  };

  // 依赖边结构体，包含类型和延迟
  struct DependencyEdge {
    DependencyType type;    // 依赖类型
    int latency;       // 延迟权重
    
    DependencyEdge(DependencyType t, int lat) : type(t), latency(lat) {}

    // 为了使用set，需要定义比较运算符
    bool operator<(const DependencyEdge &other) const {
      return type < other.type;
    }
    
    // 为了方便判断相等，也可以添加相等运算符
    bool operator==(const DependencyEdge &other) const {
      return type == other.type;
    }
  };

  // 添加依赖边
  void addDependency(mt::InstructionSchedulingInterface source, 
                    mt::InstructionSchedulingInterface target, 
                    DependencyType type);

  using DependencyInfo = std::tuple<mt::InstructionSchedulingInterface, DependencyType, int>;
  // 获取一个操作的所有出边
  SmallVector<DependencyInfo, 4> getOutgoingDependencies(mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的所有入边
  SmallVector<DependencyInfo, 4> getIncomingDependencies(mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的特定类型的出边
  SmallVector<DependencyInfo, 4> getOutgoingDependenciesByType(
      mt::InstructionSchedulingInterface op, uint32_t typeMask) const;

  // 获取一个操作的特定类型的入边
  SmallVector<DependencyInfo, 4> getIncomingDependenciesByType(
      mt::InstructionSchedulingInterface op, uint32_t typeMask) const;

  // 获取一个操作的循环内出边及其目标操作
  SmallVector<DependencyInfo, 4> getIntraLoopOutgoingDependencies(
      mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的循环内入边及其源操作
  SmallVector<DependencyInfo, 4> getIntraLoopIncomingDependencies(
      mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的循环间出边及其目标操作
  SmallVector<DependencyInfo, 4> getInterLoopOutgoingDependencies(
      mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的循环间入边及其源操作
  SmallVector<DependencyInfo, 4> getInterLoopIncomingDependencies(
      mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的真依赖出边及其目标操作
  SmallVector<DependencyInfo, 4> getOutgoingTrueDependencies(
      mt::InstructionSchedulingInterface op) const;

  // 获取一个操作的真依赖入边及其目标操作
  SmallVector<DependencyInfo, 4> getIncomingTrueDependencies(
      mt::InstructionSchedulingInterface op) const;

  static const uint32_t DEPENDENCY_NATURE_MASK = 0x07;  // 0000 0111 (低3位)
  static const uint32_t LOOP_RELATION_MASK = 0x30;      // 0011 0000 (第5-6位)

  // 检查依赖性质（无论循环内外）
  static bool isTrueDependency(DependencyType type);
  static bool isAntiDependency(DependencyType type);
  static bool isOutputDependency(DependencyType type);

  // 检查循环关系
  static bool isIntraLoopDependency(DependencyType type);
  static bool isInterLoopDependency(DependencyType type);

  static StringRef typeToString(DependencyType type);

  // 清空图
  void clear();

private:
  // 存储从源操作到目标操作的所有依赖边
  // 注意：对于每对操作，可能存在多条不同类型的边
  DenseMap<mt::InstructionSchedulingInterface, 
           DenseMap<mt::InstructionSchedulingInterface, 
                    SmallVector<DependencyEdge, 2>>> forwardEdges;
  
  // 存储从目标操作到源操作的反向映射（便于快速查询）
  DenseMap<mt::InstructionSchedulingInterface, 
           DenseMap<mt::InstructionSchedulingInterface, 
                    SmallVector<DependencyEdge, 2>>> backwardEdges;
};

} // namespace mt
} // namespace mlir

#endif // MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYGRAPH_H