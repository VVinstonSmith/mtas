//===- DependencyGraph.cpp - Instruction Dependency Graph -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DependencyGraph class for instruction scheduling.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/Transforms/DependencyGraph.h"
#include "mtas/Dialect/Mt/IR/InstructionSchedulingInterface.h"

using namespace mlir;
using namespace mt;

void DependencyGraph::addDependency(mt::InstructionSchedulingInterface source, 
                                   mt::InstructionSchedulingInterface target, 
                                   DependencyType type) {
  // 获取源操作的延迟作为边的权重
  int latency = isTrueDependency(type) ? source.getLatency() : 0;
  
  // 创建一个新的依赖边
  DependencyEdge edge(type, latency);

  // 在forwardEdges中查找是否已经存在相同类型的边
  auto &sourceEdges = forwardEdges[source][target];
  // 检查是否存在相同类型的边
  for (const auto &existingEdge : sourceEdges) {
    // 如果存在完全相同类型的边，直接返回
    if (existingEdge.type == type)
      return;
      
    // 如果要添加的是循环间依赖，且存在对应的循环内依赖，也返回
    if (isInterLoopDependency(type) && 
        existingEdge.type == static_cast<DependencyType>((type & DEPENDENCY_NATURE_MASK) | IntraLoop))
      return;
  }
  
  // 将边添加到列表中
  forwardEdges[source][target].push_back(edge);
  backwardEdges[target][source].push_back(edge);
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getOutgoingDependencies(mt::InstructionSchedulingInterface op) const {
  SmallVector<DependencyInfo, 4> result;
  auto it = forwardEdges.find(op);
  if (it != forwardEdges.end()) {
    for (const auto &targetEntry : it->second) {
      mt::InstructionSchedulingInterface target = targetEntry.first;
      for (const auto &edge : targetEntry.second) {
        result.push_back(std::make_tuple(target, edge.type, edge.latency));
      }
    }
  }
  return result;
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getIncomingDependencies(mt::InstructionSchedulingInterface op) const {
  SmallVector<DependencyInfo, 4> result;
  auto it = backwardEdges.find(op);
  if (it != backwardEdges.end()) {
    for (const auto &sourceEntry : it->second) {
      mt::InstructionSchedulingInterface source = sourceEntry.first;
      for (const auto &edge : sourceEntry.second) {
        result.push_back(std::make_tuple(source, edge.type, edge.latency));
      }
    }
  }
  return result;
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getOutgoingDependenciesByType(
    mt::InstructionSchedulingInterface op, uint32_t typeMask) const {
  SmallVector<DependencyInfo, 4> result;
  auto it = forwardEdges.find(op);
  if (it != forwardEdges.end()) {
    for (const auto &targetEntry : it->second) {
      mt::InstructionSchedulingInterface target = targetEntry.first;
      for (const auto &edge : targetEntry.second) {
        // 检查依赖类型是否匹配掩码
        if (edge.type & typeMask) {
          result.push_back(std::make_tuple(target, edge.type, edge.latency));
        }
      }
    }
  }
  return result;
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getIncomingDependenciesByType(
    mt::InstructionSchedulingInterface op, uint32_t typeMask) const {
  SmallVector<DependencyInfo, 4> result;
  auto it = backwardEdges.find(op);
  if (it != backwardEdges.end()) {
    for (const auto &sourceEntry : it->second) {
      mt::InstructionSchedulingInterface source = sourceEntry.first;
      for (const auto &edge : sourceEntry.second) {
        // 检查依赖类型是否匹配掩码
        if (edge.type & typeMask) {
          result.push_back(std::make_tuple(source, edge.type, edge.latency));
        }
      }
    }
  }
  return result;
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getIntraLoopOutgoingDependencies(
    mt::InstructionSchedulingInterface op) const {
  return getOutgoingDependenciesByType(op, IntraLoop);
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getIntraLoopIncomingDependencies(
    mt::InstructionSchedulingInterface op) const {
  return getIncomingDependenciesByType(op, IntraLoop);
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getInterLoopOutgoingDependencies(
    mt::InstructionSchedulingInterface op) const {
  return getOutgoingDependenciesByType(op, InterLoop);
}

SmallVector<DependencyGraph::DependencyInfo, 4> 
DependencyGraph::getInterLoopIncomingDependencies(
    mt::InstructionSchedulingInterface op) const {
  return getIncomingDependenciesByType(op, InterLoop);
}

bool DependencyGraph::isTrueDependency(DependencyType type) {
  return (type & DEPENDENCY_NATURE_MASK) == TrueDependency;
}

bool DependencyGraph::isAntiDependency(DependencyType type) {
  return (type & DEPENDENCY_NATURE_MASK) == AntiDependency;
}

bool DependencyGraph::isOutputDependency(DependencyType type) {
  return (type & DEPENDENCY_NATURE_MASK) == OutputDependency;
}

bool DependencyGraph::isIntraLoopDependency(DependencyType type) {
  return (type & LOOP_RELATION_MASK) == IntraLoop;
}

bool DependencyGraph::isInterLoopDependency(DependencyType type) {
  return (type & LOOP_RELATION_MASK) == InterLoop;
}

StringRef DependencyGraph::typeToString(DependencyType type) {
  // 根据位掩码组合确定字符串
  if (isInterLoopDependency(type)) {
    if (isTrueDependency(type)) {
      return "循环间真依赖(RAW)";
    } else if (isAntiDependency(type)) {
      return "循环间反依赖(WAR)";
    } else if (isOutputDependency(type)) {
      return "循环间输出依赖(WAW)";
    }
  } else { // 循环内依赖
    if (isTrueDependency(type)) {
      return "循环内真依赖(RAW)";
    } else if (isAntiDependency(type)) {
      return "循环内反依赖(WAR)";
    } else if (isOutputDependency(type)) {
      return "循环内输出依赖(WAW)";
    }
  }
  return "未知依赖";
}

void DependencyGraph::clear() {
  forwardEdges.clear();
  backwardEdges.clear();
}