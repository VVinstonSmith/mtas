//===- DependencyAnalyzer.h - Instruction Dependency Analyzer ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DependencyAnalyzer class for instruction dependency
// analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYANALYZER_H
#define MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYANALYZER_H

#include "mtas/Dialect/Mt/Transforms/DependencyGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace mt {

class DependencyAnalyzer {
public:
  // 构造函数改为接受一个Region指针
  DependencyAnalyzer(Region *region);

  // 添加一个接受FuncOp的便捷构造方法（兼容原有用法）
  static DependencyAnalyzer forFunction(func::FuncOp funcOp);

  // 添加一个专门针对ForOp的便捷构造方法
  static DependencyAnalyzer forForOp(scf::ForOp forOp);

  // 运行分析，生成依赖关系图
  void analyze();

  // 分析单个操作及其嵌套操作
  void analyzeOperation(Operation *op, 
      DependencyGraph::DependencyType loopRelation = DependencyGraph::IntraLoop);

  // 获取依赖关系图
  const DependencyGraph &getDependencyGraph() const;

  // 按照区域中操作的顺序输出依赖关系
  void printDependenciesInOrder(raw_ostream &os = llvm::outs()) const;

private:
  // 被分析的区域
  Region *region;
  
  // 依赖关系图
  DependencyGraph graph;

  // 整个遍历过程中只需要维护一个以下的状态
  DenseMap<Value, mt::InstructionSchedulingInterface> defMap;
  DenseMap<Value, DenseSet<mt::InstructionSchedulingInterface>> useMap;

  template <typename OpRange>
  void printOperationDependencies(OpRange &&ops, raw_ostream &os, StringRef indent) const;
};

} // namespace mt
} // namespace mlir

#endif // MTAS_DIALECT_MT_TRANSFORMS_DEPENDENCYANALYZER_H