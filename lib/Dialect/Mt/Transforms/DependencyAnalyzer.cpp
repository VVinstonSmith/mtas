//===- DependencyAnalyzer.cpp - Instruction Dependency Analyzer --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DependencyAnalyzer class for instruction dependency
// analysis.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/Transforms/DependencyAnalyzer.h"
#include "mtas/Dialect/Mt/IR/InstructionSchedulingInterface.h"

#include "mtas/Dialect/Mt/IR/Mt.h"

using namespace mlir;
using namespace mt;

DependencyAnalyzer::DependencyAnalyzer(Region *region) 
  : region(region), graph() {
}

DependencyAnalyzer DependencyAnalyzer::forFunction(func::FuncOp funcOp) {
  return DependencyAnalyzer(&funcOp.getBody());
}

DependencyAnalyzer DependencyAnalyzer::forForOp(scf::ForOp forOp) {
  return DependencyAnalyzer(&forOp.getRegion());
}

void DependencyAnalyzer::analyze() {
  // 清空之前的分析结果
  graph.clear();
  defMap.clear();
  useMap.clear();

  // 确保有有效的区域可分析
  if (!region) return;
  
  // 遍历指定区域中的所有操作
  for (Operation &op : region->getOps()) {
    analyzeOperation(&op);
  }
  // // 如果region来自于ForOp，运行第二次，并添加循环间依赖
  // if (isa<scf::ForOp>(region->getParentOp())) {
  //   // 第二次遍历，标记为循环间依赖
  //   for (Operation &nestedOp : region->getOps()) {
  //     analyzeOperation(&nestedOp, DependencyGraph::InterLoop);
  //   }
  // }
}

void DependencyAnalyzer::analyzeOperation(Operation *op, 
                                          DependencyGraph::DependencyType loopRelation) {
  // 递归分析嵌套区域中的操作
  for (Region &nestedRegion : op->getRegions()) {
    for (Operation &nestedOp : nestedRegion.getOps()) {
      analyzeOperation(&nestedOp);
    }
    // 如果region来自于ForOp，运行第二次，并添加循环间依赖
    if (isa<scf::ForOp>(nestedRegion.getParentOp())) {
      // 第二次遍历，标记为循环间依赖
      for (Operation &nestedOp : nestedRegion.getOps()) {
        analyzeOperation(&nestedOp, DependencyGraph::InterLoop);
      }
    }
  }

  auto interface = dyn_cast<mt::InstructionSchedulingInterface>(op);
  if(interface == nullptr)
    return;

  // OpPrintingFlags flags;
  // // 输出defMap和useMap的当前状态
  // llvm::outs() << "当前操作: " << *op << "\n";
  // llvm::outs() << "当前defMap: \n";
  // for (auto [reg, defOp] : defMap) {
  //   // 只输出寄存器名称
  //   reg.printAsOperand(llvm::outs(), flags);
  //   llvm::outs() << " -> " << *defOp << "\n";
  // }
  // llvm::outs() << "当前useMap: \n";
  // for (auto [reg, ops] : useMap) {
  //   // 只输出寄存器名称
  //   if(ops.empty()) {
  //     continue;
  //   }
  //   reg.printAsOperand(llvm::outs(), flags);
  //   llvm::outs() << " -> ";
  //   for (auto useOp : ops) {
  //     llvm::outs() << *useOp << "\n";
  //   }
  // }
  // llvm::outs() << "\n\n";

  // 定义真依赖和反依赖
  DependencyGraph::DependencyType trueDependency = 
      static_cast<DependencyGraph::DependencyType>(DependencyGraph::TrueDependency | loopRelation);
  DependencyGraph::DependencyType antiDependency = 
      static_cast<DependencyGraph::DependencyType>(DependencyGraph::AntiDependency | loopRelation);

  // 读寄存器
  auto readRegs = interface.getReadRegisters();
  for (auto readReg : readRegs) {
    useMap[readReg].insert(interface);
    // 定义readReg的操作到op存在真依赖
    if (defMap.contains(readReg)) {
      graph.addDependency(defMap[readReg], interface, trueDependency);
    }
  }
  
  // 写寄存器
  auto writeRegs = interface.getWrittenRegisters();
  for (auto writeReg : writeRegs) {
    // 使用writeReg的所有操作到op存在反依赖
    for (auto readOp : useMap[writeReg]) {
      if (readOp.getOperation() != op) {
        graph.addDependency(readOp, interface, antiDependency);
      }
    }
    // 更新defMap和useMap
    defMap[writeReg] = interface;
    useMap[writeReg].clear();
  }
}

const DependencyGraph &DependencyAnalyzer::getDependencyGraph() const {
  return graph;
}

void DependencyAnalyzer::printDependenciesInOrder(raw_ostream &os) const {
  if (!region) {
    os << "无有效区域可分析\n";
    return;
  }

  // 打印区域内的所有操作依赖
  printOperationDependencies(region->getOps(), os, "");
}

template <typename OpRange>
void DependencyAnalyzer::printOperationDependencies(OpRange &&ops, raw_ostream &os, StringRef indent) const {
  for (Operation &op : ops) {
    // 处理嵌套区域中的操作
    for (Region &nestedRegion : op.getRegions()) {
      if (!nestedRegion.empty()) {
        os << indent << "嵌套区域中的操作:\n";
        // 创建新的缩进字符串，而不是使用 Twine 对象
        std::string newIndent = (indent + "  ").str();
        printOperationDependencies(nestedRegion.getOps(), os, newIndent);
      }
    }

    // 如果不是寄存器读写接口，则跳过
    auto interface = dyn_cast<mt::InstructionSchedulingInterface>(&op);
    if (!interface)
      continue;
      
    // 输出当前操作
    os << indent << op << "\n";
    
    // 输出当前操作的出边（被哪些操作依赖）
    auto outDeps = graph.getOutgoingDependencies(interface);
    if (!outDeps.empty()) {
      os << indent << "被以下操作依赖:\n";
      for (auto [target, type, latency] : outDeps) {
        os << indent << "  " << *target.getOperation() << "  ";
        os << DependencyGraph::typeToString(type) << "  ";
        os << "延迟: " << latency << "\n";
      }
    } else {
      os << indent << "没有被其他操作依赖\n";
    }
    
    // // 输出当前操作的入边（依赖哪些操作）
    // auto inDeps = graph.getIncomingDependencies(interface);
    // if (!inDeps.empty()) {
    //   os << indent << "依赖以下操作:\n";
    //   for (auto [source, type, latency] : inDeps) {
    //     os << indent << "  " << *source.getOperation() << "  ";
    //     os << DependencyGraph::typeToString(type) << "  ";
    //     os << "延迟: " << latency << "\n";
    //   }
    // } else {
    //   os << indent << "不依赖其他操作\n";
    // }
    
    os << "\n"; // 操作之间添加空行分隔
  }
}