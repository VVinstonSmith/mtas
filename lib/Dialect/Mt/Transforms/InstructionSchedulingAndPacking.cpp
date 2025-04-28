//===- InstructionSchedulingAndPacking.cpp - Lower Ftm to Mt dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to lower Ftm dialect operations to Mt dialect.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"
#include "mtas/Dialect/Mt/Transforms/DependencyGraph.h"
#include "mtas/Dialect/Mt/Transforms/DependencyAnalyzer.h"
#include "mtas/Dialect/Mt/Transforms/ExecutionPacketUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <queue>
#include <set>
#include <unordered_set>

namespace mlir {
#define GEN_PASS_DEF_INSTRUCTIONSCHEDULINGANDPACKING
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class InstructionSchedulingAndPackingPass : 
    public impl::InstructionSchedulingAndPackingBase<InstructionSchedulingAndPackingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implInstructionSchedulingAndPacking(funcOp);
  }
private:
  // 查找函数中唯一的 ForOp，如果有且仅有一个则返回该 ForOp，否则发出错误并返回 nullptr
  scf::ForOp findUniqueForOp(func::FuncOp funcOp) {
    // 用于存储找到的 ForOp
    scf::ForOp targetForOp;
    // 计数器
    int forOpCount = 0;
    
    // 遍历函数体寻找 ForOp
    funcOp.walk([&](scf::ForOp forOp) {
      forOpCount++;
      targetForOp = forOp;
    });
    
    // 检查是否只有一个 ForOp
    if (forOpCount == 0) {
      funcOp.emitError("当前函数中没有找到 ForOp");
      return scf::ForOp();
    } else if (forOpCount > 1) {
      funcOp.emitError("当前函数中存在多个 ForOp，预期只有一个");
      return scf::ForOp();
    }
    
    // 只有一个 ForOp，返回
    return targetForOp;
  }

  std::tuple<int, int, int> getMatrixDimensions(scf::ForOp forOp) {
    int k_size = -1, m_size = -1, n_size = -1;
    forOp.walk([&](mt::Vfmulas32Op vfmulas32Op) {
      if (vfmulas32Op->hasAttr("matmul.k")){
        auto k = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.k").getInt();
        if(k > k_size) k_size = k;
      }
      if (vfmulas32Op->hasAttr("matmul.m")){
        auto m = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.m").getInt();
        if(m > m_size) m_size = m;
      }
      if (vfmulas32Op->hasAttr("matmul.n")){
        auto n = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.n").getInt();
        if(n > n_size) n_size = n;
      }
    });
    return {k_size + 1, m_size + 1, n_size + 1};
  }

  // 计算调度窗口大小
  int calculateWindowSize(int k_size, int m_size, int n_size) {
    SmallVector<int, 3> windowSizeCandidates;
    windowSizeCandidates.push_back(7); // 跳转指令的延迟
    windowSizeCandidates.push_back(6 * k_size); // 基于 k 维度的大小
    windowSizeCandidates.push_back((k_size * m_size * n_size + 2) / 3); // 基于整体计算量的大小
    return *std::max_element(windowSizeCandidates.begin(), windowSizeCandidates.end());
  }

  // 反向调度循环内操作
  void backwardScheduleLoopOps(const DependencyGraph &graph, 
                            std::set<mt::InstructionSchedulingInterface> &inLoopOps,
                            std::map<mt::InstructionSchedulingInterface, int> &scheduledOps,
                            std::vector<ExecutionPacket> &schedulingWindow,
                            int m_size, int n_size,
                            int windowSize) {
    // 要调度的操作集合
    std::set<mt::InstructionSchedulingInterface> toScheduleOps(inLoopOps);
    // 一个数组用于表示当前活跃的依赖边（其dst已经被调度，src还不能调度），形式为(src, dst, type, latency)
    std::vector<std::tuple<mt::InstructionSchedulingInterface, mt::InstructionSchedulingInterface, DependencyGraph::DependencyType, int>> activeEdges;
    // 依赖已经满足的操作
    std::vector<mt::InstructionSchedulingInterface> readyOps;

    int startCycle = 2 * windowSize - 1;
    int endCycle = 0;

    for (auto it = toScheduleOps.begin(); it != toScheduleOps.end();) {
      auto op = *it;
      llvm::outs() << "操作 " << *op  << "\n";
      auto next_it = std::next(it); // 提前保存下一个迭代器
      // 判断该操作的出边是否都在循环外
      bool canSchedule = true;
      for (auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(op)) {
        // 如果target在循环内，不能进行调度
        llvm::outs() << *target.getOperation() << "\n";
        if (inLoopOps.find(target) != inLoopOps.end()) {
          canSchedule = false;
          break;
        }
      }
      if (canSchedule) {
        bool success;
        int cycle;
        if (auto vfmulas32Op = dyn_cast<mt::Vfmulas32Op>(op.getOperation())) {
          int m = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.m").getInt();
          int n = vfmulas32Op->getAttrOfType<IntegerAttr>("matmul.n").getInt();
          int baseCycle = startCycle - (std::max(op.getLatency(), (m_size * n_size + 2) / 3) - 1);
          // 根据n_size的大小决定放置逻辑
          if (n_size <= 3) {
            // 如果n_size小于等于3，可以直接放置在baseCycle+m行
            cycle = baseCycle + m;
          } else {
            // 计算(m,n)的编号并确定放置行
            int index = m * n_size + n;
            int offset = index / 3; // 对3取整
            cycle = baseCycle + offset;
          }
          success = schedulingWindow[cycle % windowSize].addOperation(op);
        } else {
          cycle = startCycle - (op.getLatency() - 1);
          while(cycle >= 0){
            success = schedulingWindow[cycle % windowSize].addOperation(op);
            if(success)
              break;
            cycle--;
          }
        }
        if(!success){
          // 报错并终止程序
          llvm::errs() << "调度失败，操作 " << *op << " 无法被调度到任何周期\n";
          signalPassFailure();
          return;
        }
        llvm::outs() << "操作 " << *op << " 被调度到周期 " << cycle << "\n";
        scheduledOps[op] = cycle;
        toScheduleOps.erase(op);
        // 将该操作的入边添加到活跃边中
        auto inDeps = graph.getIntraLoopIncomingDependencies(op);
        for (auto [source, type, latency] : inDeps) {
          if (inLoopOps.find(source) == inLoopOps.end()) {
            continue;
          }
          activeEdges.push_back(std::make_tuple(source, op, type, latency));
        }
      }
      it = next_it; // 无论是否删除，都使用预先保存的下一个迭代器
    }

    // // 输出当前的调度窗口
    // llvm::outs() << "当前调度窗口:\n";
    // printSchedulingWindowAsTable(schedulingWindow);

    // // 输出所有活跃边
    // llvm::outs() << "当前活跃边:\n";
    // for (auto [source, target, type, latency] : activeEdges) {
    //   llvm::outs() << "  " << *source.getOperation() << " -> " << *target.getOperation() << "  ";
    //   llvm::outs() << DependencyGraph::typeToString(type) << "  ";
    //   llvm::outs() << "延迟: " << latency << "\n";
    // }

    int cycle = startCycle;
    while(cycle >= endCycle){
      // 遍历所有活跃边
      for (auto it = activeEdges.begin(); it != activeEdges.end();) {
        auto [source, target, type, latency] = *it;
        // 检查源操作是否可以被调度
        if(scheduledOps.find(target) != scheduledOps.end() && cycle + latency <= scheduledOps[target]){
          // 从活跃边中删除该边
          it = activeEdges.erase(it);
          // 检查源操作的所有出边是否都满足条件
          bool allReady = true;
          for(auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(source)){
            // 有任何一个出边不满足条件，则不能调度该操作
            if(scheduledOps.find(target) == scheduledOps.end() || cycle + latency > scheduledOps[target]){
              allReady = false;
              break;
            }
          }
          if(allReady){
            // 如果所有出边都满足条件，则将源操作添加到准备调度的操作列表中
            llvm::outs() << "操作 " << *source.getOperation() << " 的所有出边都满足条件\n";
            readyOps.push_back(source);
          }
        } else {
          ++it;
        }
      }
        
      // 遍历所有准备调度的操作
      for(auto it = readyOps.begin(); it != readyOps.end();){
        auto op = *it;
        if(scheduledOps.find(op) != scheduledOps.end()){
          // 如果该操作已经被调度，则跳过
          llvm::outs() << "操作 " << *op << " 已经被调度到周期 " << scheduledOps[op] << "\n";
          it = readyOps.erase(it);
          continue;
        }
        bool success = schedulingWindow[cycle % windowSize].addOperation(op);
        if(success){
          it = readyOps.erase(it);
          llvm::outs() << "操作 " << *op << " 被调度到周期 " << cycle << "\n";
          // 将该操作添加到已调度操作集中
          scheduledOps[op] = cycle;
          // 从待调度操作集中删除该操作
          toScheduleOps.erase(op);
          
          // 将该操作的入边添加到活跃边中
          for (auto [source, type, latency] : graph.getIntraLoopIncomingDependencies(op)) {
            // 如果该依赖的源操作不属于循环内的操作，则跳过
            if(inLoopOps.find(source) == inLoopOps.end()){
              continue;
            }
            // 如果边的latency是0，直接检查该边的src是否可以加入readyOps
            if(latency == 0){
              bool allReady = true;
              for(auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(source)){
                // 有任何一个出边不满足条件，则不能调度该操作
                if(scheduledOps.find(target) == scheduledOps.end() || cycle + latency > scheduledOps[target]){
                  allReady = false;
                  break;
                }
              }
              if(allReady){
                llvm::outs() << "操作 " << *source.getOperation() << " 的所有出边都满足条件\n";
                readyOps.push_back(source);
              }
            } else {
              // 如果边的latency不为0，直接将该边加入活跃边中
              activeEdges.push_back(std::make_tuple(source, op, type, latency));
            }
          }
        } else {
          llvm::outs() << "操作 " << *op << " 无法被调度到周期 " << cycle << "\n";
          ++it;
        }
      }

      if(activeEdges.empty() && readyOps.empty()){
        break;
      }
      cycle--;
    }
    if(!toScheduleOps.empty()){
      llvm::errs() << "仍有操作未被调度:\n";
      for(auto op : toScheduleOps){
        llvm::errs() << "  " << *op.getOperation() << "\n";
      }
    }
  }

  // 微调循环操作的调度结果，解决循环间真依赖问题
  void adjustLoopScheduling(
      const DependencyGraph &graph,
      const std::set<mt::InstructionSchedulingInterface> &inLoopOps,
      std::map<mt::InstructionSchedulingInterface, int> &scheduledOps,
      int windowSize) {
      
    // 将循环中小于迭代窗口的操作加上迭代窗口的大小，记录为第二次迭代
    std::map<mt::InstructionSchedulingInterface, int> secondIterationScheduledOps;
    for(auto [op, cycle] : scheduledOps) {
      if(cycle < windowSize) {
        secondIterationScheduledOps[op] = cycle + windowSize;
      }
    }
    
    // 遍历第二次迭代中每个操作的入边，检查是否有循环间真依赖不满足
    for(auto [op, cycle] : secondIterationScheduledOps) {
      for (auto [source, type, latency] : graph.getInterLoopIncomingDependencies(op)) {
        // 如果source不属于循环内的操作，或依赖不属于循环间真依赖，则跳过
        if(inLoopOps.find(source) == inLoopOps.end() || !DependencyGraph::isTrueDependency(type)) {
          continue;
        }
        
        // 如果有不满足的依赖关系，将源操作的调度周期调整
        if(scheduledOps[source] + latency > cycle) {
          llvm::outs() << "操作 " << source << " 的调度周期由 " << scheduledOps[source] 
                      << " 改为 " << scheduledOps[source] - windowSize << "\n";
          scheduledOps[source] -= windowSize;
        }
      }
    }
  }

  std::set<mt::InstructionSchedulingInterface> findPostLoopOperations(
      const std::set<mt::InstructionSchedulingInterface> &inLoopOps,
      const DependencyGraph &graph) {
    std::set<mt::InstructionSchedulingInterface> postLoopOps;
    
    // 找到直接依赖于循环内操作的循环外操作
    for(auto op : inLoopOps){
      for (auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(op)) {
        if(inLoopOps.find(target) == inLoopOps.end()){
          postLoopOps.insert(target);
          break;
        }
      }
    }
    
    // 广度优先搜索找到所有可达的循环外操作
    std::queue<mt::InstructionSchedulingInterface> queue;
    for(auto op : postLoopOps){
      queue.push(op);
    }
    
    while(!queue.empty()){
      auto op = queue.front();
      queue.pop();
      for (auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(op)) {
        if(postLoopOps.find(target) == postLoopOps.end()){
          postLoopOps.insert(target);
          queue.push(target);
        }
      }
    }
    
    return postLoopOps;
  }

  void forwardSchedulePostLoopOps(const DependencyGraph &graph,
                                std::set<mt::InstructionSchedulingInterface> &postLoopOps,
                                std::map<mt::InstructionSchedulingInterface, int> &scheduledOps,
                                std::vector<ExecutionPacket> &packetsAfterLoop,
                                int startCycle) {
    // 要调度的操作集合
    std::set<mt::InstructionSchedulingInterface> toScheduleOps(postLoopOps);
    // 一个数组用于表示当前活跃的依赖边（其dst已经被调度，src还不能调度），形式为(src, dst, type, latency)
    std::vector<std::tuple<mt::InstructionSchedulingInterface, mt::InstructionSchedulingInterface, DependencyGraph::DependencyType, int>> activeEdges;
    // 依赖已经满足的操作
    std::vector<mt::InstructionSchedulingInterface> readyOps;
    // 从循环内到循环后的依赖边
    for(auto op : toScheduleOps){
      for (auto [source, type, latency] : graph.getIntraLoopIncomingDependencies(op)) {
        if(scheduledOps.find(source) != scheduledOps.end()){
          activeEdges.push_back(std::make_tuple(source, op, type, latency));
          break;
        }
      }
    }

    // 前向调度循环后的操作
    int cycle = 2 * startCycle;
    while(!toScheduleOps.empty()){
      // 创建新执行包
      packetsAfterLoop.push_back(ExecutionPacket());

      // 遍历所有活跃边
      for (auto it = activeEdges.begin(); it != activeEdges.end();) {
        auto [source, target, type, latency] = *it;
        // 检查目标操作是否可以被调度
        if(scheduledOps.find(source) != scheduledOps.end() && scheduledOps[source] + latency <= cycle){
          // 从活跃边中删除该边
          it = activeEdges.erase(it);
          // 检查目标操作的所有入边是否都满足条件
          bool allReady = true;
          for(auto [source, type, latency] : graph.getIntraLoopIncomingDependencies(target)){
            // 如果source既不属于待调度的操作，也不属于已经调度的操作，说明source是循环前的操作，暂时不考虑，因此跳过
            if(toScheduleOps.find(source) == toScheduleOps.end() && scheduledOps.find(source) == scheduledOps.end()){
              continue;
            }
            // 有任何一个入边不满足条件，则不能调度该操作
            if(scheduledOps.find(source) == scheduledOps.end() || scheduledOps[source] + latency > cycle){
              allReady = false;
              break;
            }
          }
          if(allReady){
            // 如果所有入边都满足条件，则将目标操作添加到准备调度的操作列表中
            llvm::outs() << "操作 " << *target.getOperation() << " 的所有入边都满足条件\n";
            readyOps.push_back(target);
          }
        } else {
          ++it;
        }
      }

      // 遍历所有准备调度的操作
      for(auto it = readyOps.begin(); it != readyOps.end();){
        auto op = *it;
        if(scheduledOps.find(op) != scheduledOps.end()){
          // 如果该操作已经被调度，则跳过
          llvm::outs() << "操作 " << *op << " 已经被调度到周期 " << scheduledOps[op] << "\n";
          it = readyOps.erase(it);
          continue;
        }
        bool success = packetsAfterLoop[cycle - 2 * startCycle].addOperation(op);
        if(success){
          it = readyOps.erase(it);
          llvm::outs() << "操作 " << *op << " 被调度到周期 " << cycle << "\n";
          // 将该操作添加到已调度操作集中
          scheduledOps[op] = cycle;
          // 从待调度操作集中删除该操作
          toScheduleOps.erase(op);
          
          // 将该操作的出边添加到活跃边中
          for (auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(op)) {
            // 如果边的latency是0，直接检查该边的tgt是否可以加入readyOps
            if(latency == 0){
              bool allReady = true;
              for(auto [source, type, latency] : graph.getIntraLoopIncomingDependencies(target)){
                // 如果source既不属于待调度的操作，也不属于已经调度的操作，说明source是循环前的操作，因此跳过
                if(toScheduleOps.find(source) == toScheduleOps.end() && scheduledOps.find(source) == scheduledOps.end()){
                  continue;
                }
                // 有任何一个入边不满足条件，则不能调度该操作
                if(scheduledOps.find(source) == scheduledOps.end()  || scheduledOps[source] + latency > cycle){
                  allReady = false;
                  break;
                }
              }
              if(allReady){
                llvm::outs() << "操作 " << *target.getOperation() << " 的所有入边都满足条件\n";
                readyOps.push_back(target);
              }
            } else {
              // 如果边的latency不为0，直接将该边加入活跃边中
              activeEdges.push_back(std::make_tuple(op, target, type, latency));
            }
          }
        } else {
          llvm::outs() << "操作 " << *op << " 无法被调度到周期 " << cycle << "\n";
          ++it;
        }
      }

      if(activeEdges.empty() && readyOps.empty()){
        break;
      }
      cycle++;
    }
  }

  // 实现软件流水线的prologue
  void implementPrologueForSoftwarePipelining(
      const std::vector<ExecutionPacket> &schedulingWindow,
      std::vector<ExecutionPacket> &packetsBeforeLoop,
      const std::map<mt::InstructionSchedulingInterface, int> &scheduledOps,
      int windowSize) {
      
    // 预分配prologue的空间
    packetsBeforeLoop.resize(windowSize);
    
    // 定义过滤条件：只保留那些调度周期在窗口大小范围内的操作
    auto filterCondition = [&scheduledOps, windowSize](InstructionSchedulingInterface op) -> bool {
      return op && 
            scheduledOps.find(op) != scheduledOps.end() && 
            scheduledOps.at(op) < windowSize;
    };
    
    // 将schedulingWindow中的执行包以逆序方式复制到packetsBeforeLoop
    for (int i = 0; i < windowSize; ++i) {
      // 逆序复制，并应用过滤条件
      packetsBeforeLoop[i] = schedulingWindow[windowSize - 1 - i].createFilteredCopy(filterCondition);
    }
  }


  std::set<mt::InstructionSchedulingInterface> findPreLoopOperations(
      func::FuncOp funcOp,
      const DependencyGraph &graph,
      std::map<mt::InstructionSchedulingInterface, int> &scheduledOps) {
    std::set<mt::InstructionSchedulingInterface> preLoopOps;
    
    // 遍历函数体，找到所有操作
    funcOp.walk([&](mt::InstructionSchedulingInterface op) {
      // 如果操作不在已调度操作集中，将其添加到preLoopOps中
      if (scheduledOps.find(op) == scheduledOps.end()) {
        preLoopOps.insert(op);
      }
    });
    
    return preLoopOps;
  }


  void backwardSchedulePreLoopOps(const DependencyGraph &graph,
                                std::set<mt::InstructionSchedulingInterface> &preLoopOps,
                                std::map<mt::InstructionSchedulingInterface, int> &scheduledOps,
                                std::vector<ExecutionPacket> &packetsBeforeLoop,
                                int startCycle) {
    // 要调度的操作集合
    std::set<mt::InstructionSchedulingInterface> toScheduleOps(preLoopOps);
    // 一个数组用于表示当前活跃的依赖边（其dst已经被调度，src还不能调度），形式为(src, dst, type, latency)
    std::vector<std::tuple<mt::InstructionSchedulingInterface, mt::InstructionSchedulingInterface, DependencyGraph::DependencyType, int>> activeEdges;
    // 依赖已经满足的操作
    std::vector<mt::InstructionSchedulingInterface> readyOps;
    // 活动边为循环前的操作的出边
    for(auto op : toScheduleOps){
      for (auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(op)) {
        // target不在要调度的操作内
        if(toScheduleOps.find(target) == toScheduleOps.end()){
          activeEdges.push_back(std::make_tuple(op, target, type, latency));
          break;
        }
      }
    }

    // // 输出活动边
    // llvm::outs() << "当前活跃边:\n";
    // for (auto [source, target, type, latency] : activeEdges) {
    //   llvm::outs() << "  " << *source.getOperation() << " -> " << *target.getOperation() << "  ";
    //   llvm::outs() << DependencyGraph::typeToString(type) << "  ";
    //   llvm::outs() << "延迟: " << latency << "\n";
    // }

    // 开始周期为windowSize - 1，不断递减
    int cycle = startCycle - 1;
    // 结束条件为toScheduleOps中的所有操作都被调度
    while(!toScheduleOps.empty()){
      // 调度包索引和cycle的关系为 调度索引等于windowSize - 1 - cycle
      int packetIndex = startCycle - 1 - cycle;
      // 如果索引超出范围，创建新的执行包
      if(packetIndex >= packetsBeforeLoop.size()){
        packetsBeforeLoop.push_back(ExecutionPacket());
      }
      // 获取当前执行包
      auto &currentPacket = packetsBeforeLoop[packetIndex];

      // 遍历所有活跃边
      for (auto it = activeEdges.begin(); it != activeEdges.end();) {
        auto [source, target, type, latency] = *it;
        // 检查源操作是否可以被调度
        if(scheduledOps.find(target) != scheduledOps.end() && cycle + latency <= scheduledOps[target]){

          // 检查源操作的所有出边是否都满足条件
          bool allReady = true;
          for(auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(source)){
            // // 输出当前操作的出边不满足条件，以及scheduledOps[target]
            // llvm::outs() << *source.getOperation() << "->" << *target.getOperation() << "  ";
            // llvm::outs() << DependencyGraph::typeToString(type) << "  ";
            // llvm::outs() << "延迟: " << latency << "\n";
            // 有任何一个出边不满足条件，则不能调度该操作
            if(scheduledOps.find(target) == scheduledOps.end() || cycle + latency > scheduledOps[target]){
              allReady = false;
            }
          }
          if(allReady){
            // 如果所有出边都满足条件，则将源操作添加到准备调度的操作列表中
            llvm::outs() << "操作 " << *source.getOperation() << " 的所有出边都满足条件\n";
            readyOps.push_back(source);
            // 从活跃边中删除该边
            it = activeEdges.erase(it);
          } else {
            ++it;
          }
        } else {
          ++it;
        }
      }

      // 遍历所有准备调度的操作
      for(auto it = readyOps.begin(); it != readyOps.end();){
        auto op = *it;
        if(scheduledOps.find(op) != scheduledOps.end()){
          // 如果该操作已经被调度，则跳过
          llvm::outs() << "操作 " << *op << " 已经被调度到周期 " << scheduledOps[op] << "\n";
          it = readyOps.erase(it);
          continue;
        }
        bool success = currentPacket.addOperation(op);
        if(success){
          it = readyOps.erase(it);
          llvm::outs() << "操作 " << *op << " 被调度到周期 " << cycle << "\n";
          // 将该操作添加到已调度操作集中
          scheduledOps[op] = cycle;
          // 从待调度操作集中删除该操作
          toScheduleOps.erase(op);
          
          // 将该操作的入边添加到活跃边中
          for (auto [source, type, latency] : graph.getIntraLoopIncomingDependencies(op)) {
            // 如果该依赖的源操作不属于循环前的操作，则跳过
            if(preLoopOps.find(source) == preLoopOps.end()){
              continue;
            }
            // 如果边的latency是0，直接检查该边的src是否可以加入readyOps
            if(latency == 0){
              bool allReady = true;
              for(auto [target, type, latency] : graph.getIntraLoopOutgoingDependencies(source)){
                // 有任何一个出边不满足条件，则不能调度该操作
                if(scheduledOps.find(target) == scheduledOps.end() || cycle + latency > scheduledOps[target]){
                  allReady = false;
                  break;
                }
              }
              if(allReady){
                llvm::outs() << "操作 " << *source.getOperation() << " 的所有出边都满足条件\n";
                readyOps.push_back(source);
              }
            } else {
              // 如果边的latency不为0，直接将该边加入活跃边中
              activeEdges.push_back(std::make_tuple(source, op, type, latency));
            }
          }
        } else {
          llvm::outs() << "操作 " << *op << " 无法被调度到周期 " << cycle << "\n";
          ++it;
        }
      }

      cycle--;
    }
  }

  void implInstructionSchedulingAndPacking(func::FuncOp funcOp){
    std::map<mt::InstructionSchedulingInterface, int> scheduledOps;

    // 找到唯一的 ForOp
    scf::ForOp forOp = findUniqueForOp(funcOp);
    if(!forOp){
      // 如果 findUniqueForOp 返回空对象，表示发生错误
      // 报错信息已经在 findUniqueForOp 中输出
      signalPassFailure();
      return;
    }

    // 分析函数内可调度操作的依赖关系
    DependencyAnalyzer analyzer = DependencyAnalyzer::forFunction(funcOp);
    analyzer.analyze();
    analyzer.printDependenciesInOrder();
    DependencyGraph graph = analyzer.getDependencyGraph();

    // 获取矩阵维度
    auto [k_size, m_size, n_size] = getMatrixDimensions(forOp);
    llvm::outs() << "k_size: " << k_size << "\n";
    llvm::outs() << "m_size: " << m_size << "\n";
    llvm::outs() << "n_size: " << n_size << "\n";

    // 计算调度窗口大小
    int windowSize = calculateWindowSize(k_size, m_size, n_size);
    llvm::outs() << "选择的循环窗口大小: " << windowSize << "\n";

    // 创建调度窗口
    std::vector<ExecutionPacket> schedulingWindow(windowSize);

    // 获取循环内的所有操作
    std::set<mt::InstructionSchedulingInterface> inLoopOps;
    forOp.walk([&](mt::InstructionSchedulingInterface op) {
      inLoopOps.insert(op);
    });
    // llvm::outs() << "循环内的操作:\n";
    // for(auto op : inLoopOps){
    //   llvm::outs() << "  " << *op.getOperation() << "\n";
    // }

    // 反向调度循环中的操作
    backwardScheduleLoopOps(graph, inLoopOps, scheduledOps, schedulingWindow, m_size, n_size, windowSize);
    // llvm::outs() << "最终调度窗口:\n";
    // printSchedulingWindowAsTable(schedulingWindow);

    // 对调度结果进行微调，解决循环间真依赖问题
    adjustLoopScheduling(graph, inLoopOps, scheduledOps, windowSize);
    // llvm::outs() << "最终调度结果:\n";
    // for(auto [op, cycle] : scheduledOps){
    //   llvm::outs() << "  " << op << " 被调度到周期 " << cycle << "\n";
    // }

    // 找到循环后的操作
    std::set<mt::InstructionSchedulingInterface> postLoopOps = 
        findPostLoopOperations(inLoopOps, graph);
    // llvm::outs() << "循环后的操作:\n";
    // for(auto op : postLoopOps){
    //   llvm::outs() << "  " << *op.getOperation() << "\n";
    // }

    // 前向调度循环后的操作
    std::vector<ExecutionPacket> packetsAfterLoop;
    forwardSchedulePostLoopOps(graph, postLoopOps, scheduledOps, packetsAfterLoop, windowSize);
    // llvm::outs() << "循环后操作的调度结果:\n";
    // printSchedulingWindowAsTable(packetsAfterLoop);

    // 实现软件流水线的prologue
    std::vector<ExecutionPacket> packetsBeforeLoop;
    implementPrologueForSoftwarePipelining(schedulingWindow, packetsBeforeLoop, scheduledOps, windowSize);
    // llvm::outs() << "软件流水线的prologue部分调度结果:\n";
    // printSchedulingWindowAsTable(packetsBeforeLoop, true);

    // 找到循环前的操作
    std::set<mt::InstructionSchedulingInterface> preLoopOps = 
        findPreLoopOperations(funcOp, graph, scheduledOps);
    // llvm::outs() << "循环前的操作:\n";
    // for(auto op : preLoopOps){
    //   llvm::outs() << "  " << *op.getOperation() << "\n";
    // }

    // // 输出scheduledOps，按照cycle的顺序输出
    // llvm::outs() << "调度结果:\n";
    // std::vector<std::pair<mt::InstructionSchedulingInterface, int>> sortedScheduledOps(scheduledOps.begin(), scheduledOps.end());
    // std::sort(sortedScheduledOps.begin(), sortedScheduledOps.end(), [](const auto &a, const auto &b) {
    //   return a.second < b.second;
    // });
    // for(auto [op, cycle] : sortedScheduledOps){
    //   llvm::outs() << "  " << *op.getOperation() << " 被调度到周期 " << cycle << "\n";
    // }

    // 反向调度循环前的操作
    backwardSchedulePreLoopOps(graph, preLoopOps, scheduledOps, packetsBeforeLoop, windowSize);
    
    // // 输出完整的调度结果
    // llvm::outs() << "调度结果:\n";
    // printSchedulingWindowAsTable(packetsBeforeLoop, true, windowSize - packetsBeforeLoop.size());
    // printSchedulingWindowAsTable(schedulingWindow, false, windowSize);
    // printSchedulingWindowAsTable(packetsAfterLoop, false, 2*windowSize);

    // std::vector<std::pair<mt::InstructionSchedulingInterface, int>> sortedScheduledOps(scheduledOps.begin(), scheduledOps.end());
    // // 输出scheduledOps，按照cycle的顺序输出
    // llvm::outs() << "调度结果:\n";
    // // sortedScheduledOps赋值为scheduledOps
    // sortedScheduledOps.clear();
    // sortedScheduledOps.insert(sortedScheduledOps.end(), scheduledOps.begin(), scheduledOps.end());
    // std::sort(sortedScheduledOps.begin(), sortedScheduledOps.end(), [](const auto &a, const auto &b) {
    //   return a.second < b.second;
    // });
    // for(auto [op, cycle] : sortedScheduledOps){
    //   llvm::outs() << "  " << *op.getOperation() << " 被调度到周期 " << cycle << "\n";
    // }

    // 反转packetsBeforeLoop
    std::reverse(packetsBeforeLoop.begin(), packetsBeforeLoop.end());
    printExecutionPackets(packetsBeforeLoop, schedulingWindow, packetsAfterLoop);
  }
};

}

std::unique_ptr<Pass> mlir::mt::createInstructionSchedulingAndPackingPass() {
  return std::make_unique<InstructionSchedulingAndPackingPass>();
}