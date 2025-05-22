//===- AssignRegisterIds.cpp - Assign register IDs ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to assign unique IDs to registers.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <set>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_ASSIGNREGISTERIDS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class RegisterIdManager {
 private:
  // 已使用/已分配的寄存器ID集合
  std::map<ftm::Cache, std::set<int64_t>> usedIds;

  // 可用范围定义
  const std::map<ftm::Cache, std::vector<std::pair<int64_t, int64_t>>>
      availableRanges = {
          {ftm::Cache::ScalarRegister,
           {{7, 9}, {26, 31}, {42, 61}}},                 // 标量寄存器范围
          {ftm::Cache::VectorRegister, {{0, 63}}},        // 向量寄存器范围
          {ftm::Cache::VectorAddressRegister, {{0, 6}}},  // 向量基址寄存器范围
          {ftm::Cache::ScalarAddressRegister,
           {{10, 15}}},                                    // 标量基址寄存器范围
          {ftm::Cache::VectorOffsetRegister, {{0, 7}}},    // 向量偏移寄存器范围
          {ftm::Cache::ScalarOffsetRegister, {{8, 15}}},   // 标量偏移寄存器范围
          {ftm::Cache::ScalarConditionRegister, {{0, 6}}}  // 标量条件寄存器范围
  };

 public:
  // 初始化，检查寄存器需求与可用数量，并记录已使用的ID
  // 如果任意一种寄存器类型的需求超出可用数量，返回false
  bool initialize(func::FuncOp &funcOp) {
    // 清空所有寄存器类型的已使用ID集合
    for (auto regType :
         {ftm::Cache::ScalarRegister, ftm::Cache::VectorRegister,
          ftm::Cache::VectorAddressRegister, ftm::Cache::ScalarAddressRegister,
          ftm::Cache::VectorOffsetRegister, ftm::Cache::ScalarOffsetRegister,
          ftm::Cache::ScalarConditionRegister}) {
      usedIds[regType] = {};
    }

    // 统计寄存器需求和记录已使用ID
    std::map<ftm::Cache, unsigned> registerNeeds;

    // 遍历所有的DeclareRegisterOp操作
    funcOp.walk([&](mt::DeclareRegisterOp declareOp) {
      if (auto memLevelAttr = declareOp->getAttrOfType<ftm::MemLevelAttr>(
              ftm::MemLevelAttr::name)) {
        ftm::Cache regType = memLevelAttr.getLevel();

        // 增加需求计数
        registerNeeds[regType]++;

        // 如果已有ID，则记录为已使用
        if (auto regIdAttr = declareOp->getAttrOfType<ftm::RegisterIdAttr>(
                ftm::RegisterIdAttr::name)) {
          usedIds[regType].insert(regIdAttr.getId());
        }
      }
    });

    // 检查每种寄存器类型的需求是否超出可用数量
    bool allDemandsSatisfiable = true;
    llvm::outs() << "Register demands and availability:\n";

    for (const auto &[regType, needs] : registerNeeds) {
      // 计算可用数量
      unsigned available = 0;
      if (availableRanges.count(regType)) {
        for (const auto &[start, end] : availableRanges.at(regType)) {
          available += end - start + 1;
        }
      }

      // 输出需求和可用情况
      llvm::outs() << "  " << stringifyCache(regType) << ": needs=" << needs
                   << ", available=" << available
                   << ", already used=" << usedIds[regType].size() << "\n";

      // 检查是否可满足需求
      if (needs > available) {
        llvm::outs() << "ERROR: Register type " << stringifyCache(regType)
                     << " demand exceeds available count\n";
        allDemandsSatisfiable = false;
      }
    }

    return allDemandsSatisfiable;
  }

  // 分配一个未使用的寄存器ID
  int64_t allocateId(ftm::Cache regType) {
    // 获取对应类型的可用范围
    const auto &ranges = availableRanges.at(regType);

    // 遍历所有可用范围，找到最小的未使用ID
    for (const auto &range : ranges) {
      int64_t start = range.first;
      int64_t end = range.second;

      // 在当前范围内查找最小的未使用ID
      for (int64_t id = start; id <= end; ++id) {
        if (usedIds[regType].find(id) == usedIds[regType].end()) {
          // 将新ID标记为已使用
          usedIds[regType].insert(id);
          return id;
        }
      }
    }

    // 如果所有范围都已用尽，抛出错误
    llvm::report_fatal_error(
        llvm::Twine("No available register IDs left for type ") +
        llvm::Twine(static_cast<uint64_t>(regType)));
    return -1;
  }

  // 分配两个连续的寄存器ID，且小的ID必须能够被2整除
  std::pair<int64_t, int64_t> allocatePairedIds(ftm::Cache regType) {
    // 获取对应类型的可用范围
    const auto &ranges = availableRanges.at(regType);

    // 遍历所有可用范围，寻找合适的连续ID对
    for (const auto &range : ranges) {
      int64_t start = range.first;
      int64_t end = range.second;

      // 如果起始ID不是偶数，调整为下一个偶数
      if (start % 2 != 0) {
        start++;
      }

      // 在当前范围内查找第一个可用的偶数ID，且其下一个ID也必须可用
      for (int64_t id = start; id < end; id += 2) {  // 每次增加2以保持偶数
        if (usedIds[regType].find(id) == usedIds[regType].end() && 
            usedIds[regType].find(id + 1) == usedIds[regType].end()) {
          // 将两个ID都标记为已使用
          usedIds[regType].insert(id);
          usedIds[regType].insert(id + 1);
          return {id, id + 1};
        }
      }
    }

    // 如果所有范围都无法找到合适的连续ID对，抛出错误
    llvm::report_fatal_error(
        llvm::Twine("No available paired register IDs left for type ") +
        llvm::Twine(static_cast<uint64_t>(regType)));
    return {-1, -1};
  }

  // 释放寄存器ID，将其标记为未使用
  void releaseId(ftm::Cache regType, int64_t id) { usedIds[regType].erase(id); }
};

class AssignRegisterIdsPass
    : public impl::AssignRegisterIdsBase<AssignRegisterIdsPass> {
 public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implAssignRegisterIds(funcOp);
  }

 private:
  void implAssignRegisterIds(func::FuncOp funcOp) {
    // 1. 创建RegisterIdManager实例
    RegisterIdManager manager;
    // 2.初始化管理器，统计函数中每种寄存器类型的需求，并记录已使用的register_id
    if (!manager.initialize(funcOp)) return;
    // 3. 遍历所有mt.declare_register操作
    funcOp.walk([&](mt::DeclareRegisterOp declareOp) {
      if (auto regIdAttr = declareOp->getAttrOfType<ftm::RegisterIdAttr>(
              ftm::RegisterIdAttr::name)) {
        return;
      }
      // 4. 为没有register_id的操作分配并添加ID
      auto regType = declareOp->getAttr(ftm::MemLevelAttr::name)
                         .cast<ftm::MemLevelAttr>()
                         .getLevel();

      // 如果是向量寄存器，且被VLDDW或VSTDW使用，需要为两个寄存器一起分配寄存器编号，
      // 也就是一次分配两个连续的编号，且小的编号必须能够被2整除
      if(regType == ftm::Cache::VectorRegister || regType == ftm::Cache::ScalarRegister){
        auto reg = declareOp.getResult();
        for(auto user : reg.getUsers()){
          if(auto dwOp = dyn_cast<mt::DoubleWordMemoryAccessInterface>(user)){
            auto pairedIds = manager.allocatePairedIds(regType);
            if(reg == dwOp.getHighRegister()){
              auto otherDelOp = dwOp.getLowRegister().getDefiningOp();
              declareOp->setAttr(ftm::RegisterIdAttr::name,
                                ftm::RegisterIdAttr::get(declareOp.getContext(), pairedIds.second));
              otherDelOp->setAttr(ftm::RegisterIdAttr::name,
                                ftm::RegisterIdAttr::get(declareOp.getContext(), pairedIds.first));
            } else if(reg == dwOp.getLowRegister()){
              auto otherDelOp = dwOp.getHighRegister().getDefiningOp();
              declareOp->setAttr(ftm::RegisterIdAttr::name,
                                ftm::RegisterIdAttr::get(declareOp.getContext(), pairedIds.first));
              otherDelOp->setAttr(ftm::RegisterIdAttr::name,
                                ftm::RegisterIdAttr::get(declareOp.getContext(), pairedIds.second));
            }
            return;
          }
        }
      }

      int64_t id = manager.allocateId(regType);
      declareOp->setAttr(ftm::RegisterIdAttr::name,
                         ftm::RegisterIdAttr::get(declareOp.getContext(), id));
    });
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::mt::createAssignRegisterIdsPass() {
  return std::make_unique<AssignRegisterIdsPass>();
}