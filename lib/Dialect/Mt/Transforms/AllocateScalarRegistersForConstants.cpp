//===--------- AllocateScalarRegistersForConstants.cpp - Scalar Register Allocation for Constants Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCATESCALARREGISTERSFORCONSTANTS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class AllocateScalarRegistersForConstantsPass : 
    public impl::AllocateScalarRegistersForConstantsBase<AllocateScalarRegistersForConstantsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implAllocateScalarRegistersForConstants(funcOp);
  }
private:
  // 定义可用寄存器ID范围
  const std::vector<std::pair<int64_t, int64_t>> availableRanges = {
    {7, 31},   // 第一个范围
    {42, 61}   // 第二个范围
  };

  // 查找当前已使用的最大寄存器ID
  int64_t findMaxUsedRegisterId(func::FuncOp &funcOp) {
    int64_t maxId = -1;
    
    // 遍历所有的DeclareRegisterOp操作
    funcOp.walk([&](ftm::DeclareRegisterOp declareOp) {
      // 检查是否包含scalar_register属性
      if (!declareOp->hasAttr("scalar_register"))
        return;
      
      // 检查是否有RegisterId属性
      if (auto regIdAttr = declareOp->getAttrOfType<ftm::RegisterIdAttr>(ftm::RegisterIdAttr::name)) {
        maxId = std::max<int64_t>(maxId, regIdAttr.getId());
      }
    });
    
    return maxId;
  }

  // 获取下一个可用的寄存器ID
  int64_t getNextAvailableId(int64_t currentId) {
    // 遍历所有可用范围，找到适合的ID
    for (const auto &range : availableRanges) {
      int64_t start = range.first;
      int64_t end = range.second;
      
      // 如果当前ID小于起始ID，则返回起始ID
      if (currentId < start) {
        return start;
      }
      
      // 如果当前ID在范围内但未到范围末尾，则返回下一个ID
      if (currentId >= start && currentId <= end) {
        return currentId + 1;
      }
      
      // 如果已到当前范围末尾，继续检查下一个范围
    }
    
    // 如果所有范围都已用尽，抛出错误或返回一个特殊值
    llvm::report_fatal_error("No available register IDs left for allocation");
    return -1;
  }

  void implAllocateScalarRegistersForConstants(func::FuncOp funcOp){
    // 实现常量到标量寄存器的转换
    auto ctx = funcOp.getContext();
    OpBuilder builder(ctx);
    
    // 查找当前已使用的最大寄存器ID
    int64_t maxUsedId = findMaxUsedRegisterId(funcOp);
    int64_t nextId = getNextAvailableId(maxUsedId);

    // 遍历函数中的所有常量操作arith.constant
    funcOp.walk([&](LLVM::ConstantOp constantOp) {
      builder.setInsertionPointAfter(constantOp);
      // 为每个常量创建一个标量寄存器
      auto declareR = builder.create<ftm::DeclareRegisterOp>(
        funcOp.getLoc(), builder.getI64Type());
      declareR->setAttr(ftm::MemLevelAttr::name, 
        ftm::MemLevelAttr::get(ctx, ftm::Cache::ScalarRegister));
      declareR->setAttr(ftm::RegisterIdAttr::name, 
        ftm::RegisterIdAttr::get(ctx, nextId));
      // 替换常量操作为对应的寄存器操作
      constantOp.replaceAllUsesWith(declareR.getResult());
      // 创建SmoviOp移动常量值到寄存器
      auto intValue = constantOp.getValueAttr().cast<IntegerAttr>().getInt();
      builder.create<mt::SmoviOp>(constantOp.getLoc(),
             IntegerAttr::get(builder.getI64Type(), intValue),
             declareR.getResult());

      // 准备下一个ID
      nextId = getNextAvailableId(nextId);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mt::createAllocateScalarRegistersForConstantsPass() {
  return std::make_unique<AllocateScalarRegistersForConstantsPass>();
}