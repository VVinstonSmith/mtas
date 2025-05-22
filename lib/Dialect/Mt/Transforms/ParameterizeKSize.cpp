//===- ParameterizeKSize.cpp - Parameterize K dimension size ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parameterize K dimension size by adding 
// function parameter and replacing hardcoded loop bounds.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_PARAMETERIZEKSIZE
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class ParameterizeKSizePass : 
    public impl::ParameterizeKSizeBase<ParameterizeKSizePass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implParameterizeKSize(funcOp);
  }
private:
    // 第一步：添加 i64 类型的函数参数
    Value addFunctionParameter(func::FuncOp funcOp) {
        // 获取当前函数类型
        auto currentFuncType = funcOp.getFunctionType();
        
        // 创建新的参数类型列表（在原有参数基础上添加 i64）
        SmallVector<Type> newInputTypes(currentFuncType.getInputs());
        newInputTypes.push_back(IntegerType::get(funcOp.getContext(), 64));
        
        // 创建新的函数类型
        auto newFuncType = FunctionType::get(
            funcOp.getContext(),
            newInputTypes,
            currentFuncType.getResults()
        );
        
        // 更新函数类型
        funcOp.setType(newFuncType);
        
        // 在函数体的入口块添加新的参数
        Block &entryBlock = funcOp.getBody().front();
        Value newParam = entryBlock.addArgument(
            IntegerType::get(funcOp.getContext(), 64), 
            funcOp.getLoc()
        );
        
        return newParam;
    }
    
    // 第二步：找到循环的上界常量
    std::pair<Value, Operation*> findLoopUpperBoundConstant(func::FuncOp funcOp) {
        Value loopUpperBound = nullptr;
        Operation* constantOp = nullptr;
        
        // 遍历函数中的所有 scf.for 操作
        funcOp.walk([&](scf::ForOp forOp) {
            Value upperBound = forOp.getUpperBound();
            
            // 检查上界是否来自常量操作
            if (auto definingOp = upperBound.getDefiningOp()) {
                if (auto constOp = dyn_cast<LLVM::ConstantOp>(definingOp)) {
                    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                        // 找到第一个整数常量上界
                        loopUpperBound = upperBound;
                        constantOp = constOp;
                        return WalkResult::interrupt();
                    }
                }
            }
            return WalkResult::advance();
        });
        
        return std::make_pair(loopUpperBound, constantOp);
    }
    
    // 第三步：用新的参数替换硬编码的循环上界
    void replaceConstantWithParameter(func::FuncOp funcOp, Value newParam, Operation* constantOp) {
        OpBuilder builder(funcOp.getContext());

        // 找到常量操作的位置并替换
        builder.setInsertionPoint(constantOp);
        
        // 创建寄存器声明操作
        auto regOp = builder.create<ftm::DeclareRegisterOp>(
            funcOp.getLoc(), 
            newParam.getType()  // i64 类型
        );
        
        // 设置内存级别属性为标量寄存器
        regOp->setAttr(ftm::MemLevelAttr::name, 
            ftm::MemLevelAttr::get(funcOp.getContext(), ftm::Cache::ScalarRegister));
        
        // 创建 SmovOp 操作，将参数值移动到寄存器
        builder.create<mt::SmovOp>(funcOp.getLoc(), newParam, regOp.getResult());
        
        // 用寄存器值替换原来的常量
        constantOp->getResult(0).replaceAllUsesWith(regOp.getResult());
        
        // 删除原常量操作
        constantOp->erase();
    }
    
    void implParameterizeKSize(func::FuncOp funcOp) {
        // 第一步：添加函数参数
        Value newParam = addFunctionParameter(funcOp);
        
        // 第二步：找到循环上界常量
        auto [loopUpperBound, constantOp] = findLoopUpperBoundConstant(funcOp);
        
        if (!loopUpperBound || !constantOp) {
            return; // 没找到合适的常量，直接返回
        }
        
        // 第三步：替换常量为参数
        replaceConstantWithParameter(funcOp, newParam, constantOp);
    }
};

}

std::unique_ptr<Pass> mlir::mt::createParameterizeKSizePass() {
  return std::make_unique<ParameterizeKSizePass>();
}