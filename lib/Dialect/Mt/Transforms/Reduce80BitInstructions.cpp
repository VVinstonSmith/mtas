//===- Reduce80BitInstructions.cpp - Reduce 80-bit instructions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to reduce the number of 80-bit instructions.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_REDUCE80BITINSTRUCTIONS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class Reduce80BitInstructionsPass : 
    public impl::Reduce80BitInstructionsBase<Reduce80BitInstructionsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implReduce80BitInstructions(funcOp);
  }
private:
    void implReduce80BitInstructions(func::FuncOp funcOp) {
        // 一个map用于记录当前的寄存器的值
        DenseMap<mlir::Value, uint64_t> vectorRegMap;
        // 遍历函数体中的所有实现了InstructionSchedulingInterface的操作
        funcOp.walk([&](mt::InstructionSchedulingInterface op) {
            // 如果op是mt::VmoviOp操作
            if (auto vmoviOp = llvm::dyn_cast<mt::VmoviOp>(op.getOperation())) {
                auto imm = vmoviOp.getImm();
                auto reg = vmoviOp.getReg();
                // 遍历vectorRegMap，检查是否有相同的值
                bool found = false;
                for (auto &pair : vectorRegMap) {
                    if (pair.second == imm) {
                        // 如果找到了相同的值，替换VmoviOp为VmovOp
                        // 创建OpBuilder
                        mlir::OpBuilder builder(vmoviOp.getContext());
                        // 设置插入位置为当前操作的插入位置
                        builder.setInsertionPoint(vmoviOp);
                        // 创建VmovOp操作
                        auto vmovOp = builder.create<mt::VmovOp>(vmoviOp.getLoc(), pair.first, reg);
                        // 删除原来的VmoviOp操作
                        vmoviOp.erase();
                        found = true;
                        // // 输出funcOp
                        // llvm::outs() << funcOp << "\n";
                        break;
                    }
                }
                if(!found){
                    vectorRegMap[reg] = imm;
                }
            } else {
                auto writeRegs = op.getWrittenRegisters();
                for (auto writeReg : writeRegs) {
                    if(vectorRegMap.find(writeReg) != vectorRegMap.end()){
                        vectorRegMap.erase(writeReg);
                    }
                }
            }
        });
    }
};

}

std::unique_ptr<Pass> mlir::mt::createReduce80BitInstructionsPass() {
  return std::make_unique<Reduce80BitInstructionsPass>();
}