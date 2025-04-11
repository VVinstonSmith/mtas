//===--------- ConvertForOpIndexToI64.cpp - Convert ForOp Index Type to I64 Pass -----===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTFOROPINDEXTOI64
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

class ConvertForOpIndexToI64Pass : 
    public impl::ConvertForOpIndexToI64Base<ConvertForOpIndexToI64Pass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implConvertForOpIndexToI64(funcOp);
  }
private:
  void implConvertForOpIndexToI64(func::FuncOp funcOp){
    // 实现 ForOp index 类型到 i64 类型的转换
    auto ctx = funcOp.getContext();
    OpBuilder builder(ctx);

    // 遍历函数中的所有 scf.for 操作
    funcOp.walk([&](scf::ForOp forOp) {
      builder.setInsertionPoint(forOp);
      
      // 获取循环的下界、上界和步长
      Value lowerBound = forOp.getLowerBound();
      Value upperBound = forOp.getUpperBound();
      Value step = forOp.getStep();
      
      // 判断下界、上界和步长的类型是否为 index
      if (lowerBound.getType().isIndex() && 
          upperBound.getType().isIndex() && 
          step.getType().isIndex()) {
        
        // 将下界、上界和步长转换为 i64 类型
        lowerBound = builder.create<UnrealizedConversionCastOp>(forOp.getLoc(), builder.getI64Type(), lowerBound).getResult(0);
        upperBound = builder.create<UnrealizedConversionCastOp>(forOp.getLoc(), builder.getI64Type(), upperBound).getResult(0);
        step = builder.create<UnrealizedConversionCastOp>(forOp.getLoc(), builder.getI64Type(), step).getResult(0);
      }

      // 创建新的 ForOp，使用 i64 类型的下界、上界和步长
      auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        lowerBound,
        upperBound,
        step,
        forOp.getInitArgs(),
        [&](OpBuilder &nestedBuilder, Location loc, Value iv, ValueRange iterArgs){
          // 创建映射
          IRMapping mapper;
          // 映射迭代变量，添加必要的类型转换
          Value indexIV = nestedBuilder.create<UnrealizedConversionCastOp>(
              loc, 
              builder.getIndexType(), 
              iv).getResult(0);
          mapper.map(forOp.getInductionVar(), indexIV);
          // 克隆循环体中的操作
          for (auto &op : forOp.getRegion().front()) {
            nestedBuilder.clone(op, mapper);
          }
        });
      
      // 删除原始循环
      forOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mt::createConvertForOpIndexToI64Pass() {
  return std::make_unique<ConvertForOpIndexToI64Pass>();
}