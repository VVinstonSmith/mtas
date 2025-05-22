//===- LowerFtmToMt.cpp - Lower Ftm to Mt dialect -----------------------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <set>
#include <map>

namespace mlir {
#define GEN_PASS_DEF_LOWERFTMTOMT
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

// 基础模式匹配类，用于Ftm到Mt的转换
template <typename FtmOpTy>
class FtmToMtOpConversion : public OpConversionPattern<FtmOpTy> {
public:
  FtmToMtOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<FtmOpTy>(typeConverter, context){}

  // 辅助方法：创建寄存器并替换原操作
  Value createRegisterAndReplace(Operation *op, Type resultType,
                                 ftm::Cache regType,
                                 ConversionPatternRewriter &rewriter) const {
    // 设置插入位置到函数入口
    auto funcOp = op->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    
    // 创建新的寄存器
    auto regOp = rewriter.create<ftm::DeclareRegisterOp>(op->getLoc(), resultType);
    
    // 设置内存级别属性
    regOp->setAttr(ftm::MemLevelAttr::name, 
        ftm::MemLevelAttr::get(rewriter.getContext(), regType));
    
    // 替换原操作
    rewriter.replaceOp(op, regOp.getResult());
    
    // 恢复插入点到原操作位置
    rewriter.setInsertionPoint(op);
    
    return regOp.getResult();
  }
};

class LLVMConstantOpToMtSmoviOp : public FtmToMtOpConversion<LLVM::ConstantOp> {
public:
  using FtmToMtOpConversion<LLVM::ConstantOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 创建标量寄存器
    Value dst = createRegisterAndReplace(op, op.getType(), ftm::Cache::ScalarRegister, rewriter);
    
    // 获取常量值并创建SmoviOp
    auto intValue = op.getValue().cast<IntegerAttr>().getInt();
    rewriter.create<mt::SmoviOp>(op.getLoc(),
                              IntegerAttr::get(rewriter.getI64Type(), intValue),
                              dst);
    
    return success();
  }
};

class FtmLoadOpToMtLdwOp : public FtmToMtOpConversion<ftm::LoadOp> {
public:
  using FtmToMtOpConversion<ftm::LoadOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = adaptor.getAddr();
    ValueRange offsetValues = adaptor.getOffsetRegister();

    auto memLevelAttr = op->getAttr(ftm::MemLevelAttr::name);
    ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

    // 根据内存级别确定寄存器类型
    ftm::Cache regType = (memLevel == ftm::Cache::SM) ? 
                          ftm::Cache::ScalarRegister : 
                          ftm::Cache::VectorRegister;
    
    // 使用辅助方法创建寄存器并替换原操作
    Value dst = createRegisterAndReplace(op, op.getType(), regType, rewriter);

    // 根据内存级别创建不同的操作
    if (regType == ftm::Cache::ScalarRegister) {
      // 创建SldwOp
      if (offsetValues.empty()) {
        rewriter.create<mt::SldwOp>(op.getLoc(), base, dst, nullptr);
      } else {
        rewriter.create<mt::SldwOp>(op.getLoc(), base, dst, offsetValues.front());
      }
    } else {
      // 创建VldwOp
      if (offsetValues.empty()) {
        rewriter.create<mt::VldwOp>(op.getLoc(), base, dst, nullptr);
      } else {
        rewriter.create<mt::VldwOp>(op.getLoc(), base, dst, offsetValues.front());
      }
    }
    
    return success();
  }
};

class FtmStoreOpToMtStwOp : public FtmToMtOpConversion<ftm::StoreOp> {
public:
  using FtmToMtOpConversion<ftm::StoreOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Value base = adaptor.getAddr();
    ValueRange offsetValues = adaptor.getOffsetRegister();

    auto memLevelAttr = op->getAttr(ftm::MemLevelAttr::name);
    ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

    // 根据内存级别确定寄存器类型
    ftm::Cache regType = (memLevel == ftm::Cache::SM) ? 
                          ftm::Cache::ScalarRegister : 
                          ftm::Cache::VectorRegister;

    // 根据内存级别创建不同的操作
    if (regType == ftm::Cache::ScalarRegister) {
      // // 创建SstwOp
      // if (offsetValues.empty()) {
      //   rewriter.create<mt::SldwOp>(op.getLoc(), base, dst, nullptr);
      // } else {
      //   rewriter.create<mt::SldwOp>(op.getLoc(), base, dst, offsetValues.front());
      // }
    } else {
      // 创建VstwOp
      if (offsetValues.empty()) {
        rewriter.create<mt::VstwOp>(op.getLoc(), value, base, nullptr);
      } else {
        rewriter.create<mt::VstwOp>(op.getLoc(), value, base, offsetValues.front());
      }
    }

    rewriter.eraseOp(op);
    
    return success();
  }
};

class FtmLoadImmOpToMtLdwiOp : public FtmToMtOpConversion<ftm::LoadImmOp> {
public:
  using FtmToMtOpConversion<ftm::LoadImmOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::LoadImmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = adaptor.getAddr();
    int64_t immOffset = adaptor.getImmOffset();

    auto memLevelAttr = op->getAttr(ftm::MemLevelAttr::name);
    ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

    // 根据内存级别确定寄存器类型
    ftm::Cache regType = (memLevel == ftm::Cache::SM) ? 
                          ftm::Cache::ScalarRegister : 
                          ftm::Cache::VectorRegister;
    
    // 使用辅助方法创建寄存器并替换原操作
    Value dst = createRegisterAndReplace(op, op.getType(), regType, rewriter);

    // 根据内存级别创建不同的立即数操作
    if (regType == ftm::Cache::ScalarRegister) {
      // 创建SldwiOp - 使用立即数偏移
      rewriter.create<mt::SldwiOp>(op.getLoc(), base, dst, immOffset);
    } else {
      // 创建VldwiOp - 使用立即数偏移
      rewriter.create<mt::VldwiOp>(op.getLoc(), base, dst, immOffset);
    }
    
    return success();
  }
};

class FtmStoreImmOpToMtStwiOp : public FtmToMtOpConversion<ftm::StoreImmOp> {
public:
  using FtmToMtOpConversion<ftm::StoreImmOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::StoreImmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Value base = adaptor.getAddr();
    int64_t immOffset = adaptor.getImmOffset();

    auto memLevelAttr = op->getAttr(ftm::MemLevelAttr::name);
    ftm::Cache memLevel = memLevelAttr.cast<ftm::MemLevelAttr>().getLevel();

    // 根据内存级别确定寄存器类型
    ftm::Cache regType = (memLevel == ftm::Cache::SM) ? 
                          ftm::Cache::ScalarRegister : 
                          ftm::Cache::VectorRegister;
    
    // 根据内存级别创建不同的立即数操作
    if (regType == ftm::Cache::ScalarRegister) {
      // 创建SstwiOp - 目前没有实现
      // rewriter.create<mt::SstwiOp>(op.getLoc(), value, base, immOffset);
    } else {
      // 创建VstwiOp - 使用立即数偏移
      rewriter.create<mt::VstwiOp>(op.getLoc(), value, base, immOffset);
    }
    
    rewriter.eraseOp(op);
    
    return success();
  }
};

class FtmBroadcastOpToMtSvbcastOp : public FtmToMtOpConversion<ftm::BroadcastOp> {
public:
  using FtmToMtOpConversion<ftm::BroadcastOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value source = adaptor.getSource();

    Value dst = createRegisterAndReplace(op, op.getType(), ftm::Cache::VectorRegister, rewriter);

    rewriter.create<mt::SvbcastOp>(op.getLoc(), source, dst);

    return success();
  }
};

class FtmVbale2lOpToMtVbale2Op : public FtmToMtOpConversion<ftm::Vbale2lOp> {
public:
  using FtmToMtOpConversion<ftm::Vbale2lOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::Vbale2lOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value source = adaptor.getSource();

    Value dst = createRegisterAndReplace(op, op.getType(), ftm::Cache::VectorRegister, rewriter);

    rewriter.create<mt::Vbale2Op>(op.getLoc(), source, source, dst);

    return success();
  }
};

class FtmVbale2hOpToMtVbale2hOp : public FtmToMtOpConversion<ftm::Vbale2hOp> {
public:
  using FtmToMtOpConversion<ftm::Vbale2hOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::Vbale2hOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value source = adaptor.getSource();

    Value dst = createRegisterAndReplace(op, op.getType(), ftm::Cache::VectorRegister, rewriter);

    rewriter.create<mt::Vbale2hOp>(op.getLoc(), source, source, dst);

    return success();
  }
};

class FtmFMAOpToMtVfmulas32Op : public FtmToMtOpConversion<ftm::FMAOp> {
public:
  using FtmToMtOpConversion<ftm::FMAOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::FMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value acc = adaptor.getAcc();

    auto accOp = acc.getDefiningOp();
    rewriter.replaceOp(op, accOp);

    auto vfmulas32Op = rewriter.create<mt::Vfmulas32Op>(op.getLoc(), lhs, rhs, acc, acc);

    for (auto namedAttr : op->getAttrs()) {
      vfmulas32Op->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    return success();
  }
};

class FtmVFMAOpToMtVfmulas32Op : public FtmToMtOpConversion<ftm::VFMAOp> {
public:
  using FtmToMtOpConversion<ftm::VFMAOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::VFMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value acc = adaptor.getAcc();
    Value dst = adaptor.getDst();

    auto vfmulas32Op = rewriter.create<mt::Vfmulas32Op>(op.getLoc(), lhs, rhs, acc, dst);

    for (auto namedAttr : op->getAttrs()) {
      vfmulas32Op->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.eraseOp(op);

    return success();
  }
};

class FtmSmoviOpToMtSmoviOp : public FtmToMtOpConversion<ftm::SmoviOp> {
public:
  using FtmToMtOpConversion<ftm::SmoviOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::SmoviOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto imm = adaptor.getImmAttr();
    Value reg = adaptor.getReg();

    rewriter.create<mt::SmoviOp>(op.getLoc(), imm, reg);

    rewriter.eraseOp(op);

    return success();
  }
};

class FtmSmvagaOpToMtSmvagaOp : public FtmToMtOpConversion<ftm::SmvagaOp> {
public:
  using FtmToMtOpConversion<ftm::SmvagaOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::SmvagaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    rewriter.create<mt::SmvagaOp>(op.getLoc(), src, dst);

    rewriter.eraseOp(op);

    return success();
  }
};

class FtmVmoviOpToMtVmoviOp : public FtmToMtOpConversion<ftm::VmoviOp> {
public:
  using FtmToMtOpConversion<ftm::VmoviOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::VmoviOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto imm = adaptor.getImmAttr();
    Value reg = adaptor.getReg();

    rewriter.create<mt::VmoviOp>(op.getLoc(), imm, reg);

    rewriter.eraseOp(op);

    return success();
  }
};

class FtmDecRegOpToMtDecRegOp : public FtmToMtOpConversion<ftm::DeclareRegisterOp> {
public:
  using FtmToMtOpConversion<ftm::DeclareRegisterOp>::FtmToMtOpConversion;

  LogicalResult
  matchAndRewrite(ftm::DeclareRegisterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // 设置插入位置到函数入口
    auto funcOp = op->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());

    mt::DeclareRegisterOp decRegOp = rewriter.create<mt::DeclareRegisterOp>(op.getLoc(), op.getType());

    rewriter.replaceOp(op, decRegOp);

    // 设置内存级别属性
    if (auto memLevelAttr = op->getAttrOfType<ftm::MemLevelAttr>(ftm::MemLevelAttr::name)) {
      decRegOp->setAttr(ftm::MemLevelAttr::name, 
                        ftm::MemLevelAttr::get(rewriter.getContext(), 
                        static_cast<ftm::Cache>(memLevelAttr.getLevel())));
    }
    
    // 设置寄存器ID
    if (auto regIdAttr = op->getAttrOfType<ftm::RegisterIdAttr>(ftm::RegisterIdAttr::name)) {
      decRegOp->setAttr(ftm::RegisterIdAttr::name,
                       ftm::RegisterIdAttr::get(rewriter.getContext(), regIdAttr.getId()));
    }
    

    return success();
  }
};

class LowerFtmToMtPass : 
    public impl::LowerFtmToMtBase<LowerFtmToMtPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    
    // 设置转换目标
    ConversionTarget target(*context);
    
    // 将Mt方言设为合法
    target.addLegalDialect<mt::MtDialect>();

    // 设置非法操作
    target.addIllegalOp<LLVM::ConstantOp>();
    target.addIllegalOp<ftm::LoadOp>();
    target.addIllegalOp<ftm::StoreOp>();
    target.addIllegalOp<ftm::LoadImmOp>();
    target.addIllegalOp<ftm::StoreImmOp>();
    target.addIllegalOp<ftm::BroadcastOp>();
    target.addIllegalOp<ftm::Vbale2lOp>();
    target.addIllegalOp<ftm::Vbale2hOp>();
    target.addIllegalOp<ftm::VFMAOp>();
    target.addIllegalOp<ftm::FMAOp>();
    target.addIllegalOp<ftm::SmoviOp>();
    target.addIllegalOp<ftm::SmvagaOp>();
    target.addIllegalOp<ftm::VmoviOp>();
    target.addIllegalOp<ftm::DeclareRegisterOp>();
    
    // 设置类型转换器
    TypeConverter typeConverter;
    // 默认类型保持不变
    typeConverter.addConversion([](Type type) { return type; });
    
    // 注册转换模式
    RewritePatternSet patterns(context);
    
    // 添加所有转换模式
    patterns.add<
        LLVMConstantOpToMtSmoviOp,
        FtmLoadOpToMtLdwOp,
        FtmStoreOpToMtStwOp,
        FtmLoadImmOpToMtLdwiOp,
        FtmStoreImmOpToMtStwiOp,
        FtmBroadcastOpToMtSvbcastOp,
        FtmVbale2lOpToMtVbale2Op,
        FtmVbale2hOpToMtVbale2hOp,
        FtmFMAOpToMtVfmulas32Op,
        FtmVFMAOpToMtVfmulas32Op,
        FtmSmoviOpToMtSmoviOp,
        FtmSmvagaOpToMtSmvagaOp,
        FtmVmoviOpToMtVmoviOp,
        FtmDecRegOpToMtDecRegOp
    >(typeConverter, context);
    
    // 应用转换
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mt::createLowerFtmToMtPass() {
  return std::make_unique<LowerFtmToMtPass>();
}