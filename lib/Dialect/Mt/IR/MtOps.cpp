//===- MtOps.cpp - Implementation of Mt Dialect Ops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::mt;

#define GET_OP_CLASSES
#include "mtas/Dialect/Mt/IR/MtOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MtDialect
//===----------------------------------------------------------------------===//
void MtDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {

}

//===----------------------------------------------------------------------===//
// Mt_SldwOp
//===----------------------------------------------------------------------===//

ParseResult SldwOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand baseInfo;
  OpAsmParser::UnresolvedOperand offsetInfo;
  OpAsmParser::UnresolvedOperand dstInfo;
  Type baseType;
  Type resultType;
  
  // 解析基址操作数
  if (parser.parseOperand(baseInfo))
    return failure();

  // 检查是否有偏移量
  bool hasOffset = false;
  if (succeeded(parser.parseOptionalLSquare())) {
    hasOffset = true;
    if (parser.parseOperand(offsetInfo) ||
        parser.parseRSquare())
      return failure();
  }
  
  // 解析逗号和目标向量操作数
  if (parser.parseComma() || 
      parser.parseOperand(dstInfo))
    return failure();
  
  // 解析冒号和类型信息
  if (parser.parseColon() || 
      parser.parseType(baseType) || 
      parser.parseComma() || 
      parser.parseType(resultType))
    return failure();
  
  // 验证目标向量类型
  auto vectorType = resultType.dyn_cast<VectorType>();
  if (!vectorType || vectorType.getShape().size() != 1 || 
      vectorType.getShape()[0] != 2 || 
      !vectorType.getElementType().isF32())
    return parser.emitError(parser.getNameLoc(), "expected 2-element vector of f32");
  
  // 添加基址操作数到结果状态
  if (parser.resolveOperand(baseInfo, baseType, result.operands))
    return failure();
  
  // 如果有偏移量，添加偏移量操作数
  if (hasOffset) {
    if (parser.resolveOperand(offsetInfo, IntegerType::get(parser.getContext(), 64), result.operands))
      return failure();
  }
  
  // 添加目标向量操作数
  if (parser.resolveOperand(dstInfo, resultType, result.operands))
    return failure();
  
  // 添加类型到结果状态
  result.addTypes({});
  
  return success();
}

void SldwOp::print(OpAsmPrinter &p) {
  // 打印操作名和基址
  p << ' ' << getBase();
  
  // 如果有偏移量，则打印偏移量
  if (getOffset())
    p << "[" << getOffset() << "]";
  
  // 打印逗号和目标向量以及类型信息
  p << ", " << getDst() << " : " << getBase().getType() << ", " << getDst().getType();
}

//===----------------------------------------------------------------------===//
// Mt_VldwOp
//===----------------------------------------------------------------------===//

ParseResult VldwOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand baseInfo;
  OpAsmParser::UnresolvedOperand offsetInfo;
  OpAsmParser::UnresolvedOperand dstInfo;
  Type baseType;
  Type resultType;
  
  // 解析基址操作数
  if (parser.parseOperand(baseInfo))
    return failure();

  // 检查是否有偏移量
  bool hasOffset = false;
  if (succeeded(parser.parseOptionalLSquare())) {
    hasOffset = true;
    if (parser.parseOperand(offsetInfo) ||
        parser.parseRSquare())
      return failure();
  }
  
  // 解析逗号和目标向量操作数
  if (parser.parseComma() || 
      parser.parseOperand(dstInfo))
    return failure();
  
  // 解析冒号和类型信息
  if (parser.parseColon() || 
      parser.parseType(baseType) || 
      parser.parseComma() || 
      parser.parseType(resultType))
    return failure();
  
  // 验证目标向量类型
  auto vectorType = resultType.dyn_cast<VectorType>();
  if (!vectorType || vectorType.getShape().size() != 1 || 
      vectorType.getShape()[0] != 2 || 
      !vectorType.getElementType().isF32())
    return parser.emitError(parser.getNameLoc(), "expected 32-element vector of f32");
  
  // 添加基址操作数到结果状态
  if (parser.resolveOperand(baseInfo, baseType, result.operands))
    return failure();
  
  // 如果有偏移量，添加偏移量操作数
  if (hasOffset) {
    if (parser.resolveOperand(offsetInfo, IntegerType::get(parser.getContext(), 64), result.operands))
      return failure();
  }
  
  // 添加目标向量操作数
  if (parser.resolveOperand(dstInfo, resultType, result.operands))
    return failure();
  
  // 添加类型到结果状态
  result.addTypes({});
  
  return success();
}

void VldwOp::print(OpAsmPrinter &p) {
  // 打印操作名和基址
  p << ' ' << getBase();
  
  // 直接在基址后打印偏移量
  if (getOffset()) {
    p << "[" << getOffset() << "]";
  }
  
  // 打印逗号和目标向量以及类型信息
  p << ", " << getDst() << " : " << getBase().getType() << ", " << getDst().getType();
}

//===----------------------------------------------------------------------===//
// Mt_VstwOp
//===----------------------------------------------------------------------===//

ParseResult VstwOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand srcInfo;
  OpAsmParser::UnresolvedOperand baseInfo;
  OpAsmParser::UnresolvedOperand offsetInfo;
  Type srcType;
  Type baseType;
  
  // 解析源向量操作数
  if (parser.parseOperand(srcInfo))
    return failure();
    
  // 解析逗号和基址操作数
  if (parser.parseComma() || 
      parser.parseOperand(baseInfo))
    return failure();
  
  // 检查是否有偏移量
  bool hasOffset = false;
  if (succeeded(parser.parseOptionalLSquare())) {
    hasOffset = true;
    if (parser.parseOperand(offsetInfo) ||
        parser.parseRSquare())
      return failure();
  }
  
  // 解析冒号和类型信息
  if (parser.parseColon() || 
      parser.parseType(srcType) || 
      parser.parseComma() || 
      parser.parseType(baseType))
    return failure();
  
  // 验证源向量类型
  auto vectorType = srcType.dyn_cast<VectorType>();
  if (!vectorType || vectorType.getShape().size() != 1 || 
      vectorType.getShape()[0] != 32 || 
      !vectorType.getElementType().isF32())
    return parser.emitError(parser.getNameLoc(), "expected 32-element vector of f32");
  
  // 添加源向量操作数到结果状态
  if (parser.resolveOperand(srcInfo, srcType, result.operands))
    return failure();
  
  // 添加基址操作数到结果状态
  if (parser.resolveOperand(baseInfo, baseType, result.operands))
    return failure();
  
  // 如果有偏移量，添加偏移量操作数
  if (hasOffset) {
    if (parser.resolveOperand(offsetInfo, IntegerType::get(parser.getContext(), 64), result.operands))
      return failure();
  }
  
  // 添加类型到结果状态
  result.addTypes({});
  
  return success();
}

void VstwOp::print(OpAsmPrinter &p) {
  // 打印操作名和源向量
  p << ' ' << getSrc();
  
  // 打印逗号和基址
  p << ", " << getBase();
  
  // 如果有偏移量，打印偏移量
  if (getOffset()) {
    p << "[" << getOffset() << "]";
  }
  
  // 打印类型信息
  p << " : " << getSrc().getType() << ", " << getBase().getType();
}