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

//===----------------------------------------------------------------------===//
// Mt_LabelOp
//===----------------------------------------------------------------------===//

ParseResult LabelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  
  // 解析标签名称属性
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();
  
  // 不添加类型到结果状态，因为该操作没有返回值
  result.addTypes({});
  
  return success();
}

void LabelOp::print(OpAsmPrinter &p) {
  // 打印操作名和标签名称
  p << " " << getName();
}

//===----------------------------------------------------------------------===//
// Mt_SbrLabelOp
//===----------------------------------------------------------------------===//

ParseResult SbrLabelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr labelAttr;
  OpAsmParser::UnresolvedOperand condInfo;
  Type condType;
  bool hasCondition = false;
  
  // 尝试解析条件操作数
  auto optRes = parser.parseOptionalOperand(condInfo);
  if (optRes.has_value() && succeeded(*optRes)) {
    hasCondition = true;
    
    // 解析逗号
    if (parser.parseComma())
      return failure();
    
    // 解析标签属性
    if (parser.parseAttribute(labelAttr, "label", result.attributes))
      return failure();
    
    // 解析冒号和条件类型
    if (parser.parseColon() ||
        parser.parseType(condType))
      return failure();
    
    // 验证条件类型是否为I64
    if (!condType.isInteger(64))
      return parser.emitError(parser.getNameLoc(), "expected i64 for condition");
    
    // 添加条件操作数到结果状态
    if (parser.resolveOperand(condInfo, condType, result.operands))
      return failure();
  } else {
    // 没有条件，直接解析标签属性
    if (parser.parseAttribute(labelAttr, "label", result.attributes))
      return failure();
  }
  
  // 不添加类型到结果状态，因为该操作没有返回值
  result.addTypes({});
  
  return success();
}

void SbrLabelOp::print(OpAsmPrinter &p) {
  // 如果有条件，先打印条件
  if (hasCondition()) {
    p << " " << getCond() << ", ";
    p << getLabel() << " : " << getCond().getType();
  } else {
    // 没有条件，只打印标签
    p << " " << getLabel();
  }
}

//===----------------------------------------------------------------------===//
// Mt_SbrRegOp
//===----------------------------------------------------------------------===//

ParseResult SbrRegOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand targetInfo;
  OpAsmParser::UnresolvedOperand condInfo;
  Type targetType;
  Type condType;
  bool hasCondition = false;
  
  // 解析目标寄存器操作数
  if (parser.parseOperand(targetInfo))
    return failure();
  
  // 检查是否有条件操作数
  if (succeeded(parser.parseOptionalComma())) {
    hasCondition = true;
    
    // 解析条件操作数
    if (parser.parseOperand(condInfo))
      return failure();
  }
  
  // 解析冒号和类型信息
  if (parser.parseColon() || 
      parser.parseType(targetType))
    return failure();
  
  // 验证目标寄存器类型是否为I64
  if (!targetType.isInteger(64))
    return parser.emitError(parser.getNameLoc(), "expected i64 for target register");
  
  // 添加目标寄存器操作数到结果状态
  if (parser.resolveOperand(targetInfo, targetType, result.operands))
    return failure();
  
  // 如果有条件，解析条件类型并添加条件操作数
  if (hasCondition) {
    if (parser.parseComma() || 
        parser.parseType(condType))
      return failure();
    
    // 验证条件类型是否为I64
    if (!condType.isInteger(64))
      return parser.emitError(parser.getNameLoc(), "expected i64 for condition");
    
    // 添加条件操作数到结果状态
    if (parser.resolveOperand(condInfo, condType, result.operands))
      return failure();
  }
  
  // 不添加类型到结果状态，因为该操作没有返回值
  result.addTypes({});
  
  return success();
}

void SbrRegOp::print(OpAsmPrinter &p) {
  // 打印操作名和目标寄存器
  p << " " << getTarget();
  
  // 如果有条件，则打印条件
  if (hasCondition())
    p << ", " << getCond();
  
  // 打印类型信息
  p << " : " << getTarget().getType();
  
  // 如果有条件，则打印条件类型
  if (hasCondition())
    p << ", " << getCond().getType();
}