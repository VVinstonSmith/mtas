//===- MtOps.cpp - Implementation of Mt Dialect Ops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/IR/InstructionFormatter.h"

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
#include <sstream>   // 用于std::ostringstream
#include <iomanip>   // 用于std::setw和std::left

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
// Mt_DeclareRegisterOp
//===----------------------------------------------------------------------===//

std::string DeclareRegisterOp::formatRegisterOutput(){
  auto memLevel = this->getOperation()->getAttr(ftm::MemLevelAttr::name).cast<ftm::MemLevelAttr>().getLevel();
  auto regId = this->getOperation()->getAttr(ftm::RegisterIdAttr::name).cast<ftm::RegisterIdAttr>().getId();
  std::ostringstream oss;
  switch (memLevel){
    case ftm::Cache::AddressRegister:
      oss << "AR" << regId;
      break;
    case ftm::Cache::OffsetRegister:
      oss << "OR" << regId;
      break;
    case ftm::Cache::ScalarRegister:
      oss << "R" << regId;
      break;
    case ftm::Cache::VectorRegister:
      oss << "VR" << regId;
      break;
    default:
      oss << "<Unknown>";
      break;
  }
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SmvagaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SmvagaOp::getReadRegisters(){
  return {getSrc()};
}

Value SmvagaOp::getWrittenRegister(){
  return getDst();
}

int SmvagaOp::getLatency(){
  return 2;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SmvagaOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SMAC};
}

std::string SmvagaOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SMVAGA", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SmoviOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SmoviOp::getReadRegisters(){
  return {};
}

Value SmoviOp::getWrittenRegister(){
  return getReg();
}

int SmoviOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SmoviOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SMAC, mt::FunctionalUnit::SIEU};
}

int SmoviOp::getSchedulingPriority(){
  return 0;
}

std::string SmoviOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SMOVI", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string operand = InstructionFormatter::formatDecimalImmediate(getImm());
  operand.append(", ");
  operand.append(InstructionFormatter::formatRegisterName(getReg()));
  oss << std::left << std::setw(28) << operand;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_VmoviOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> VmoviOp::getReadRegisters(){
  return {};
}

Value VmoviOp::getWrittenRegister(){
  return getReg();
}

int VmoviOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> VmoviOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::VMAC, mt::FunctionalUnit::VIEU};
}

int VmoviOp::getSchedulingPriority(){
  return 0;
}

std::string VmoviOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VMOVI", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string operand = InstructionFormatter::formatHexImmediate(getImm());
  operand.append(", ");
  operand.append(InstructionFormatter::formatRegisterName(getReg()));
  oss << std::left << std::setw(28) << operand;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SldwOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SldwOp::getReadRegisters(){
  llvm::SmallVector<mlir::Value, 3> ReadRegs{getBase()};
  if(getOffset())
    ReadRegs.push_back(getOffset());
  return ReadRegs;
}

Value SldwOp::getWrittenRegister(){
  return getDst();
}

int SldwOp::getLatency(){
  return 7;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SldwOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SLDST};
}

std::string SldwOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SLDW", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string operand = InstructionFormatter::formatMemoryOperand(getBase(), getOffset());
  operand.append(", ");
  operand.append(InstructionFormatter::formatRegisterName(getDst()));
  oss << std::left << std::setw(28) << operand;
  
  return oss.str();
}

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

llvm::SmallVector<mlir::Value, 3> VldwOp::getReadRegisters(){
  llvm::SmallVector<mlir::Value, 3> ReadRegs{getBase()};
  if(getOffset())
    ReadRegs.push_back(getOffset());
  return ReadRegs;
}

Value VldwOp::getWrittenRegister(){
  return getDst();
}

int VldwOp::getLatency(){
  return 9;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> VldwOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::VLDST};
}

std::string VldwOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VLDW", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string operand = InstructionFormatter::formatMemoryOperand(getBase(), getOffset());
  operand.append(", ");
  operand.append(InstructionFormatter::formatRegisterName(getDst()));
  oss << std::left << std::setw(28) << operand;
  
  return oss.str();
}

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

llvm::SmallVector<mlir::Value, 3> VstwOp::getReadRegisters(){
  llvm::SmallVector<mlir::Value, 3> ReadRegs{getSrc(), getBase()};
  if(getOffset())
    ReadRegs.push_back(getOffset());
  return ReadRegs;
}

Value VstwOp::getWrittenRegister(){
  return nullptr;
}

int VstwOp::getLatency(){
  return 4;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> VstwOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::VLDST};
}

std::string VstwOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VSTW", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string operand = InstructionFormatter::formatRegisterName(getSrc());
  operand.append(", ");
  operand.append(InstructionFormatter::formatMemoryOperand(getBase(), getOffset()));
  oss << std::left << std::setw(28) << operand;
  
  return oss.str();
}

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
// Mt_SvbcastOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SvbcastOp::getReadRegisters(){
  return {getSrc()};
}

Value SvbcastOp::getWrittenRegister(){
  return getDst();
}

int SvbcastOp::getLatency(){
  return 4;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SvbcastOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SMAC};
}

std::string SvbcastOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SVBCAST", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_Vbale2Op
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> Vbale2Op::getReadRegisters(){
  return {getSrc1(), getSrc2()};
}

Value Vbale2Op::getWrittenRegister(){
  return getDst();
}

int Vbale2Op::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> Vbale2Op::getFunctionalUnits() {
  return {mt::FunctionalUnit::VIEU};
}

std::string Vbale2Op::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VBALE2", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc1(), getSrc2(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_Vbale2hOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> Vbale2hOp::getReadRegisters(){
  return {getSrc1(), getSrc2()};
}

Value Vbale2hOp::getWrittenRegister(){
  return getDst();
}

int Vbale2hOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> Vbale2hOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::VIEU};
}

std::string Vbale2hOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VBALE2h", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc1(), getSrc2(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_Vfmulas32Op
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> Vfmulas32Op::getReadRegisters(){
  return {getSrc1(), getSrc2(), getSrc3()};
}

Value Vfmulas32Op::getWrittenRegister(){
  return getDst();
}

int Vfmulas32Op::getLatency(){
  return 6;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> Vfmulas32Op::getFunctionalUnits() {
  return {mt::FunctionalUnit::VMAC};
}

std::string Vfmulas32Op::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VFMULAS32", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc1(), getSrc2(), getSrc3(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SmovOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SmovOp::getReadRegisters(){
  return {getSrc()};
}

Value SmovOp::getWrittenRegister(){
  return getDst();
}

int SmovOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SmovOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SMAC, mt::FunctionalUnit::SIEU};
}

int SmovOp::getSchedulingPriority(){
  return 0;
}

std::string SmovOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SMOV", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_VmovOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> VmovOp::getReadRegisters(){
  return {getSrc()};
}

Value VmovOp::getWrittenRegister(){
  return getDst();
}

int VmovOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> VmovOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::VMAC, mt::FunctionalUnit::VIEU};
}

int VmovOp::getSchedulingPriority(){
  return 0;
}

std::string VmovOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("VMOV", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getSrc(), getDst()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_AddaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> AddaOp::getReadRegisters(){
  return {getLhs(), getRhs()};
}

Value AddaOp::getWrittenRegister(){
  return getRes();
}

int AddaOp::getLatency() {
  // 获取操作数类型
  Value lhs = getLhs();
  
  // 检查寄存器 ID 属性
  if (auto declareRegisterOp = lhs.getDefiningOp<DeclareRegisterOp>()) {
    // 如果是 DeclareRegisterOp 定义的寄存器，检查其 register_id
    if (auto regIdAttr = declareRegisterOp->getAttr(
            ftm::RegisterIdAttr::name).cast<ftm::RegisterIdAttr>()) {
      // 获取寄存器 ID 值
      uint32_t regId = regIdAttr.getId();
      
      if (regId < 8) {
        return 3;
      }
    }
  }

  return 2;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> AddaOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SIEU, mt::FunctionalUnit::SMAC};
}

int AddaOp::getSchedulingPriority(){
  return 0;
}

std::string AddaOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("ADDA", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getLhs(), getRhs(), getRes()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SaddOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SaddOp::getReadRegisters(){
  return {getLhs(), getRhs()};
}

Value SaddOp::getWrittenRegister(){
  return getRes();
}

int SaddOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SaddOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SMAC, mt::FunctionalUnit::SIEU};
}

int SaddOp::getSchedulingPriority(){
  return 0;
}

std::string SaddOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SADD", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getLhs(), getRhs(), getRes()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Mt_SltOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Value, 3> SltOp::getReadRegisters(){
  return {getLhs(), getRhs()};
}

Value SltOp::getWrittenRegister(){
  return getRes();
}

int SltOp::getLatency(){
  return 1;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SltOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SIEU};
}

std::string SltOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  std::string opName = InstructionFormatter::formatInstructionName("SLT", functionalUnit);
  // 使用setw设置字段宽度为18，并使用left进行左对齐
  oss << std::left << std::setw(18) << opName;

  std::string regName = InstructionFormatter::formatOperandList({getLhs(), getRhs(), getRes()});
  oss << std::left << std::setw(28) << regName;
  
  return oss.str();
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

llvm::SmallVector<mlir::Value, 3> SbrLabelOp::getReadRegisters(){
  if(getCond())
    return {getCond()};
  return {};
}

Value SbrLabelOp::getWrittenRegister(){
  return nullptr;
}

int SbrLabelOp::getLatency(){
  return 7;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SbrLabelOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SBR};
}

std::string SbrLabelOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  oss << std::left << std::setw(18) << InstructionFormatter::formatBranchWithCondition("SBR", getCond());
  oss << std::left << std::setw(28) << getLabel().str();
  return oss.str();
}

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

llvm::SmallVector<mlir::Value, 3> SbrRegOp::getReadRegisters(){
  if(getCond())
    return {getCond()};
  return {};
}

Value SbrRegOp::getWrittenRegister(){
  return nullptr;
}

int SbrRegOp::getLatency(){
  return 7;
}

llvm::SmallVector<mlir::mt::FunctionalUnit, 2> SbrRegOp::getFunctionalUnits() {
  return {mt::FunctionalUnit::SBR};
}

std::string SbrRegOp::formatOperationOutput(StringRef functionalUnit){
  std::ostringstream oss;
  oss << std::left << std::setw(18) << InstructionFormatter::formatBranchWithCondition("SBR", getCond());
  oss << std::left << std::setw(28) << InstructionFormatter::formatRegisterName(getTarget());
  return oss.str();
}

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