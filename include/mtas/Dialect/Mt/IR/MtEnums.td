//===- MtEnums.td - Mt dialect functional unit enums ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in mt operations.
//
//===----------------------------------------------------------------------===//

#ifndef MT_IR_ENUMS
#define MT_IR_ENUMS

include "mlir/IR/EnumAttr.td"

// 功能单元类型的枚举定义
def SMAC : I64EnumAttrCase<"SMAC", 0, "SMAC">;        // 标量乘加单元
def SIEU : I64EnumAttrCase<"SIEU", 1, "SIEU">;        // 标量指令执行单元
def SLDST : I64EnumAttrCase<"SLDST", 2, "SLDST">;     // 标量加载/存储单元
def SBR : I64EnumAttrCase<"SBR", 3, "SBR">;           // 标量分支单元
def VMAC : I64EnumAttrCase<"VMAC", 4, "VMAC">;        // 向量乘加单元
def VIEU : I64EnumAttrCase<"VIEU", 5, "VIEU">;        // 向量指令执行单元
def VLDST : I64EnumAttrCase<"VLDST", 6, "VLDST">;     // 向量加载/存储单元

def Mt_FunctionalUnitEnum : I64EnumAttr<"FunctionalUnit", 
    "Functional units for instruction scheduling and packing", [
    SMAC, SIEU, SLDST, SBR, VMAC, VIEU, VLDST]> {
  let cppNamespace = "::mlir::mt";
}

// 寄存器类型的枚举定义
def SCALAR : I64EnumAttrCase<"Scalar", 0, "scalar">;       // 标量寄存器
def VECTOR : I64EnumAttrCase<"Vector", 1, "vector">;       // 向量寄存器
def ADDRESS : I64EnumAttrCase<"Address", 2, "address">;    // 地址寄存器
def OFFSET : I64EnumAttrCase<"Offset", 3, "offset">;       // 偏移寄存器

def Mt_RegisterTypeEnum : I64EnumAttr<"RegisterType", 
    "Register types in the Mt dialect", [
    SCALAR, VECTOR, ADDRESS, OFFSET]> {
  let cppNamespace = "::mlir::mt";
}

// // 添加寄存器号的定义，方便识别寄存器类型
// // 地址寄存器: 0-15
// // 向量寄存器: 16-47
// // 标量寄存器: 48-63
// def Mt_RegisterRangeAttr : AttrDef<Mt_Dialect, "RegisterRange"> {
//   let mnemonic = "register_range";
//   let parameters = (ins
//     "unsigned":$startId,
//     "unsigned":$endId
//   );
//   let assemblyFormat = "`<` $startId `:` $endId `>`";
// }

#endif // MT_IR_ENUMS