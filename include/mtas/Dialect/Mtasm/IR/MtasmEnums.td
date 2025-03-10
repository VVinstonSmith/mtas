//===- MtasmEnums.td - Mtasm dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in mtasm operations.
//
//===----------------------------------------------------------------------===//

#ifndef MTASM_IR_ENUMS
#define MTASM_IR_ENUMS

include "mlir/IR/EnumAttr.td"

class Mtasm_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtasm";
}

def DimX : I64EnumAttrCase<"DimX", 0, "x">;
def DimY : I64EnumAttrCase<"DimY", 1, "y">;
def DimZ : I64EnumAttrCase<"DimZ", 2, "z">;
def LinearDim0 : I64EnumAttrCase<"LinearDim0", 3, "linear_dim_0">;
def LinearDim1 : I64EnumAttrCase<"LinearDim1", 4, "linear_dim_1">;
def LinearDim2 : I64EnumAttrCase<"LinearDim2", 5, "linear_dim_2">;

def MappingIdEnum : I64EnumAttr<"MappingId", "Mapping ids for loop mapping", [
    DimX, DimY, DimZ, LinearDim0, LinearDim1, LinearDim2]> {
  let cppNamespace = "::mlir::mtasm";
}

def AxisM : I64EnumAttrCase<"M", 0, "dim_M">;
def AxisN : I64EnumAttrCase<"N", 1, "dim_N">;
def AxisK : I64EnumAttrCase<"K", 2, "dim_K">;

def Mtasm_AxisEnum : I64EnumAttr<"Axis", "axis", [
    AxisM, AxisN, AxisK]> {
  let cppNamespace = "::mlir::mtasm";
}

def MatA : I64EnumAttrCase<"MatA", 0, "mat_A">;
def MatB : I64EnumAttrCase<"MatB", 1, "mat_B">;
def MatC : I64EnumAttrCase<"MatC", 2, "mat_C">;
def MatE : I64EnumAttrCase<"MatE", 3, "mat_E">;

def Mtasm_MatrixEnum : I64EnumAttr<"Matrix", "matrix", [
    MatA, MatB, MatC, MatE]> {
  let cppNamespace = "::mlir::mtasm";
}

def CacheDDR : I64EnumAttrCase<"DDR", 0, "ddr">;
def CacheGSM : I64EnumAttrCase<"GSM", 1, "gsm">;
def CacheAM : I64EnumAttrCase<"AM", 2, "am">;
def CacheSM : I64EnumAttrCase<"SM", 3, "sm">;

def Mtasm_CacheEnum : I64EnumAttr<"Cache", "cache", [
    CacheDDR, CacheGSM, CacheAM, CacheSM]> {
  let cppNamespace = "::mlir::mtasm";
}

// Enums for pipelining.
def Prelogue : I64EnumAttrCase<"Prelogue", 0, "prelogue">;
def MainBody : I64EnumAttrCase<"MainBody", 1, "main_body">;
def Epilogue : I64EnumAttrCase<"Epilogue", 2, "epilogue">;
def Prefetch : I64EnumAttrCase<"Prefetch", 3, "prefetch">;
def PostStore : I64EnumAttrCase<"PostStore", 4, "post_store">;

def Mtasm_StageEnum : I64EnumAttr<"Stage", "stage", [
    Prelogue, MainBody, Epilogue, Prefetch, PostStore]> {
  let cppNamespace = "::mlir::mtasm";
}

#endif // MTASM_IR_ENUMS