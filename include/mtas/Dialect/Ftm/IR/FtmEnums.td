//===- FtmEnums.td - Ftm dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in ftm operations.
//
//===----------------------------------------------------------------------===//

#ifndef FTM_IR_ENUMS
#define FTM_IR_ENUMS

include "mlir/IR/EnumAttr.td"

class Ftm_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::ftm";
}

def DimX : I64EnumAttrCase<"DimX", 0, "x">;
def DimY : I64EnumAttrCase<"DimY", 1, "y">;
def DimZ : I64EnumAttrCase<"DimZ", 2, "z">;
def LinearDim0 : I64EnumAttrCase<"LinearDim0", 3, "linear_dim_0">;
def LinearDim1 : I64EnumAttrCase<"LinearDim1", 4, "linear_dim_1">;
def LinearDim2 : I64EnumAttrCase<"LinearDim2", 5, "linear_dim_2">;

def MappingIdEnum : I64EnumAttr<"MappingId", "Mapping ids for loop mapping", [
    DimX, DimY, DimZ, LinearDim0, LinearDim1, LinearDim2]> {
  let cppNamespace = "::mlir::ftm";
}

def AxisM : I64EnumAttrCase<"M", 0, "dim_M">;
def AxisN : I64EnumAttrCase<"N", 1, "dim_N">;
def AxisK : I64EnumAttrCase<"K", 2, "dim_K">;

def Ftm_AxisEnum : I64EnumAttr<"Axis", "axis", [
    AxisM, AxisN, AxisK]> {
  let cppNamespace = "::mlir::ftm";
}

def MatX : I64EnumAttrCase<"MatX", 0, "mat_X">;
def MatA : I64EnumAttrCase<"MatA", 1, "mat_A">;
def MatB : I64EnumAttrCase<"MatB", 2, "mat_B">;
def MatC : I64EnumAttrCase<"MatC", 3, "mat_C">;
def MatD : I64EnumAttrCase<"MatD", 4, "mat_D">;
def MatE : I64EnumAttrCase<"MatE", 5, "mat_E">;

def Ftm_MatrixEnum : I64EnumAttr<"Matrix", "matrix", [
    MatX, MatA, MatB, MatC, MatD, MatE]> {
  let cppNamespace = "::mlir::ftm";
}

def Unknown : I64EnumAttrCase<"Unknown", 0, "unknown">;
def CacheDDR : I64EnumAttrCase<"DDR", 1, "ddr">;
def CacheGSM : I64EnumAttrCase<"GSM", 2, "gsm">;
def CacheAM : I64EnumAttrCase<"AM", 3, "am">;
def CacheSM : I64EnumAttrCase<"SM", 4, "sm">;
def ScalarRegister : I64EnumAttrCase<"ScalarRegister", 5, "scalar_register">;
def VectorRegister : I64EnumAttrCase<"VectorRegister", 6, "vector_register">;
def AddressRegister : I64EnumAttrCase<"AddressRegister", 7, "address_register">;
def OffsetRegister : I64EnumAttrCase<"OffsetRegister", 8, "offset_register">;

def Ftm_CacheEnum : I64EnumAttr<"Cache", "cache", [
    Unknown, CacheDDR, CacheGSM, CacheAM, CacheSM,
    ScalarRegister, VectorRegister, AddressRegister, OffsetRegister]> {
  let cppNamespace = "::mlir::ftm";
}

// Enums for pipelining.
def Prelogue : I64EnumAttrCase<"Prelogue", 0, "prelogue">;
def MainBody : I64EnumAttrCase<"MainBody", 1, "main_body">;
def Epilogue : I64EnumAttrCase<"Epilogue", 2, "epilogue">;
def Prefetch : I64EnumAttrCase<"Prefetch", 3, "prefetch">;
def PostStore : I64EnumAttrCase<"PostStore", 4, "post_store">;

def Ftm_StageEnum : I64EnumAttr<"Stage", "stage", [
    Prelogue, MainBody, Epilogue, Prefetch, PostStore]> {
  let cppNamespace = "::mlir::ftm";
}

#endif // FTM_IR_ENUMS