//===- MtasmAttrs.td - Mtasm dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enumeration and attributes used in mt3000
// asm operations.
//
//===----------------------------------------------------------------------===//

#ifndef MTASM_IR_MTASMATTRS
#define MTASM_IR_MTASMATTRS

include "mtas/Dialect/Mtasm/IR/MtasmBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Mtasm Attributes.
//===----------------------------------------------------------------------===//

class Mtasm_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Mtasm_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Mtasm_I32EnumAttr<string mnemonic, Mtasm_I32Enum enumInfo>
    : EnumAttr<Mtasm_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Mtasm_MemLevelAttr : Mtasm_Attr<"MemLevel", "memory_level"> {
  let parameters = (ins
    EnumParameter<Mtasm_CacheEnum>:$level
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which cache to allocate a dynamic buffer.
  }];
}

def Mtasm_FuncArgMatAttr : Mtasm_Attr<"FuncArgMat", "func_arg_mat"> {
  let parameters = (ins
    EnumParameter<Mtasm_MatrixEnum>:$matrix
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which matrix in matmul.
  }];
}

def Mtasm_FuncArgDimAttr : Mtasm_Attr<"FuncArgDim", "func_arg_dim"> {
  let parameters = (ins
    EnumParameter<Mtasm_AxisEnum>:$dim
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which dimension in matmul.
  }];
}

def Mtasm_MultiStageAttr : Mtasm_Attr<"MultiStage", "multi_stage"> {
  let parameters = (ins
    EnumParameter<Mtasm_StageEnum>:$stage
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which stage during pipelining.
  }];
}

def Mtasm_UnrollFactor : Mtasm_Attr<"UnrollFactor", "unroll_factor"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$factor
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Mtasm loop unrolling factor. 
  }];
}

def Mtasm_UnrollSegment : Mtasm_Attr<"UnrollSegment", "unroll_segment"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$segmentId
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Mtasm loop unrolling segment id. 
  }];
}

#endif // MTASM_IR_MTASMATTRS