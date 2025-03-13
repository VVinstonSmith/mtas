//===- FtmAttrs.td - Ftm dialect base support ------*- tablegen -*-===//
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

#ifndef FTM_IR_FTMATTRS
#define FTM_IR_FTMATTRS

include "mtas/Dialect/Ftm/IR/FtmBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Ftm Attributes.
//===----------------------------------------------------------------------===//

class Ftm_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Ftm_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Ftm_I32EnumAttr<string mnemonic, Ftm_I32Enum enumInfo>
    : EnumAttr<Ftm_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Ftm_MemLevelAttr : Ftm_Attr<"MemLevel", "memory_level"> {
  let parameters = (ins
    EnumParameter<Ftm_CacheEnum>:$level
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which cache to allocate a dynamic buffer.
  }];
}

def Ftm_FuncArgMatAttr : Ftm_Attr<"FuncArgMat", "func_arg_mat"> {
  let parameters = (ins
    EnumParameter<Ftm_MatrixEnum>:$matrix
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which matrix in matmul.
  }];
}

def Ftm_FuncArgDimAttr : Ftm_Attr<"FuncArgDim", "func_arg_dim"> {
  let parameters = (ins
    EnumParameter<Ftm_AxisEnum>:$dim
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which dimension in matmul.
  }];
}

def Ftm_MultiStageAttr : Ftm_Attr<"MultiStage", "multi_stage"> {
  let parameters = (ins
    EnumParameter<Ftm_StageEnum>:$stage
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which stage during pipelining.
  }];
}

def Ftm_UnrollFactorAttr : Ftm_Attr<"UnrollFactor", "unroll_factor"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$factor
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Ftm loop unrolling factor. 
  }];
}

def Ftm_UnrollSegmentAttr : Ftm_Attr<"UnrollSegment", "unroll_segment"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$segmentId
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Ftm loop unrolling segment id. 
  }];
}

def Ftm_OperandMemLevelAttr : Ftm_Attr<"OperandMemLevel", "matmul_memory_level"> {
  let parameters = (ins 
    OptionalArrayRefParameter<"mlir::ftm::Cache">:$memLevels
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    memory levels of each operand in a operation.
  }];
}

#endif // FTM_IR_FTMATTRS