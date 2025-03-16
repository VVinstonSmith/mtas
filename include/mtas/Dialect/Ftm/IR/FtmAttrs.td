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

def Ftm_MatrixNameAttr : Ftm_Attr<"MatrixName", "matrix_name"> {
  let parameters = (ins
    EnumParameter<Ftm_MatrixEnum>:$matrix
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    matrix name.s
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

def Ftm_UnrollLoopNumber : Ftm_Attr<"UnrollLoopNumber", "unroll_loop_number"> {
  let parameters = (ins
    OptionalParameter<"unsigned">:$number
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    How many loop layers to unroll?
  }];
}


def Ftm_OperandMatrixNameAttr : Ftm_Attr<"OperandMatrixName", "operand_matrix_name"> {
  let parameters = (ins 
    OptionalArrayRefParameter<"mlir::ftm::Matrix">:$matrixNames
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    matrix name of each operand in a operation.
  }];
}

def Ftm_OperandMemLevelAttr : Ftm_Attr<"OperandMemLevel", "operand_memory_level"> {
  let parameters = (ins 
    OptionalArrayRefParameter<"mlir::ftm::Cache">:$memLevels
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    memory levels of each operand in a operation.
  }];
}

def Ftm_LoopIdAttr : Ftm_Attr<"LoopId", "loop_id"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$loopId
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    loop id number
  }];
}

def Ftm_RegisterIdAttr : Ftm_Attr<"RegisterId", "register_id"> {
  let parameters = (ins 
    OptionalParameter<"unsigned">:$id
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    register id
  }];
}

#endif // FTM_IR_FTMATTRS