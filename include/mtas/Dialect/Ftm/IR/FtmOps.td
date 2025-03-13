//===-------- FtmOps.td - Ftm op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FTM_IR_FTMOPS
#define FTM_IR_FTMOPS

include "mtas/Dialect/Ftm/IR/FtmBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for Ftm dialect ops that do not correspond to library calls.
class Ftm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ftm_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AnnotateOp
//===----------------------------------------------------------------------===//

def Ftm_AnnotateOp : Ftm_Op<"annotate"> {
  let summary = "annotate operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Ftm_BroadcastOp :
  Ftm_Op<"broadcast", [Pure]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyVectorOfAnyRank:$destiny)> {
  let summary = "broadcast operation";
  
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getDestiny().getType());
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($destiny)";
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def Ftm_Vbale2lOp:
  Ftm_Op<"vbale2l", [Pure]>,
    Arguments<(ins AnyVectorOfAnyRank:$source)>,
    Results<(outs AnyVectorOfAnyRank:$destiny)> {
  let summary = "VBALE2 operation";
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getDestiny().getType());
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($destiny)";
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def Ftm_Vbale2hOp:
  Ftm_Op<"vbale2h", [Pure]>,
    Arguments<(ins AnyVectorOfAnyRank:$source)>,
    Results<(outs AnyVectorOfAnyRank:$destiny)> {
  let summary = "VBALE2H operation";
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getDestiny().getType());
    }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($destiny)";
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

// def Ftm_Load64 : Ftm_Op<"load64"> {
//   let summary = "annotate operation";
//   let arguments = (ins Variadic<AnyType>:$operands);
//   let builders = [
//     OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
//   ];
//   let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
// }


#endif // FTM_IR_FTMOPS