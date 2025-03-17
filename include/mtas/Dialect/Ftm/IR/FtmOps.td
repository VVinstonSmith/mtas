//===-------- FtmOps.td - Ftm op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FTM_IR_FTMOPS
#define FTM_IR_FTMOPS

include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
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


def Ftm_MemRefLoadOp : Ftm_Op<"memref_load"> {
  let summary = "reads an n-D slice of memory into an n-D vector";
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$base);
  let results = (outs AnyVectorOfAnyRank:$result);
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat =
      "$base attr-dict `:` type($base) `,` type($result)";
}

def Ftm_MemRefStoreOp : Ftm_Op<"memref_store"> {
  let summary = "writes an n-D vector to an n-D slice of memory";
  let arguments = (ins
      AnyVectorOfAnyRank:$valueToStore,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$base
  );
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
  }];
  let assemblyFormat = "$valueToStore `,` $base attr-dict "
                       "`:` type($base) `,` type($valueToStore)";
}

def Ftm_FMAOp :
  Op<Ftm_Dialect, "fma", [Pure, AllTypesMatch<["lhs", "rhs", "acc", "result"]>]>,
    Arguments<(ins VectorOfAnyRankOf<[AnyFloat]>:$lhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$rhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$acc)>,
    Results<(outs VectorOfAnyRankOf<[AnyFloat]>:$result)> {
  let summary = "vector fused multiply-add";
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)";
  let extraClassDeclaration = [{
    VectorType getVectorType() { return ::llvm::cast<VectorType>(getLhs().getType()); }
  }];
}

def Ftm_VFMAOp :
  Op<Ftm_Dialect, "vfma", [AllTypesMatch<["lhs", "rhs", "acc", "dst"]>]>,
    Arguments<(ins VectorOfAnyRankOf<[AnyFloat]>:$lhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$rhs,
                   VectorOfAnyRankOf<[AnyFloat]>:$acc,
                   VectorOfAnyRankOf<[AnyFloat]>:$dst)> {
  let summary = "vector fused multiply-add";
  let assemblyFormat = "$lhs `,` $rhs `,` $acc `,` $dst attr-dict `:` type($lhs)";
  let extraClassDeclaration = [{
    VectorType getVectorType() { return ::llvm::cast<VectorType>(getLhs().getType()); }
  }];
}

def Ftm_LoadOp : Ftm_Op<"load", [Pure]> {
  let arguments = (ins LLVM_AnyPointer:$addr, Variadic<I64>:$offsetRegister);
  let results = (outs LLVM_LoadableType:$res);
  let assemblyFormat = [{
    $addr `,` $offsetRegister attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
  let extraClassDeclaration = [{
    void setOffsetRegister(Value reg) {
      setOperand(1, reg);
    }
  }];
}

def Ftm_StoreOp : Ftm_Op<"store"> {
  let arguments = (ins LLVM_LoadableType:$value,
                       LLVM_AnyPointer:$addr);
  let assemblyFormat = [{
    $value `,` $addr attr-dict `:` type($value) `,` qualified(type($addr))
  }];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), getAddr(),
                           SideEffects::DefaultResource::get());
    }
  }];
}

def Ftm_CastOp:
  Ftm_Op<"cast", [Pure]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnyType:$destiny)> {
  let summary = "Cast operation";
  let extraClassDeclaration = [{
    Type getSourceType() { return getSource().getType(); }
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($destiny)";
}

def Ftm_DeclareRegisterOp: Ftm_Op<"declare_register">,
    Results<(outs AnyType:$output)> {
  let summary = "declare register operation";
  let assemblyFormat = "attr-dict `:` type($output)";
}

def Ftm_SmoviOp :
  Op<Ftm_Dialect, "smovi">,
    Arguments<(ins I64Attr:$imm, I64:$reg)>,
    Results<(outs I64:$result)> {
  let summary = "scalar move immediate number";
  let assemblyFormat = "$imm `,` $reg attr-dict `:` type($result)";
}



#endif // FTM_IR_FTMOPS