//===-------- MtOps.td - Mt op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MT_IR_MTOPS
#define MT_IR_MTOPS

include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mtas/Dialect/Mt/IR/MtBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for Mt dialect ops that do not correspond to library calls.
class Mt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mt_Dialect, mnemonic, traits>;

def Mt_SmvagaOp :
  Op<Mt_Dialect, "SMVAGA", [AllTypesMatch<["src", "dst"]>]>,
  Arguments<(ins AnyTypeOf<[I64, LLVM_AnyPointer]>:$src,
                 AnyTypeOf<[I64, LLVM_AnyPointer]>:$dst)>{
  let summary = "scalar move to address/offset register operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
}

def Mt_SmoviOp :
  Op<Mt_Dialect, "SMOVI">,
  Arguments<(ins I64Attr:$imm, I64:$reg)>{
  let summary = "scalar move immediate operation";
  let assemblyFormat = "$imm `,` $reg attr-dict `:` type($reg)";
}

def Mt_SmovOp :
  Op<Mt_Dialect, "SMOV", [AllTypesMatch<["src", "dst"]>]>,
  Arguments<(ins I64:$src, 
                 I64:$dst)>{
  let summary = "scalar move operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
}

def Mt_VmovOp :
  Op<Mt_Dialect, "VMOV", [AllTypesMatch<["src", "dst"]>]>,
  Arguments<(ins VectorOfAnyRankOf<[AnyFloat]>:$src,
                 VectorOfAnyRankOf<[AnyFloat]>:$dst)>{
  let summary = "vector move operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
}

def Mt_AddaOp :
  Op<Mt_Dialect, "ADDA", [AllTypesMatch<["lhs", "res"]>]>,
  Arguments<(ins LLVM_AnyPointer:$lhs,
                 I64:$rhs,
                 LLVM_AnyPointer:$res)>{
  let summary = "address addition operation";
  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($res)";
  let extraClassDeclaration = [{
    LLVM::LLVMPointerType getPointerType() { return ::llvm::cast<LLVM::LLVMPointerType>(getLhs().getType()); }
  }];
}

#endif // MT_IR_MTOPS