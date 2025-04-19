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

def Mt_DeclareRegisterOp: Mt_Op<"declare_register">,
    Results<(outs AnyType:$output)> {
  let summary = "declare register operation";
  let assemblyFormat = "attr-dict `:` type($output)";
}

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

def Mt_VmoviOp :
  Op<Mt_Dialect, "VMOVI">,
  Arguments<(ins I64Attr:$imm, VectorOfLengthAndType<[32], [F32]>:$reg)>{
  let summary = "vector move immediate operation";
  let assemblyFormat = "$imm `,` $reg attr-dict `:` type($reg)";
}

def Mt_SldwOp :
  Op<Mt_Dialect, "SLDW">,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[2], [F32]>:$dst,
                 Optional<I64>:$offset)> {
  let summary = "scalar load word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VldwOp :
  Op<Mt_Dialect, "VLDW">,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[32], [F32]>:$dst,
                 Optional<I64>:$offset)> {
  let summary = "vector load word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VstwOp :
  Op<Mt_Dialect, "VSTW">,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src,
                 LLVM_AnyPointer:$base,
                 Optional<I64>:$offset)> {
  let summary = "vector store word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_SvbcastOp :
  Op<Mt_Dialect, "SVBCAST">,
  Arguments<(ins VectorOfLengthAndType<[2], [F32]>:$src,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "scalar vector broadcast operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Mt_Vbale2Op :
  Op<Mt_Dialect, "VBALE2">,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "This instruction packs the low half-word of source operand Src1 with the low half-word of source operand Src2 into a 64-bit value stored in Dst";
  let assemblyFormat = "$src1 `,` $src2 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($dst)";
}

def Mt_Vbale2hOp :
  Op<Mt_Dialect, "VBALE2H">,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "This instruction packs the high half-word of source operand Src1 with the high half-word of source operand Src2 into a 64-bit value stored in Dst";
  let assemblyFormat = "$src1 `,` $src2 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($dst)";
}

def Mt_Vfmulas32Op :
  Op<Mt_Dialect, "VFMULAS32">,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$src3,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "vector fused multiply-add single precision floating-point operation";
  let assemblyFormat = "$src1 `,` $src2 `,` $src3 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($src3) `,` type($dst)";
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