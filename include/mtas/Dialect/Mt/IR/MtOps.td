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
include "mtas/Dialect/Mt/IR/InstructionSchedulingInterface.td"
include "mtas/Dialect/Mt/IR/DoubleWordMemoryAccessInterface.td"


class Mt_Op<string mnemonic, list<Trait> traits = []> :
  Op<Mt_Dialect, mnemonic, !listconcat(traits, [
    DeclareOpInterfaceMethods<InstructionSchedulingInterface>,
  ])>;

def Mt_DeclareRegisterOp: 
  Op<Mt_Dialect, "declare_register">,
  Results<(outs AnyType:$output)> {
  let summary = "declare register operation";
  let assemblyFormat = "attr-dict `:` type($output)";
  let extraClassDeclaration = [{
    // 格式化寄存器输出
    std::string formatRegisterOutput();
  }];
}

def Mt_SmvagaOp :
  Mt_Op<"SMVAGA", [
    AllTypesMatch<["src", "dst"]>, 
  ]>,
  Arguments<(ins AnyTypeOf<[I64, LLVM_AnyPointer]>:$src,
                 AnyTypeOf<[I64, LLVM_AnyPointer]>:$dst)>{
  let summary = "scalar move to address/offset register operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
}

def Mt_SmoviOp :
  Mt_Op<"SMOVI", [
  ]>,
  Arguments<(ins I64Attr:$imm, I64:$reg)>{
  let summary = "scalar move immediate operation";
  let assemblyFormat = "$imm `,` $reg attr-dict `:` type($reg)";
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
    int getOperationSize();
  }];
}

def Mt_VmoviOp :
  Mt_Op<"VMOVI", [
  ]>,
  Arguments<(ins I64Attr:$imm, VectorOfLengthAndType<[32], [F32]>:$reg)>{
  let summary = "vector move immediate operation";
  let assemblyFormat = "$imm `,` $reg attr-dict `:` type($reg)";
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
    int getOperationSize();
  }];
}

def Mt_SldwOp :
  Mt_Op<"SLDW", [
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[2], [F32]>:$dst,
                 Optional<I64>:$offset)> {
  let summary = "scalar load word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VldwOp :
  Mt_Op<"VLDW", [
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[32], [F32]>:$dst,
                 Optional<I64>:$offset)> {
  let summary = "vector load word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VstwOp :
  Mt_Op<"VSTW", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src,
                 LLVM_AnyPointer:$base,
                 Optional<I64>:$offset)> {
  let summary = "vector store word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_SldwiOp :
  Mt_Op<"SLDWI", [
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[2], [F32]>:$dst,
                 I64Attr:$immOffset)> {
  let summary = "scalar load word with immediate offset operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VldwiOp :
  Mt_Op<"VLDWI", [
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[32], [F32]>:$dst,
                 I64Attr:$immOffset)> {
  let summary = "vector load word with immediate offset operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VstwiOp :
  Mt_Op<"VSTWI", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src,
                 LLVM_AnyPointer:$base,
                 I64Attr:$immOffset)> {
  let summary = "vector store word with immediate offset operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VlddwOp :
  Mt_Op<"VLDDW", [
    DeclareOpInterfaceMethods<DoubleWordMemoryAccessInterface>, 
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[32], [F32]>:$dst1,
                 VectorOfLengthAndType<[32], [F32]>:$dst2,
                 Optional<I64>:$offset)> {
  let summary = "vector load double word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VstdwOp :
  Mt_Op<"VSTDW", [
    DeclareOpInterfaceMethods<DoubleWordMemoryAccessInterface>, 
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 LLVM_AnyPointer:$base,
                 Optional<I64>:$offset)> {
  let summary = "vector store double word operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VlddwiOp :
  Mt_Op<"VLDDWI", [
    DeclareOpInterfaceMethods<DoubleWordMemoryAccessInterface>, 
  ]>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 VectorOfLengthAndType<[32], [F32]>:$dst1,
                 VectorOfLengthAndType<[32], [F32]>:$dst2,
                 I64Attr:$immOffset)> {
  let summary = "vector load double word with immediate offset operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_VstdwiOp :
  Mt_Op<"VSTDWI", [
    DeclareOpInterfaceMethods<DoubleWordMemoryAccessInterface>, 
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 LLVM_AnyPointer:$base,
                 I64Attr:$immOffset)> {
  let summary = "vector store double word with immediate offset operation";
  let hasCustomAssemblyFormat = 1;
}

def Mt_SvbcastOp :
  Mt_Op<"SVBCAST", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[2], [F32]>:$src,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "scalar vector broadcast operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Mt_Vbale2Op :
  Mt_Op<"VBALE2", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "This instruction packs the low half-word of source operand Src1 with the low half-word of source operand Src2 into a 64-bit value stored in Dst";
  let assemblyFormat = "$src1 `,` $src2 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($dst)";
}

def Mt_Vbale2hOp :
  Mt_Op<"VBALE2H", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "This instruction packs the high half-word of source operand Src1 with the high half-word of source operand Src2 into a 64-bit value stored in Dst";
  let assemblyFormat = "$src1 `,` $src2 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($dst)";
}

def Mt_Vfmulas32Op :
  Mt_Op<"VFMULAS32", [
  ]>,
  Arguments<(ins VectorOfLengthAndType<[32], [F32]>:$src1,
                 VectorOfLengthAndType<[32], [F32]>:$src2,
                 VectorOfLengthAndType<[32], [F32]>:$src3,
                 VectorOfLengthAndType<[32], [F32]>:$dst)> {
  let summary = "vector fused multiply-add single precision floating-point operation";
  let assemblyFormat = "$src1 `,` $src2 `,` $src3 `,` $dst attr-dict `:` type($src1) `,` type($src2) `,` type($src3) `,` type($dst)";
}

def Mt_SmovOp :
  Mt_Op<"SMOV", [
    AllTypesMatch<["src", "dst"]>,
  ]>,
  Arguments<(ins I64:$src, 
                 I64:$dst)>{
  let summary = "scalar move operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
  }];
}

def Mt_VmovOp :
  Mt_Op<"VMOV", [
    AllTypesMatch<["src", "dst"]>,
  ]>,
  Arguments<(ins VectorOfAnyRankOf<[AnyFloat]>:$src,
                 VectorOfAnyRankOf<[AnyFloat]>:$dst)>{
  let summary = "vector move operation";
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($dst)";
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
  }];
}

def Mt_SaddaOp :
  Mt_Op<"SADDA", [
    AllTypesMatch<["rhs", "res"]>,
  ]>,
  Arguments<(ins I64:$lhs,
                 LLVM_AnyPointer:$rhs,
                 LLVM_AnyPointer:$res)>{
  let summary = "address addition operation";
  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($res)";
  let extraClassDeclaration = [{
    LLVM::LLVMPointerType getPointerType() { return ::llvm::cast<LLVM::LLVMPointerType>(getRes().getType()); }
  }];
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
  }];
}

def Mt_SaddOp :
  Mt_Op<"SADD", [
    AllTypesMatch<["lhs", "rhs", "res"]>,
  ]>,
  Arguments<(ins I64:$lhs,
                 I64:$rhs,
                 I64:$res)>{
  let summary = "scalar addition operation";
  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($res)";
  let extraClassDeclaration = [{
    // 覆盖 getSchedulingPriority 方法
    int getSchedulingPriority();
  }];
}

def Mt_SltOp :
  Mt_Op<"SLT", [
    AllTypesMatch<["lhs", "rhs", "res"]>,
  ]>,
  Arguments<(ins I64:$lhs,
                 I64:$rhs,
                 I64:$res)>{
  let summary = "scalar less than operation";
  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($res)";
}

def Mt_LabelOp :
  Op<Mt_Dialect, "LABEL", [
  ]>,
  Arguments<(ins StrAttr:$name)> {
  let summary = "Label operation for control flow";
  let description = [{
    This operation defines a label that can be used as a target for branching operations.
    It is primarily used for implementing control flow structures like loops.
    
    Example:
      mt.LABEL "loop_start"
  }];
  // let assemblyFormat = "$name attr-dict";
  let hasCustomAssemblyFormat = 1;
}

def Mt_SbrLabelOp :
  Mt_Op<"SBR_label", [
  ]>,
  Arguments<(ins StrAttr:$label, Optional<I64>:$cond)> {
  let summary = "Scalar branch to label operation";
  let description = [{
    This operation implements a branch to a label.
    It has 6 branch delay slots. The branch target is specified by a label.
    Optionally controlled by a condition register (R0-R6).
    
    Example:
      mt.sbr_label "loop_start" : () -> ()       // Unconditional branch
      mt.sbr_label "loop_start", %cond : i64     // Conditional branch
  }];
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    // Whether this branch has a condition
    bool hasCondition() { return getCond() != nullptr; }
  }];
}

def Mt_SbrRegOp :
  Mt_Op<"SBR_reg", [
  ]>,
  Arguments<(ins I64:$target, Optional<I64>:$cond)> {
  let summary = "Scalar branch to register address operation";
  let description = [{
    This operation implements a branch to the address contained in a register.
    It has 6 branch delay slots. The branch target is specified by a register.
    Optionally controlled by a condition register (R0-R6).
    
    Example:
      mt.sbr_reg %addr : i64                   // Unconditional branch
      mt.sbr_reg %addr, %cond : i64, i64       // Conditional branch
  }];
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    // Whether this branch has a condition
    bool hasCondition() { return getCond() != nullptr; }
  }];
}

#endif // MT_IR_MTOPS