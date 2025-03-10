//===-------- MtasmOps.td - Mtasm op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTASM_IR_MTASMOPS
#define MTASM_IR_MTASMOPS

include "mtas/Dialect/Mtasm/IR/MtasmBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for Mtasm dialect ops that do not correspond to library calls.
class Mtasm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mtasm_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AnnotateOp
//===----------------------------------------------------------------------===//

def Mtasm_AnnotateOp : Mtasm_Op<"annotate"> {
  let summary = "annotate operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // MTASM_IR_MTASMOPS