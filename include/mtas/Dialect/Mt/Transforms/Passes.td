//===-- Passes.td - Mt dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_TRANSFORMS_PASSES
#define MTAS_DIALECT_MT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AllocateAddressRegisters : Pass<"allocate-address-registers", "func::FuncOp"> {
  let summary = "A pass to allocate address registers for pointer arguments in loops";
  let constructor = "mlir::mt::createAllocateAddressRegistersPass()";
  let dependentDialects = [
    "mt::MtDialect",
  ];
}

def ConvertForOpIndexToI64 : Pass<"convert-for-index-to-i64", "func::FuncOp"> {
  let summary = "A pass to convert SCF ForOp index type to i64 type";
  let constructor = "mlir::mt::createConvertForOpIndexToI64Pass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];
}

def AllocateScalarRegistersForConstants : Pass<"allocate-scalar-registers-for-constants", "func::FuncOp"> {
  let summary = "A pass to allocate scalar registers for constants";
  let constructor = "mlir::mt::createAllocateScalarRegistersForConstantsPass()";
  let dependentDialects = [
    "mt::MtDialect",
  ];
}

#endif // MTAS_DIALECT_MT_TRANSFORMS_PASSES