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

def LowerFtmToMt : Pass<"lower-ftm-to-mt", "func::FuncOp"> {
  let summary = "A pass to lower all ftm dialect operations to mt dialect";
  let constructor = "mlir::mt::createLowerFtmToMtPass()";
  let dependentDialects = [
    "mt::MtDialect",
    "ftm::FtmDialect"
  ];
}

def GenerateConditionAndLabelsForLoops : Pass<"generate-condition-and-labels-for-loops", "func::FuncOp"> {
  let summary = "A pass to generate condition checking and labels for for loops";
  let constructor = "mlir::mt::createGenerateConditionAndLabelsForLoopsPass()";
  let dependentDialects = [
    "mt::MtDialect",
    "scf::SCFDialect"
  ];
}

#endif // MTAS_DIALECT_MT_TRANSFORMS_PASSES