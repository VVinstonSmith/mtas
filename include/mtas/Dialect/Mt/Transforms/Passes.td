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

def Reduce80BitInstructions : Pass<"reduce-80bit-instructions", "func::FuncOp"> {
  let summary = "A pass to reduce the number of 80-bit instructions";
  let description = "Optimizes code to use fewer 80-bit instructions by replacing them with equivalent sequences of narrower instructions where possible";
  let constructor = "mlir::mt::createReduce80BitInstructionsPass()";
  let dependentDialects = [
    "mt::MtDialect"
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

def AssignRegisterIds : Pass<"assign-register-ids", "func::FuncOp"> {
  let summary = "A pass to assign unique IDs to registers";
  let constructor = "mlir::mt::createAssignRegisterIdsPass()";
  let dependentDialects = [
    "mt::MtDialect",
  ];
}

def InstructionSchedulingAndPacking : Pass<"instruction-scheduling-and-packing", "func::FuncOp"> {
  let summary = "A pass to perform instruction scheduling and packing";
  let constructor = "mlir::mt::createInstructionSchedulingAndPackingPass()";
  let dependentDialects = [
    "mt::MtDialect"
  ];
}

def MatMulInPlaceAccumulation : Pass<"matmul-in-place-accumulation", "func::FuncOp"> {
  let summary = "A pass to optimize MatMul by accumulating directly on destination registers";
  let constructor = "mlir::mt::createMatMulInPlaceAccumulationPass()";
  let dependentDialects = [
    "mt::MtDialect"
  ];
}

def RegisterReuseOptimization : Pass<"register-reuse-optimization", "func::FuncOp"> {
  let summary = "A pass to optimize register usage by reusing registers that are no longer needed";
  let constructor = "mlir::mt::createRegisterReuseOptimizationPass()";
  let dependentDialects = [
    "mt::MtDialect"
  ];
}

def MtDeadCodeElimination : Pass<"mt-dead-code-elimination", "func::FuncOp"> {
  let summary = "A pass to eliminate dead code in the Mt dialect";
  let constructor = "mlir::mt::createMtDeadCodeEliminationPass()";
  let dependentDialects = [
    "mt::MtDialect"
  ];
}


#endif // MTAS_DIALECT_MT_TRANSFORMS_PASSES