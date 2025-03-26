//===- Passes.h - Ftm dialect pass entrypoints --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef MTAS_DIALECT_FTM_TRANSFORMS_PASSES_H
#define MTAS_DIALECT_FTM_TRANSFORMS_PASSES_H

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"

namespace ftm {

/// Create a pass to do multi-buffering
std::unique_ptr<Pass> createMultiBufferingPass();

/// Create a pass to tile dimensions of linalg op
std::unique_ptr<Pass> createTileLinalgDimsPass();

/// Create a pass to unroll the innermost loop
std::unique_ptr<Pass> createLoopUnrollingPass();

/// Create a pass to fold loops with static parameters
std::unique_ptr<Pass> createLoopFoldingPass();

/// Create a pass to split linalg.matmul
std::unique_ptr<Pass> createSplitMatmulPass();

/// Create a pass to lower linalg ops
std::unique_ptr<Pass> createLowerLinalgOpsPass();

/// Create a pass to lower kernel arguments
std::unique_ptr<Pass> createLowerKernelArgumentsPass();

/// Create a pass to lower lower load and store memref to ptr
std::unique_ptr<Pass> createLowerLoadAndStoreMemRefToPtrPass();

/// Create a pass to cast ptr to int64
std::unique_ptr<Pass> createCastPtrToInt64Pass();

/// Create a pass to fold register alloca
std::unique_ptr<Pass> createFoldRegisterAllocaPass();

/// Create a pass to allocate offset registers
std::unique_ptr<Pass> createAllocateOffsetRegistersPass();

/// Create a pass to reduce loop strength
std::unique_ptr<Pass> createLoopStrengthReducePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mtas/Dialect/Ftm/Transforms/Passes.h.inc"

} // namespace ftm
} // namespace mlir

#endif // MTAS_DIALECT_FTM_TRANSFORMS_PASSES_H
