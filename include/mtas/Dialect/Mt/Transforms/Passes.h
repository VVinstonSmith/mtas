//===- Passes.h - Mt dialect pass entrypoints --------------*- C++ -*-===//
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
#ifndef MTAS_DIALECT_MT_TRANSFORMS_PASSES_H
#define MTAS_DIALECT_MT_TRANSFORMS_PASSES_H

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"

namespace mt {

/// Create a pass to allocate address registers for pointer arguments in loops
std::unique_ptr<Pass> createAllocateAddressRegistersPass();

/// Create a pass to convert SCF ForOp index type to i64 type
std::unique_ptr<Pass> createConvertForOpIndexToI64Pass();

/// Create a pass to allocate scalar registers for constants
std::unique_ptr<Pass> createAllocateScalarRegistersForConstantsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"

} // namespace mt
} // namespace mlir

#endif // MTAS_DIALECT_MT_TRANSFORMS_PASSES_H
