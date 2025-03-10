//===- Passes.h - Mtasm dialect pass entrypoints --------------*- C++ -*-===//
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
#ifndef MTAS_DIALECT_MTASM_TRANSFORMS_PASSES_H
#define MTAS_DIALECT_MTASM_TRANSFORMS_PASSES_H

#include "mtas/Dialect/Mtasm/IR/Mtasm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace mtasm {
namespace opfusion {
class FusableHelper;
class FusableBlock;
using FusableBlocks = SmallVector<FusableBlock, 8>;
} // namespace opfusion

} // namespace mtasm
} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "mtas/Dialect/Mtasm/Transforms/Passes.h.inc"

namespace mtasm {

/// Create a pass to do multi-buffering
std::unique_ptr<Pass> createMultiBufferingPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mtas/Dialect/Mtasm/Transforms/Passes.h.inc"

} // namespace mtasm
} // namespace mlir

#endif // MTAS_DIALECT_MTASM_TRANSFORMS_PASSES_H
