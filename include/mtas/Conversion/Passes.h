//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_CONVERSION_PASSES_H
#define MTAS_CONVERSION_PASSES_H

#include "mtas/Conversion/MemRefToPtr/MemRefToPtr.h"

#include "mlir/Pass/Pass.h"

namespace mtas {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mtas/Conversion/Passes.h.inc"

} // namespace mtas

#endif // MTAS_CONVERSION_PASSES_H