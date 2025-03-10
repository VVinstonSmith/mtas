//===- InitAllPasses.h - MLIR Passes Registration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_INITALLPASSES_H
#define MTAS_INITALLPASSES_H

#include "mtas/Dialect/Mtasm/Transforms/Passes.h"

namespace mtas {

// This function may be called to register the hivm-specific MLIR passes with
// the global registry.
inline void registerAllPasses() {
  // Dialect passes
  mlir::mtasm::registerMtasmPasses();
}

} // namespace mtas

#endif // MTAS_INITALLPASSES_H
