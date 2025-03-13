//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all
// ftm-specific dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_INITALLDIALECTS_H
#define MTAS_INITALLDIALECTS_H

#include "mtas/Dialect/Ftm/IR/Ftm.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mtas {

/// Add all the ftm-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::ftm::FtmDialect>();
  // clang-format on
}

/// Append all the mtas-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mtas::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mtas

#endif // MTAS_INITALLDIALECTS_H
