//===- mtas-opt.cpp - Ftm Optimizer Driver -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mtas-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mtas/InitAllDialects.h"
#include "mtas/InitAllPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mtas::registerAllDialects(registry);
  // Register dialect extensions.
  mlir::registerAllExtensions(registry);
  // Register passes.
  mlir::registerAllPasses();
  mtas::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MT3000 ASM MLIR optimizer driver\n", registry));
}
