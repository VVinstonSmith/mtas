//===- ConversionPasses.td - Conversion pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_CONVERSION_PASSES_H
#define MTAS_CONVERSION_PASSES_H

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// MemRefToPtr
//===----------------------------------------------------------------------===//

def ConvertMemRefToPtr : Pass<"convert-memref-to-ptr"> {
  let summary = "Convert MemRef types to ptr";
  let constructor = "mlir::createMemRefToPtrConversionPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "memref::MemRefDialect", 
    "LLVM::LLVMDialect",
    "ftm::FtmDialect",
  ];
}

#endif // MTAS_CONVERSION_PASSES_H