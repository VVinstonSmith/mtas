//===- MtasmBase.td - Base definitions for mt3000 fusion dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTASM_IR_MTASMBASE
#define MTASM_IR_MTASMBASE

include "mtas/Dialect/Mtasm/IR/MtasmEnums.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Mtasm_Dialect : Dialect {
  let name = "mtasm";
  let cppNamespace = "::mlir::mtasm";
  let description = [{
    Mtasm (Mt3000 Fusion) dialect.
  }];
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
  let hasCanonicalizer = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // MTASM_IR_MTASMBASE
