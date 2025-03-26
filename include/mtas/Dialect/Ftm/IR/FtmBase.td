//===- FtmBase.td - Base definitions for mt3000 fusion dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FTM_IR_FTMBASE
#define FTM_IR_FTMBASE

include "mtas/Dialect/Ftm/IR/FtmEnums.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Ftm_Dialect : Dialect {
  let name = "ftm";
  let cppNamespace = "::mlir::ftm";
  let description = [{
    Ftm (Ftm7032) dialect.
  }];
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
  let hasCanonicalizer = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // FTM_IR_FTMBASE
