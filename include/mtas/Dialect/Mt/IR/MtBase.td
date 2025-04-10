//===- MtBase.td - Base definitions for mt3000 fusion dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MT_IR_MTBASE
#define MT_IR_MTBASE

// include "mtas/Dialect/Mt/IR/MtEnums.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Mt_Dialect : Dialect {
  let name = "mt";
  let cppNamespace = "::mlir::mt";
  let description = [{
    The Mt dialect represents the assembly instruction set for the MT3000 DSP architecture.
    This dialect provides operations and attributes specific to MT3000 DSP assembly programming.
  }];
  let hasCanonicalizer = 1;
//   let useDefaultAttributePrinterParser = 1;
}

#endif // MT_IR_MTBASE
