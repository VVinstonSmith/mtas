//===- MtDialect.cpp - Implementation of Mt dialect and types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mt;

// #define GET_ATTRDEF_CLASSES
// #include "mtas/Dialect/Mt/IR/MtAttrs.cpp.inc"

void mlir::mt::MtDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mtas/Dialect/Mt/IR/MtOps.cpp.inc"
        >();

//     addAttributes<
// #define GET_ATTRDEF_LIST
// #include "mtas/Dialect/Mt/IR/MtAttrs.cpp.inc"
//         >();
}

#include "mtas/Dialect/Mt/IR/MtEnums.cpp.inc"

#include "mtas/Dialect/Mt/IR/MtOpsDialect.cpp.inc"
