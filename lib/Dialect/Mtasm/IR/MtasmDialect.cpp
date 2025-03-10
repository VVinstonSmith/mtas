//===- MtasmDialect.cpp - Implementation of Mtasm dialect and types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mtasm/IR/Mtasm.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mtasm;

#define GET_ATTRDEF_CLASSES
#include "mtas/Dialect/Mtasm/IR/MtasmAttrs.cpp.inc"

void mlir::mtasm::MtasmDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mtas/Dialect/Mtasm/IR/MtasmOps.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "mtas/Dialect/Mtasm/IR/MtasmAttrs.cpp.inc"
        >();
}

#include "mtas/Dialect/Mtasm/IR/MtasmEnums.cpp.inc"

#include "mtas/Dialect/Mtasm/IR/MtasmOpsDialect.cpp.inc"
