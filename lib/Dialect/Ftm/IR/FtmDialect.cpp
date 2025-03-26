//===- FtmDialect.cpp - Implementation of Ftm dialect and types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ftm;

#define GET_ATTRDEF_CLASSES
#include "mtas/Dialect/Ftm/IR/FtmAttrs.cpp.inc"

void mlir::ftm::FtmDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mtas/Dialect/Ftm/IR/FtmOps.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "mtas/Dialect/Ftm/IR/FtmAttrs.cpp.inc"
        >();
}

#include "mtas/Dialect/Ftm/IR/FtmEnums.cpp.inc"

#include "mtas/Dialect/Ftm/IR/FtmOpsDialect.cpp.inc"
