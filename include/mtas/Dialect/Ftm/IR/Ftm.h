//===- Ftm.h - Mt3000 ASM dialect -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_FTM_IR_FTM_H
#define MTAS_DIALECT_FTM_IR_FTM_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir {
namespace ftm {

class FtmOp;

} // namespace ftm
} // namespace mlir

//===----------------------------------------------------------------------===//
// Ftm Dialect
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/FtmOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Ftm Enums
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Ftm/IR/FtmEnums.h.inc"

//===----------------------------------------------------------------------===//
// Ftm Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mtas/Dialect/Ftm/IR/FtmAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Ftm Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mtas/Dialect/Ftm/IR/FtmOps.h.inc"

#endif // MTAS_DIALECT_FTM_IR_FTM_H

