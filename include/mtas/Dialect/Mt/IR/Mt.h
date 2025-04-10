//===- Mt.h - Mt3000 ASM dialect -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MT_IR_MT_H
#define MTAS_DIALECT_MT_IR_MT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
namespace mt {

class MtOp;

} // namespace mt
} // namespace mlir

//===----------------------------------------------------------------------===//
// Mt Dialect
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/IR/MtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Mt Enums
//===----------------------------------------------------------------------===//

// #include "mtas/Dialect/Mt/IR/MtEnums.h.inc"

//===----------------------------------------------------------------------===//
// Mt Attributes
//===----------------------------------------------------------------------===//

// #define GET_ATTRDEF_CLASSES
// #include "mtas/Dialect/Mt/IR/MtAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Mt Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mtas/Dialect/Mt/IR/MtOps.h.inc"

#endif // MTAS_DIALECT_MT_IR_MT_H

