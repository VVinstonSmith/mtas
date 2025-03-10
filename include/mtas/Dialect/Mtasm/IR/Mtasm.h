//===- Mtasm.h - Mt3000 ASM dialect -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MTFUSION_IR_MTFUSION_H
#define MTAS_DIALECT_MTFUSION_IR_MTFUSION_H

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
namespace mtasm {

class MtasmOp;

} // namespace mtasm
} // namespace mlir

//===----------------------------------------------------------------------===//
// Mtasm Dialect
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mtasm/IR/MtasmOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Mtasm Enums
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mtasm/IR/MtasmEnums.h.inc"

//===----------------------------------------------------------------------===//
// Mtasm Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mtas/Dialect/Mtasm/IR/MtasmAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Mtasm Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mtas/Dialect/Mtasm/IR/MtasmOps.h.inc"

#endif // MTAS_DIALECT_MTASM_IR_MTASM_H

