#ifndef MT_INSTRUCTIONSCHEDULINGINTERFACE_H_
#define MT_INSTRUCTIONSCHEDULINGINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include "mtas/Dialect/Mt/IR/MtEnums.h.inc"

namespace mlir {
namespace mt {

/// Include the auto-generated declarations.
#include "mtas/Dialect/Mt/IR/InstructionSchedulingInterface.h.inc"

} // namespace mt
} // namespace mlir

#endif // MT_INSTRUCTIONSCHEDULINGINTERFACE_H_