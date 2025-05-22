#ifndef MT_DOUBLEWORDMEMORYACCESSINTERFACE_H_
#define MT_DOUBLEWORDMEMORYACCESSINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace mt {

/// Include the auto-generated declarations.
#include "mtas/Dialect/Mt/IR/DoubleWordMemoryAccessInterface.h.inc"

} // namespace mt
} // namespace mlir

#endif // MT_DOUBLEWORDMEMORYACCESSINTERFACE_H_