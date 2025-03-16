//===- MemRefToPtr.h - memref to ptr conversion ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_CONVERSION_MEMREFTOPTR_MEMREFTOPTR_H
#define MTAS_CONVERSION_MEMREFTOPTR_MEMREFTOPTR_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTMEMREFTOPTR
#include "mtas/Conversion/Passes.h.inc"

namespace mtas {
void populateMemRefToPtrConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
} // namespace mtivm

/// Creates a pass to convert the MtFusion dialect to the MtIVM dialect.
std::unique_ptr<Pass> createMemRefToPtrConversionPass();

} // namespace mlir

#endif // MTAS_CONVERSION_MEMREFTOPTR_MEMREFTOPTR_H