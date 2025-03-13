//===-- Passes.td - Ftm dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_FTM_TRANSFORMS_PASSES
#define MTAS_DIALECT_FTM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// def FtmOpFusion : Pass<"ftm-fuse-ops", "ModuleOp"> {
//   let summary = "Ftm Fuse operations on tensors";
//   let constructor = "ftm::createFtmOpFusionPass()";
//   let dependentDialects = [
//     "ftm::FtmDialect",
//   ];
//   let options = [
//     Option<"outputMode", "output-mode", "::mlir::ftm::OutputMode",
//       "ftm::OutputMode::Multiple",
//       "Outlined function output mode (default is multi, can also use single or single-aggr)",
//       [{::llvm::cl::values(
//           clEnumValN(ftm::OutputMode::Multiple, "multi",
//                     "Outlined function with multiple outputs"),
//           clEnumValN(ftm::OutputMode::Single, "single",
//                     "Outlined function with single output"),
//           clEnumValN(ftm::OutputMode::SingleAggressive, "single-aggr",
//                     "Outlined function with single output, fusing "
//                     "more aggressively by duplicating operations")
//       )}]>,
//     Option<"fusionMode", "fusion-mode", "::mlir::ftm::FusionKind",
//       "ftm::FusionKind::Unknown", "Fusion kind is determined by label",
//       [{::llvm::cl::values(
//           clEnumValN(ftm::FusionKind::PureElemwise, "PURE_ELEMWISE", ""),
//           clEnumValN(ftm::FusionKind::AnyPB, "ANY_PB", ""),
//           clEnumValN(ftm::FusionKind::LastAxisPBR, "LAST_AXIS_PBR", ""),
//           clEnumValN(ftm::FusionKind::MixCV, "MIX_CV", ""),
//           clEnumValN(ftm::FusionKind::ShallowCV, "SHALLOW_CV", ""),
//           clEnumValN(ftm::FusionKind::Unknown, "UNKNOWN", "")
//       )}]>,
//     Option<"alwaysInline", "always-inline", "bool", /*default=*/"false",
//                "Enable always inline for the outline function.">,
//     Option<"moveOutToParam", "move-out-to-param", "bool", /* default */"true",
//            "Whether move the tensor out to params or not">,
//     Option<"maxHorizontalFusionSize", "max-horizontal-fusion-size", "int", /* default */"-1",
//            "Maximum horizontal (non-dependent) fusioning allowed, -1 for unlimited attempt"
//            "of horizontal fusion">
//   ];
// }

// def AutoSchedule : Pass<"ftm-auto-schedule", "ModuleOp"> {
//   let summary = "Auto schedule fused kernels.";
//   let constructor = "mlir::ftm::createFtmAutoSchedulePass()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "ftm::FtmDialect",
//                            "transform::TransformDialect",
//                            "arith::ArithDialect"];
//   let options = [
//     Option<"blockDim", "block-dim", "unsigned", "1", "Number of blocks to use">,
//     Option<"enableAutoMultiBuffer", "enable-auto-multi-buffer", "bool", "false",
//            "Enable auto multi buffer">,
//     Option<"maxBufferCntTuning", "max-buffer-count-tuning", "int64_t", "0",
//            "allow maxBufferCnt tuning">,
//   ];
// }

def MultiBuffering : Pass<"ftm-multi-buffering", "func::FuncOp"> {
  let summary = "multiple buffering memref.copy";
  let constructor = "mlir::ftm::createMultiBufferingPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "ftm::FtmDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"enablePostStore", "enable-post-store", "bool", "false",
           "If the storing stage is postponed one iteration later.">,
  ];
}

def TileDynamicDims : Pass<"tile-dynamic-dims", "func::FuncOp"> {
  let summary = "A pass to tile dynamic dimensions";
  let constructor = "mlir::ftm::createTileDynamicDimsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "ftm::FtmDialect",
  ];
}

def LoopUnrolling : Pass<"loop-unrolling", "func::FuncOp"> {
  let summary = "A pass to unroll the innermost loop";
  let constructor = "mlir::ftm::createLoopUnrollingPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "ftm::FtmDialect",
  ];
  let options = [
    Option<"unrollingFactor", "unrolling-factor", "unsigned", "2", "loop unrolling factor">,
  ];
}

def SplitMatmul : Pass<"split-matmul", "func::FuncOp"> {
  let summary = "A pass to split linalg.matmul";
  let constructor = "mlir::ftm::createSplitMatmulPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "ftm::FtmDialect",
  ];
}

def LowerMatmulToFma : Pass<"lower-matmul-to-fma", "func::FuncOp"> {
  let summary = "A pass to lower linalg.matmul to fma";
  let constructor = "mlir::ftm::createLowerMatmulToFmaPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "ftm::FtmDialect",
  ];
}

#endif // MTAS_DIALECT_FTM_TRANSFORMS_PASSES