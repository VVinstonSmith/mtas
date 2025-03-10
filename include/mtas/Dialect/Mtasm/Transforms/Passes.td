//===-- Passes.td - Mtasm dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTAS_DIALECT_MTASM_TRANSFORMS_PASSES
#define MTAS_DIALECT_MTASM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// def MtasmOpFusion : Pass<"mtasm-fuse-ops", "ModuleOp"> {
//   let summary = "Mtasm Fuse operations on tensors";
//   let constructor = "mtasm::createMtasmOpFusionPass()";
//   let dependentDialects = [
//     "mtasm::MtasmDialect",
//   ];
//   let options = [
//     Option<"outputMode", "output-mode", "::mlir::mtasm::OutputMode",
//       "mtasm::OutputMode::Multiple",
//       "Outlined function output mode (default is multi, can also use single or single-aggr)",
//       [{::llvm::cl::values(
//           clEnumValN(mtasm::OutputMode::Multiple, "multi",
//                     "Outlined function with multiple outputs"),
//           clEnumValN(mtasm::OutputMode::Single, "single",
//                     "Outlined function with single output"),
//           clEnumValN(mtasm::OutputMode::SingleAggressive, "single-aggr",
//                     "Outlined function with single output, fusing "
//                     "more aggressively by duplicating operations")
//       )}]>,
//     Option<"fusionMode", "fusion-mode", "::mlir::mtasm::FusionKind",
//       "mtasm::FusionKind::Unknown", "Fusion kind is determined by label",
//       [{::llvm::cl::values(
//           clEnumValN(mtasm::FusionKind::PureElemwise, "PURE_ELEMWISE", ""),
//           clEnumValN(mtasm::FusionKind::AnyPB, "ANY_PB", ""),
//           clEnumValN(mtasm::FusionKind::LastAxisPBR, "LAST_AXIS_PBR", ""),
//           clEnumValN(mtasm::FusionKind::MixCV, "MIX_CV", ""),
//           clEnumValN(mtasm::FusionKind::ShallowCV, "SHALLOW_CV", ""),
//           clEnumValN(mtasm::FusionKind::Unknown, "UNKNOWN", "")
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

// def AutoSchedule : Pass<"mtasm-auto-schedule", "ModuleOp"> {
//   let summary = "Auto schedule fused kernels.";
//   let constructor = "mlir::mtasm::createMtasmAutoSchedulePass()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtasm::MtasmDialect",
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

def MultiBuffering : Pass<"mtasm-multi-buffering", "func::FuncOp"> {
  let summary = "multiple buffering memref.copy";
  let constructor = "mlir::mtasm::createMultiBufferingPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "mtasm::MtasmDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"enablePostStore", "enable-post-store", "bool", "false",
           "If the storing stage is postponed one iteration later.">,
  ];
}

#endif // MTAS_DIALECT_MTASM_TRANSFORMS_PASSES