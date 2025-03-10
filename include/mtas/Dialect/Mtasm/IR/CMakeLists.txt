
add_mlir_dialect(MtasmOps mtasm)
add_mlir_doc(MtasmOps MtasmOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MtasmEnums.td)
mlir_tablegen(MtasmEnums.h.inc -gen-enum-decls)
mlir_tablegen(MtasmEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMtasmEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS MtasmAttrs.td)
mlir_tablegen(MtasmAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(MtasmAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMtasmAttrsIncGen)



