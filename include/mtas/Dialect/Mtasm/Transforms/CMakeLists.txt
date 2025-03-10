set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mtasm)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Mtasm)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Mtasm)
add_public_tablegen_target(MLIRMtasmPassIncGen)

add_mlir_doc(Passes MtasmPasses ./ -gen-pass-doc)