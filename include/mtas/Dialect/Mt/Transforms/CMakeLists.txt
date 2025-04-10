set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mt)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Mt)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Mt)
add_public_tablegen_target(MLIRMtPassIncGen)

add_mlir_doc(Passes MtPasses ./ -gen-pass-doc)