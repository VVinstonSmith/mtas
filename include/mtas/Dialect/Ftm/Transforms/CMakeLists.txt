set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Ftm)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Ftm)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Ftm)
add_public_tablegen_target(MLIRFtmPassIncGen)

add_mlir_doc(Passes FtmPasses ./ -gen-pass-doc)