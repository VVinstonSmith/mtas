
add_mlir_dialect(FtmOps ftm)
add_mlir_doc(FtmOps FtmOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FtmEnums.td)
mlir_tablegen(FtmEnums.h.inc -gen-enum-decls)
mlir_tablegen(FtmEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRFtmEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS FtmAttrs.td)
mlir_tablegen(FtmAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(FtmAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFtmAttrsIncGen)



