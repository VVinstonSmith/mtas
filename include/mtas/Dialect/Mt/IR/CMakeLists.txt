
add_mlir_dialect(MtOps mt)
add_mlir_doc(MtOps MtOps Dialects/ -gen-op-doc)

# set(LLVM_TARGET_DEFINITIONS MtEnums.td)
# mlir_tablegen(MtEnums.h.inc -gen-enum-decls)
# mlir_tablegen(MtEnums.cpp.inc -gen-enum-defs)
# add_public_tablegen_target(MLIRMtEnumsIncGen)

# set(LLVM_TARGET_DEFINITIONS MtAttrs.td)
# mlir_tablegen(MtAttrs.h.inc -gen-attrdef-decls)
# mlir_tablegen(MtAttrs.cpp.inc -gen-attrdef-defs)
# add_public_tablegen_target(MLIRMtAttrsIncGen)



