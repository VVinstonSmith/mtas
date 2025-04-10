add_mlir_dialect_library(MLIRMtTransforms
    AllocateAddressRegisters.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mt

    DEPENDS
    MLIRMtPassIncGen

    LINK_LIBS PUBLIC
    MLIRTransformDialectTransforms
)