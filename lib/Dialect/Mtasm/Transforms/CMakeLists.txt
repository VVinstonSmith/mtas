add_mlir_dialect_library(MLIRMtasmTransforms
    MultiBuffering.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mtasm

    DEPENDS
    MLIRMtasmPassIncGen

    LINK_LIBS PUBLIC
    MLIRTransformDialectTransforms
)