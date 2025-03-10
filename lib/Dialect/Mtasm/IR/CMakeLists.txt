add_mlir_dialect_library(MLIRMtasmDialect
    MtasmOps.cpp
    MtasmDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mtasm

    DEPENDS
    MLIRMtasmOpsIncGen
    MLIRMtasmEnumsIncGen
    MLIRMtasmAttrsIncGen

    LINK_COMPONENTS
    Core
)
