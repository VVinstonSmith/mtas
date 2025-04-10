add_mlir_dialect_library(MLIRMtDialect
    MtOps.cpp
    MtDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mt

    DEPENDS
    MLIRMtOpsIncGen
    # MLIRMtEnumsIncGen
    # MLIRMtAttrsIncGen

    LINK_COMPONENTS
    Core
)
