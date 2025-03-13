add_mlir_dialect_library(MLIRFtmDialect
    FtmOps.cpp
    FtmDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Ftm

    DEPENDS
    MLIRFtmOpsIncGen
    MLIRFtmEnumsIncGen
    MLIRFtmAttrsIncGen

    LINK_COMPONENTS
    Core
)
