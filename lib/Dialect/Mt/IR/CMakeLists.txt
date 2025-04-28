add_mlir_dialect_library(MLIRMtDialect
    MtOps.cpp
    MtDialect.cpp
    InstructionSchedulingInterface.cpp
    InstructionFormatter.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mt

    DEPENDS
    MLIRMtOpsIncGen
    MLIRMtInstructionSchedulingInterfaceIncGen
    MLIRMtEnumsIncGen
    # MLIRMtAttrsIncGen

    LINK_COMPONENTS
    Core
)
