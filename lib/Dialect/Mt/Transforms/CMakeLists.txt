add_mlir_dialect_library(MLIRMtTransforms
    AllocateAddressRegisters.cpp
    ConvertForOpIndexToI64.cpp
    AllocateScalarRegistersForConstants.cpp
    LowerFtmToMt.cpp
    GenerateConditionAndLabelsForLoops.cpp
    InstructionSchedulingAndPacking.cpp
    DependencyGraph.cpp
    DependencyAnalyzer.cpp
    ExecutionPacketUtils.cpp
    Reduce80BitInstructions.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Mt

    DEPENDS
    MLIRMtPassIncGen

    LINK_LIBS PUBLIC
    MLIRMtDialect
    MLIRTransformDialectTransforms
)