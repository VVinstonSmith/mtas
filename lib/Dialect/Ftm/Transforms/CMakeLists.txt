add_mlir_dialect_library(MLIRFtmTransforms
    MultiBuffering.cpp
    TileLinalgDims.cpp
    LoopUnrolling.cpp
    LoopFolding.cpp
    SplitMatmul.cpp
    LowerKernelArguments.cpp
    LowerLinalgOps.cpp
    LowerLoadAndStoreMemRefToPtr.cpp
    CastPtrToInt64.cpp
    FoldRegisterAlloca.cpp
    AllocateOffsetRegisters.cpp
    LoopStrengthReduce.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Ftm

    DEPENDS
    MLIRFtmPassIncGen

    LINK_LIBS PUBLIC
    MLIRTransformDialectTransforms
)