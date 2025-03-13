add_mlir_dialect_library(MLIRFtmTransforms
    MultiBuffering.cpp
    TileDynamicDims.cpp
    LoopUnrolling.cpp
    SplitMatmul.cpp
    LowerMatmulToFma.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTAS_SOURCE_DIR}/include/mtas/Dialect/Ftm

    DEPENDS
    MLIRFtmPassIncGen

    LINK_LIBS PUBLIC
    MLIRTransformDialectTransforms
)