add_mlir_conversion_library(MemRefToPtr
  MemRefToPtr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MTAS_SOURCE_DIR}/include/Conversion/MemRefToPtr

  DEPENDS
  MLIRConversionPassIncGen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFtmDialect
)

