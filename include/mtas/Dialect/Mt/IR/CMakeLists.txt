
add_mlir_dialect(MtOps mt)
add_mlir_doc(MtOps MtOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS InstructionSchedulingInterface.td)
mlir_tablegen(InstructionSchedulingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InstructionSchedulingInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRMtInstructionSchedulingInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS DoubleWordMemoryAccessInterface.td)
mlir_tablegen(DoubleWordMemoryAccessInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(DoubleWordMemoryAccessInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRMtDoubleWordMemoryAccessInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS MtEnums.td)
mlir_tablegen(MtEnums.h.inc -gen-enum-decls)
mlir_tablegen(MtEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMtEnumsIncGen)

# set(LLVM_TARGET_DEFINITIONS MtAttrs.td)
# mlir_tablegen(MtAttrs.h.inc -gen-attrdef-decls)
# mlir_tablegen(MtAttrs.cpp.inc -gen-attrdef-defs)
# add_public_tablegen_target(MLIRMtAttrsIncGen)



