#ifndef MLIR_DIALECT_MT_IR_INSTRUCTIONSCHEDULINGINTERFACE
#define MLIR_DIALECT_MT_IR_INSTRUCTIONSCHEDULINGINTERFACE

include "mlir/IR/OpBase.td"

def InstructionSchedulingInterface : OpInterface<"InstructionSchedulingInterface"> {
  let description = "Comprehensive scheduling interface for instruction scheduling, performance analysis and resource allocation, providing functional unit assignment, latency calculation and register dependency tracking capabilities";

  let methods = [
    // Functional unit related methods
    InterfaceMethod<"Get all functional units that can execute this operation",
                    "llvm::SmallVector<mlir::mt::FunctionalUnit, 2>", "getFunctionalUnits">,
    InterfaceMethod<"Get the scheduling priority of this operation (higher value means higher priority)",
                    "int", "getSchedulingPriority", (ins), [{}], /*defaultImplementation=*/[{
      // Default implementation: priority is 1 (standard priority)
      return 1;
    }]>,
    
    // Instruction latency related methods
    InterfaceMethod<"Get the latency (in cycles) of this operation",
                    "int", "getLatency">,
    
    // Register read/write related methods
    InterfaceMethod<"Get all source registers read by this operation",
                    "llvm::SmallVector<mlir::Value, 3>", "getReadRegisters">,
    InterfaceMethod<"Get target register written by this operation",
                    "llvm::SmallVector<mlir::Value, 2>", "getWrittenRegisters">,

    // New method for formatted output of an operation
    InterfaceMethod<"Format and output this operation with scheduling information based on format options",
                    "std::string", "formatOperationOutput", (ins "StringRef":$functionalUnit)>
  ];
}

#endif // MLIR_DIALECT_MT_IR_INSTRUCTIONSCHEDULINGINTERFACE