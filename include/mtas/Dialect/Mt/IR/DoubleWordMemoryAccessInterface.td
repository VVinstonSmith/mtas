#ifndef MLIR_DIALECT_MT_IR_DOUBLEWORDMEMORYACCESSINTERFACE
#define MLIR_DIALECT_MT_IR_DOUBLEWORDMEMORYACCESSINTERFACE

include "mlir/IR/OpBase.td"

def DoubleWordMemoryAccessInterface : OpInterface<"DoubleWordMemoryAccessInterface"> {
  let description = "The interface is used to represent double word (dw) memory access operations, providing the function of obtaining high and low registers.";

  let methods = [
    // 获取高位寄存器
    InterfaceMethod<"Get the high register used by this operation",
                    "mlir::Value", "getHighRegister">,
    // 获取低位寄存器
    InterfaceMethod<"Get the low register used by this operation",
                    "mlir::Value", "getLowRegister">,
  ];
}

#endif // MLIR_DIALECT_MT_IR_DOUBLEWORDMEMORYACCESSINTERFACE