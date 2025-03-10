// module {
//   func.func @matmul_only(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
//   {
//     linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
//     return
//   }
// }


module {
  func.func @matmul_only(%A: memref<6x?xf32>, %B: memref<?x48xf32>, %C: memref<6x48xf32>)
  {
    linalg.matmul ins(%A, %B : memref<6x?xf32>, memref<?x48xf32>) outs(%C : memref<6x48xf32>)
    return
  }
}


// module {
//   func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %0 : tensor<?x?xf32>
//   }
// }

// module {
//   func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>}
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %0 : tensor<?x?xf32>
//   }
// }

// module {
//   func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>}
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %2 : tensor<?x?xf32>
//   }
// }
