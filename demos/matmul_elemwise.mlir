
// module {
//   func.func @matmul_elemwise(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>, %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>}
//   {
//     %0 = tensor.empty() : tensor<1024x1024xf32>
//     %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     %2 = tensor.empty() : tensor<1024x1024xf32>
//     %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     %5 = tensor.empty() : tensor<1024x1024xf32>
//     %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %arg3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     return %4 : tensor<1024x1024xf32>
//   }
// }

// module {
//   func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>}
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    
//     %c1 = arith.constant 1 : index
//     %c0 = arith.constant 0 : index
//     %dim_m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
//     %dim_k = tensor.dim %arg0, %c1 : tensor<?x?xf32>
//     %dim_n = tensor.dim %arg1, %c1 : tensor<?x?xf32>
//     %empty_tensor = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>

//     %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
//       ins(%0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%empty_tensor : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %3 : tensor<?x?xf32>
//   }
// }

module {
  func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
  attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>}
  {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}

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


// module {
//   func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>}
//   {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
    
//     %dim_m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
//     %dim_n = tensor.dim %arg1, %c1 : tensor<?x?xf32>

//     %0 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
//     %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    
//     %2 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
//     %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    
//     %5 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
//     %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %4 : tensor<?x?xf32>
//   }
// }

// module {
//   func.func @matmul_elemwise_0(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>, %arg3: tensor<1024x1024xf32>, %arg4: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
//     %0 = tensor.empty() : tensor<1024x1024xf32>
//     %1 = tensor.empty() : tensor<1024x1024xf32>
//     %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %arg3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%arg4 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     return %4 : tensor<1024x1024xf32>
//   }
//   func.func @matmul_elemwise(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>, %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
//     %0 = tensor.empty() : tensor<1024x1024xf32>
//     %1 = call @matmul_elemwise_0(%arg0, %arg1, %arg2, %arg3, %0) : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//     return %1 : tensor<1024x1024xf32>
//   }
// }
