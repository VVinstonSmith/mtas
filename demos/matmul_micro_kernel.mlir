// module {
//   func.func @matmul_only(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
//   {
//     linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x?xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<?x3xvector<32xf32>> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x3xvector<32xf32>> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul ins(%A, %B : memref<6x?xf32>, memref<?x3xvector<32xf32>>) outs(%C : memref<6x3xvector<32xf32>>)
//     return
//   }
// }


// module {
//   func.func @matmul_only(
//     %A: memref<6x?xf32>,
//     %B: memref<?x?xf32>,
//     %C: memref<6x?xf32>,
//     %D: memref<?x?xf32>,
//     %E: memref<6x?xf32>)
//   {
    
//     linalg.matmul ins(%A, %B : memref<6x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
//     linalg.matmul ins(%C, %D : memref<6x?xf32>, memref<?x96xf32>) outs(%E : memref<6x96xf32>)
//     return
//   }
// }



module {
  func.func @matmul_only(
    %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
    %B: memref<512x64xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
    %C: memref<3x64xf32> {ftm.memory_level = #ftm.memory_level<am>})
  {
    linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>}
      ins(%A, %B : memref<3x512xf32>, memref<512x64xf32>) outs(%C : memref<3x64xf32>)
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
