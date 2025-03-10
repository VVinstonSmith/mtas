
#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
func.func @not_inplace(
    %A : tensor<?x?xf32> {bufferization.writable = true})
  -> tensor<?x?xf32>
{
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tm = arith.constant 32 : i64
  %tn = arith.constant 32 : i64
  %tk = arith.constant 32 : i64
  %0 = index.casts %tk : i64 to index
  %1 = index.casts %tm : i64 to index
  %2 = index.casts %tn : i64 to index

  %f = linalg.fill ins(%f0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>
  %dim = tensor.dim %f, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %f, %c1 : tensor<?x?xf32>
  %5 = bufferization.alloc_tensor(%dim, %dim_0) : tensor<?x?xf32>
  %dim_1 = tensor.dim %f, %c1 : tensor<?x?xf32>

  %r = scf.for %arg10 = %c0 to %dim_1 step %0 iter_args(%arg11 = %5) -> (tensor<?x?xf32>) {
    %9 = affine.min #map(%arg10)[%0, %dim_1]
    %extracted_slice = tensor.extract_slice %f[0, %arg10] [%dim, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_2 = tensor.extract_slice %f[%arg10, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    // %extracted_slice_3 = tensor.extract_slice %f[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %10 = scf.for %arg12 = %c0 to %dim step %1 iter_args(%arg13 = %f) -> (tensor<?x?xf32>) {
      %11 = affine.min #map(%arg12)[%1, %dim]
      %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg12, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %12 = bufferization.alloc_tensor(%11, %9) : tensor<?x?xf32>
      %13 = linalg.copy ins(%extracted_slice_4 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice_6 = tensor.extract_slice %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %14 = scf.for %arg14 = %c0 to %dim_0 step %2 iter_args(%arg15 = %extracted_slice_6) -> (tensor<?x?xf32>) {
        %15 = affine.min #map(%arg14)[%2, %dim_0]
        %extracted_slice_9 = tensor.extract_slice %f[0, %arg14] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %16 = bufferization.alloc_tensor(%9, %15) : tensor<?x?xf32>
        %17 = linalg.copy ins(%extracted_slice_9 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %extracted_slice_10 = tensor.extract_slice %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %28 = linalg.matmul ins(%13, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_10 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %inserted_slice_21 = tensor.insert_slice %28 into %arg15[%arg14, 0] [%9, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice_21 : tensor<?x?xf32>
      }
      %inserted_slice_11 = tensor.insert_slice %14 into %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice_11 : tensor<?x?xf32>
    }
    %inserted_slice_9 = tensor.insert_slice %10 into %f[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    scf.yield %inserted_slice_9 : tensor<?x?xf32>
  }
  return %r: tensor<?x?xf32>
}



// #map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
// func.func @not_inplace(
//     %A : tensor<?x?xf32> {bufferization.writable = true})
//   -> tensor<?x?xf32>
// {
//   %f0 = arith.constant 0.0 : f32
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %tm = arith.constant 32 : i64
//   %tn = arith.constant 32 : i64
//   %tk = arith.constant 32 : i64
//   %0 = index.casts %tk : i64 to index
//   %1 = index.casts %tm : i64 to index
//   %2 = index.casts %tn : i64 to index

//   %f = linalg.fill ins(%f0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>
//   %dim = tensor.dim %f, %c0 : tensor<?x?xf32>
//   %dim_0 = tensor.dim %f, %c1 : tensor<?x?xf32>
//   %5 = bufferization.alloc_tensor(%dim, %dim_0) : tensor<?x?xf32>
//   %dim_1 = tensor.dim %f, %c1 : tensor<?x?xf32>

//   %r = scf.for %arg10 = %c0 to %dim_1 step %0 iter_args(%arg11 = %5) -> (tensor<?x?xf32>) {
//     %9 = affine.min #map(%arg10)[%0, %dim_1]
//     %extracted_slice = tensor.extract_slice %f[0, %arg10] [%dim, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     %extracted_slice_2 = tensor.extract_slice %f[%arg10, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     // %extracted_slice_3 = tensor.extract_slice %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     %10 = scf.for %arg12 = %c0 to %dim step %1 iter_args(%arg13 = %arg11) -> (tensor<?x?xf32>) {
//       %11 = affine.min #map(%arg12)[%1, %dim]
//       %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg12, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//       %12 = bufferization.alloc_tensor(%11, %9) : tensor<?x?xf32>
//       %13 = linalg.copy ins(%extracted_slice_4 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
//       %extracted_slice_6 = tensor.extract_slice %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//       %14 = scf.for %arg14 = %c0 to %dim_0 step %2 iter_args(%arg15 = %extracted_slice_6) -> (tensor<?x?xf32>) {
//         %15 = affine.min #map(%arg14)[%2, %dim_0]
//         %extracted_slice_9 = tensor.extract_slice %f[0, %arg14] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//         %16 = bufferization.alloc_tensor(%9, %15) : tensor<?x?xf32>
//         %17 = linalg.copy ins(%extracted_slice_9 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
//         %extracted_slice_10 = tensor.extract_slice %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//         %28 = linalg.matmul ins(%13, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_10 : tensor<?x?xf32>) -> tensor<?x?xf32>
//         %inserted_slice_21 = tensor.insert_slice %28 into %arg15[%arg14, 0] [%9, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//         scf.yield %inserted_slice_21 : tensor<?x?xf32>
//       }
//       %inserted_slice_11 = tensor.insert_slice %14 into %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//       scf.yield %inserted_slice_11 : tensor<?x?xf32>
//     }
//     %inserted_slice_9 = tensor.insert_slice %10 into %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//     scf.yield %inserted_slice_9 : tensor<?x?xf32>
//   }
//   return %r: tensor<?x?xf32>
// }

