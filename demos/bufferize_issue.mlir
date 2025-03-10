#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @matmul_elemwise_0_tiling(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<K>, mtfusion.tiling_data}, %arg6: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<GSM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatB>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<N>, mtfusion.tiling_data}, %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatC>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<SM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}) -> tensor<?x?xf32> attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = index.casts %arg5 {mtfusion.tiling_data} : i64 to index
    %1 = index.casts %arg6 {mtfusion.tiling_data} : i64 to index
    %2 = index.casts %arg7 {mtfusion.tiling_data} : i64 to index
    %3 = index.casts %arg8 {mtfusion.tiling_data} : i64 to index
    %4 = index.casts %arg9 {mtfusion.tiling_data} : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %6 = scf.for %arg10 = %c0 to %dim_1 step %0 iter_args(%arg11 = %5) -> (tensor<?x?xf32>) {
      %9 = affine.min #map(%arg10)[%0, %dim_1]
      %extracted_slice = tensor.extract_slice %arg0[0, %arg10] [%dim, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg10, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %10 = scf.for %arg12 = %c0 to %dim step %1 iter_args(%arg13 = %extracted_slice_3) -> (tensor<?x?xf32>) {
        %11 = affine.min #map(%arg12)[%1, %dim]
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg12, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %12 = tensor.empty(%11, %9) : tensor<?x?xf32>
        %13 = linalg.copy {"DDR : GSM"} ins(%extracted_slice_4 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %extracted_slice_6 = tensor.extract_slice %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %14 = scf.for %arg14 = %c0 to %dim_0 step %2 iter_args(%arg15 = %extracted_slice_6) -> (tensor<?x?xf32>) {
          %15 = affine.min #map(%arg14)[%2, %dim_0]
          %extracted_slice_9 = tensor.extract_slice %arg1[0, %arg14] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %16 = tensor.empty(%9, %15) : tensor<?x?xf32>
          %17 = linalg.copy {"DDR : AM"} ins(%extracted_slice_9 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %extracted_slice_10 = tensor.extract_slice %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %18 = scf.for %arg16 = %c0 to %11 step %3 iter_args(%arg17 = %extracted_slice_10) -> (tensor<?x?xf32>) {
            %19 = affine.min #map(%arg16)[%3, %11]
            %extracted_slice_12 = tensor.extract_slice %13[%arg16, 0] [%19, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_14 = tensor.extract_slice %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %20 = tensor.empty(%19, %15) : tensor<?x?xf32>
            %21 = linalg.copy {"DDR : AM"} ins(%extracted_slice_14 : tensor<?x?xf32>) outs(%20 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %22 = scf.for %arg18 = %c0 to %19 step %4 iter_args(%arg19 = %21) -> (tensor<?x?xf32>) {
              %25 = affine.min #map(%arg18)[%4, %19]
              %extracted_slice_18 = tensor.extract_slice %extracted_slice_12[%arg18, 0] [%25, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = tensor.empty(%25, %9) : tensor<?x?xf32>
              %27 = linalg.copy {"GSM : SM"} ins(%extracted_slice_18 : tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %extracted_slice_20 = tensor.extract_slice %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %28 = linalg.matmul ins(%27, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_20 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %inserted_slice_21 = tensor.insert_slice %28 into %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %inserted_slice_21 : tensor<?x?xf32>
            } {__tiled_for___4}
            %dim_15 = tensor.dim %22, %c0 : tensor<?x?xf32>
            %dim_16 = tensor.dim %22, %c1 : tensor<?x?xf32>
            %23 = tensor.empty(%dim_15, %dim_16) : tensor<?x?xf32>
            %24 = linalg.copy {"AM : DDR"} ins(%22 : tensor<?x?xf32>) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %inserted_slice_17 = tensor.insert_slice %24 into %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_17 : tensor<?x?xf32>
          } {__tiled_for___3}
          %inserted_slice_11 = tensor.insert_slice %18 into %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_11 : tensor<?x?xf32>
        } {__tiled_for___2}
        %inserted_slice_7 = tensor.insert_slice %14 into %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice_7 : tensor<?x?xf32>
      } {__tiled_for___1}
      %inserted_slice = tensor.insert_slice %10 into %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice : tensor<?x?xf32>
    } {__tiled_for__}
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %8 : tensor<?x?xf32>
  }
  func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64 {mtfusion.tiling_data}, %arg5: i64 {mtfusion.tiling_data}, %arg6: i64 {mtfusion.tiling_data}, %arg7: i64 {mtfusion.tiling_data}, %arg8: i64 {mtfusion.tiling_data}) -> tensor<?x?xf32> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = call @matmul_elemwise_0_tiling(%arg0, %arg1, %arg2, %arg3, %0, %arg4, %arg5, %arg6, %arg7, %arg8) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, i64, i64, i64, i64, i64) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}