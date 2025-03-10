
#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (-d0 + s2, s1, s0)>
module {
  func.func private @dma_wait_p2p(i32)
  func.func private @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
  func.func private @get_group_size() -> i32
  func.func private @get_thread_id() -> i32
  func.func @matmul_elemwise_tiling_pointerized(%arg0: i64 {mtfusion.func_arg_dim = #mtfusion.func_arg_dim<dim_M>}, %arg1: i64 {mtfusion.func_arg_dim = #mtfusion.func_arg_dim<dim_N>}, %arg2: i64 {mtfusion.func_arg_dim = #mtfusion.func_arg_dim<dim_K>}, %arg3: !llvm.ptr {mtfusion.func_arg_mat = #mtfusion.func_arg_mat<mat_A>}, %arg4: !llvm.ptr {mtfusion.func_arg_mat = #mtfusion.func_arg_mat<mat_B>}, %arg5: !llvm.ptr {mtfusion.func_arg_mat = #mtfusion.func_arg_mat<mat_C>}, %arg6: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<dim_K>, mtfusion.tiling_data}, %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<gsm>, mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.tiling_axis = #mtfusion.tiling_axis<dim_M>, mtfusion.tiling_data}, %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<am>, mtfusion.copy_mat = #mtfusion.copy_mat<mat_B>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<dim_N>, mtfusion.tiling_data}, %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<am>, mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.tiling_axis = #mtfusion.tiling_axis<dim_M>, mtfusion.tiling_data}, %arg10: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<sm>, mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.tiling_axis = #mtfusion.tiling_axis<dim_M>, mtfusion.tiling_data}) attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c14_i32 = arith.constant 14 : i32
    %c12_i32 = arith.constant 12 : i32
    %c10_i32 = arith.constant 10 : i32
    %c8_i32 = arith.constant 8 : i32
    %c6_i32 = arith.constant 6 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %0 = llvm.mlir.constant(4 : i32) : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3 = arith.constant {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} 3 : index
    %c2 = arith.constant {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<ddr>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<am>} : memref<3xi32>
    %alloca_0 = memref.alloca() {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<sm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<gsm>} : memref<2xi32>
    %alloca_1 = memref.alloca() {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>} : memref<3xi32>
    %alloca_2 = memref.alloca() {mtfusion.copy_mat = #mtfusion.copy_mat<mat_B>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>} : memref<2xi32>
    %alloca_3 = memref.alloca() {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<gsm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>} : memref<2xi32>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg3, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %c0_i64, %3[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg0, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg2, %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg2, %6[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %c1_i64, %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = builtin.unrealized_conversion_cast %8 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %10 = llvm.insertvalue %arg4, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg4, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %c0_i64, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg2, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %c1_i64, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %18 = llvm.insertvalue %arg5, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg5, %18[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %c0_i64, %19[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg0, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg1, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg1, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %c1_i64, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = builtin.unrealized_conversion_cast %24 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %26 = index.casts %arg6 {mtfusion.tiling_data} : i64 to index
    %27 = index.casts %arg7 {mtfusion.tiling_data} : i64 to index
    %alloca_4 = memref.alloca(%27, %26) : memref<2x?x?xf32>
    %28 = index.casts %arg8 {mtfusion.tiling_data} : i64 to index
    %alloca_5 = memref.alloca(%26, %28) : memref<2x?x?xf32>
    %29 = index.casts %arg9 {mtfusion.tiling_data} : i64 to index
    %alloca_6 = memref.alloca(%29, %28) : memref<3x?x?xf32>
    %30 = index.casts %arg10 {mtfusion.tiling_data} : i64 to index
    %alloca_7 = memref.alloca(%30, %26) : memref<2x?x?xf32>
    %dim = memref.dim %9, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_8 = memref.dim %9, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_9 = memref.dim %17, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %31 = call @get_thread_id() : () -> i32
    %32 = index.casts %31 : i32 to index
    %33 = call @get_group_size() : () -> i32
    %34 = index.casts %33 : i32 to index
    %35 = arith.muli %28, %34 : index
    scf.for %arg11 = %c0 to %dim_8 step %26 {
      %36 = affine.min #map(%arg11)[%26, %dim_8]
      %subview = memref.subview %9[0, %arg11] [%dim, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %17[%arg11, 0] [%36, %dim_9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %25[0, 0] [%dim, %dim_9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %37 = arith.divsi %c0, %27 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
      %38 = arith.remsi %37, %c2 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
      %39 = affine.min #map1()[%27, %dim]
      %subview_12 = memref.subview %subview[0, 0] [%39, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %40 = builtin.unrealized_conversion_cast %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %subview_13 = memref.subview %alloca_4[0, 0, 0] [2, %39, %36] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
      %subview_14 = memref.subview %subview_13[%38, 0, 0] [1, %39, %36] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %41 = builtin.unrealized_conversion_cast %subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %42 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %43 = llvm.extractvalue %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %44 = llvm.extractvalue %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %45 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %46 = llvm.trunc %44 : i64 to i32
      %47 = llvm.trunc %45 : i64 to i32
      %48 = llvm.mul %46, %0  : i32
      %49 = llvm.mul %47, %0  : i32
      %50 = llvm.sub %49, %48  : i32
      %51 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %52 = llvm.extractvalue %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %53 = llvm.extractvalue %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %54 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %55 = llvm.trunc %53 : i64 to i32
      %56 = llvm.trunc %54 : i64 to i32
      %57 = llvm.mul %55, %0  : i32
      %58 = llvm.mul %56, %0  : i32
      %59 = llvm.sub %58, %57  : i32
      %60 = func.call @dma_p2p_opt(%42, %43, %48, %50, %51, %52, %57, %59, %false, %c0_i32, %c2_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<gsm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
      memref.store %60, %alloca_3[%38] : memref<2xi32>
      %subview_15 = memref.subview %subview_11[0, 0] [%39, %dim_9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %61:3 = scf.for %arg12 = %c0 to %dim step %27 iter_args(%arg13 = %subview_12, %arg14 = %subview_14, %arg15 = %subview_15) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
        %62 = arith.addi %arg12, %27 : index
        %63 = arith.cmpi slt, %62, %dim : index
        %64:3 = scf.if %63 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
          %94 = arith.divsi %62, %27 : index
          %95 = arith.remsi %94, %c2 : index
          %96 = affine.min #map(%62)[%27, %dim]
          %subview_20 = memref.subview %subview[%62, 0] [%96, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %97 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_21 = memref.subview %alloca_4[0, 0, 0] [2, %96, %36] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_22 = memref.subview %subview_21[%95, 0, 0] [1, %96, %36] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %98 = builtin.unrealized_conversion_cast %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %99 = llvm.extractvalue %97[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %100 = llvm.extractvalue %97[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %101 = llvm.extractvalue %97[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %102 = llvm.extractvalue %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %103 = llvm.trunc %101 : i64 to i32
          %104 = llvm.trunc %102 : i64 to i32
          %105 = llvm.mul %103, %0  : i32
          %106 = llvm.mul %104, %0  : i32
          %107 = llvm.sub %106, %105  : i32
          %108 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %109 = llvm.extractvalue %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %110 = llvm.extractvalue %98[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %111 = llvm.extractvalue %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %112 = llvm.trunc %110 : i64 to i32
          %113 = llvm.trunc %111 : i64 to i32
          %114 = llvm.mul %112, %0  : i32
          %115 = llvm.mul %113, %0  : i32
          %116 = llvm.sub %115, %114  : i32
          %117 = func.call @dma_p2p_opt(%99, %100, %105, %107, %108, %109, %114, %116, %false, %c0_i32, %c4_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<gsm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %117, %alloca_3[%95] : memref<2xi32>
          %subview_23 = memref.subview %subview_11[%62, 0] [%96, %dim_9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          scf.yield %subview_20, %subview_22, %subview_23 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        } else {
          scf.yield %arg13, %arg14, %arg15 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        }
        %65 = affine.min #map(%arg12)[%27, %dim]
        %66 = arith.divsi %c0, %35 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
        %67 = arith.remsi %66, %c2 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
        %68 = affine.min #map2()[%28, %dim_9, %32]
        %subview_16 = memref.subview %subview_10[0, %32] [%36, %68] [1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %69 = builtin.unrealized_conversion_cast %subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %subview_17 = memref.subview %alloca_5[0, 0, 0] [2, %36, %68] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %subview_18 = memref.subview %subview_17[%67, 0, 0] [1, %36, %68] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %70 = builtin.unrealized_conversion_cast %subview_18 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %71 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %72 = llvm.extractvalue %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %73 = llvm.extractvalue %69[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %74 = llvm.extractvalue %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %75 = llvm.trunc %73 : i64 to i32
        %76 = llvm.trunc %74 : i64 to i32
        %77 = llvm.mul %75, %0  : i32
        %78 = llvm.mul %76, %0  : i32
        %79 = llvm.sub %78, %77  : i32
        %80 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %81 = llvm.extractvalue %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %82 = llvm.extractvalue %70[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %83 = llvm.extractvalue %70[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %84 = llvm.trunc %82 : i64 to i32
        %85 = llvm.trunc %83 : i64 to i32
        %86 = llvm.mul %84, %0  : i32
        %87 = llvm.mul %85, %0  : i32
        %88 = llvm.sub %87, %86  : i32
        %89 = func.call @dma_p2p_opt(%71, %72, %77, %79, %80, %81, %86, %88, %false, %c0_i32, %c6_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_B>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
        memref.store %89, %alloca_2[%67] : memref<2xi32>
        %subview_19 = memref.subview %arg15[0, %32] [%65, %68] [1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %90 = arith.divsi %arg12, %27 : index
        %91 = arith.remsi %90, %c2 : index
        %92 = memref.load %alloca_3[%91] : memref<2xi32>
        func.call @dma_wait_p2p(%92) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<gsm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
        %93:3 = scf.for %arg16 = %32 to %dim_9 step %35 iter_args(%arg17 = %subview_16, %arg18 = %subview_18, %arg19 = %subview_19) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
          %94 = arith.addi %arg16, %35 : index
          %95 = arith.cmpi slt, %94, %dim_9 : index
          %96:3 = scf.if %95 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
            %133 = arith.subi %94, %32 : index
            %134 = arith.divsi %133, %35 : index
            %135 = arith.remsi %134, %c2 : index
            %136 = affine.min #map(%94)[%28, %dim_9]
            %subview_24 = memref.subview %subview_10[0, %94] [%36, %136] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %137 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %subview_25 = memref.subview %alloca_5[0, 0, 0] [2, %36, %136] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_26 = memref.subview %subview_25[%135, 0, 0] [1, %36, %136] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %138 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %139 = llvm.extractvalue %137[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %140 = llvm.extractvalue %137[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %141 = llvm.extractvalue %137[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %142 = llvm.extractvalue %137[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %143 = llvm.trunc %141 : i64 to i32
            %144 = llvm.trunc %142 : i64 to i32
            %145 = llvm.mul %143, %0  : i32
            %146 = llvm.mul %144, %0  : i32
            %147 = llvm.sub %146, %145  : i32
            %148 = llvm.extractvalue %138[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %149 = llvm.extractvalue %138[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %150 = llvm.extractvalue %138[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %151 = llvm.extractvalue %138[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %152 = llvm.trunc %150 : i64 to i32
            %153 = llvm.trunc %151 : i64 to i32
            %154 = llvm.mul %152, %0  : i32
            %155 = llvm.mul %153, %0  : i32
            %156 = llvm.sub %155, %154  : i32
            %157 = func.call @dma_p2p_opt(%139, %140, %145, %147, %148, %149, %154, %156, %false, %c0_i32, %c8_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_B>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %157, %alloca_2[%135] : memref<2xi32>
            %subview_27 = memref.subview %arg15[0, %94] [%65, %136] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            scf.yield %subview_24, %subview_26, %subview_27 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
          } else {
            scf.yield %arg17, %arg18, %arg19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
          }
          %97 = affine.min #map(%arg16)[%28, %dim_9]
          %98 = arith.divsi %c0, %29 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
          %99 = arith.remsi %98, %c3 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
          %100 = affine.min #map3(%arg12)[%29, %27, %dim]
          %subview_20 = memref.subview %arg14[0, 0] [%100, %36] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_21 = memref.subview %arg19[0, 0] [%100, %97] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %101 = builtin.unrealized_conversion_cast %subview_21 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_22 = memref.subview %alloca_6[0, 0, 0] [3, %100, %97] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_23 = memref.subview %subview_22[%99, 0, 0] [1, %100, %97] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %102 = builtin.unrealized_conversion_cast %subview_23 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %103 = llvm.extractvalue %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %104 = llvm.extractvalue %101[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %105 = llvm.extractvalue %101[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %106 = llvm.extractvalue %101[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %107 = llvm.trunc %105 : i64 to i32
          %108 = llvm.trunc %106 : i64 to i32
          %109 = llvm.mul %107, %0  : i32
          %110 = llvm.mul %108, %0  : i32
          %111 = llvm.sub %110, %109  : i32
          %112 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %113 = llvm.extractvalue %102[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %114 = llvm.extractvalue %102[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %115 = llvm.extractvalue %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %116 = llvm.trunc %114 : i64 to i32
          %117 = llvm.trunc %115 : i64 to i32
          %118 = llvm.mul %116, %0  : i32
          %119 = llvm.mul %117, %0  : i32
          %120 = llvm.sub %119, %118  : i32
          %121 = func.call @dma_p2p_opt(%103, %104, %109, %111, %112, %113, %118, %120, %false, %c0_i32, %c10_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %121, %alloca_1[%99] : memref<3xi32>
          %122 = arith.subi %arg16, %32 : index
          %123 = arith.divsi %122, %35 : index
          %124 = arith.remsi %123, %c2 : index
          %125 = memref.load %alloca_2[%124] : memref<2xi32>
          func.call @dma_wait_p2p(%125) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_B>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          %126:3 = scf.for %arg20 = %c0 to %65 step %29 iter_args(%arg21 = %subview_20, %arg22 = %subview_21, %arg23 = %subview_23) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
            %133 = builtin.unrealized_conversion_cast %arg23 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %134 = builtin.unrealized_conversion_cast %arg22 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %135 = arith.addi %arg20, %29 : index
            %136 = arith.cmpi slt, %135, %65 : index
            %137:3 = scf.if %136 -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
              %187 = arith.divsi %135, %29 : index
              %188 = arith.remsi %187, %c3 : index
              %189 = affine.min #map(%135)[%29, %65]
              %subview_28 = memref.subview %arg14[%135, 0] [%189, %36] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %subview_29 = memref.subview %arg19[%135, 0] [%189, %97] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %190 = builtin.unrealized_conversion_cast %subview_29 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %subview_30 = memref.subview %alloca_6[0, 0, 0] [3, %189, %97] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
              %subview_31 = memref.subview %subview_30[%188, 0, 0] [1, %189, %97] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %191 = builtin.unrealized_conversion_cast %subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %192 = llvm.extractvalue %190[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %193 = llvm.extractvalue %190[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %194 = llvm.extractvalue %190[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %195 = llvm.extractvalue %190[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %196 = llvm.trunc %194 : i64 to i32
              %197 = llvm.trunc %195 : i64 to i32
              %198 = llvm.mul %196, %0  : i32
              %199 = llvm.mul %197, %0  : i32
              %200 = llvm.sub %199, %198  : i32
              %201 = llvm.extractvalue %191[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %202 = llvm.extractvalue %191[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %203 = llvm.extractvalue %191[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %204 = llvm.extractvalue %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %205 = llvm.trunc %203 : i64 to i32
              %206 = llvm.trunc %204 : i64 to i32
              %207 = llvm.mul %205, %0  : i32
              %208 = llvm.mul %206, %0  : i32
              %209 = llvm.sub %208, %207  : i32
              %210 = func.call @dma_p2p_opt(%192, %193, %198, %200, %201, %202, %207, %209, %false, %c0_i32, %c12_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
              memref.store %210, %alloca_1[%188] : memref<3xi32>
              scf.yield %subview_28, %subview_29, %subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            } else {
              scf.yield %arg21, %arg22, %arg23 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
            %138 = affine.min #map(%arg20)[%29, %65]
            %139 = arith.divsi %c0, %30 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
            %140 = arith.remsi %139, %c2 {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : index
            %141 = affine.min #map3(%arg20)[%30, %29, %65]
            %subview_24 = memref.subview %arg21[0, 0] [%141, %36] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %142 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %subview_25 = memref.subview %alloca_7[0, 0, 0] [2, %141, %36] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_26 = memref.subview %subview_25[%140, 0, 0] [1, %141, %36] [1, 1, 1] {mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>} : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %143 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %144 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %145 = llvm.extractvalue %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %146 = llvm.extractvalue %142[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %147 = llvm.extractvalue %142[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %148 = llvm.trunc %146 : i64 to i32
            %149 = llvm.trunc %147 : i64 to i32
            %150 = llvm.mul %148, %0  : i32
            %151 = llvm.mul %149, %0  : i32
            %152 = llvm.sub %151, %150  : i32
            %153 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %154 = llvm.extractvalue %143[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %155 = llvm.extractvalue %143[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %156 = llvm.extractvalue %143[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %157 = llvm.trunc %155 : i64 to i32
            %158 = llvm.trunc %156 : i64 to i32
            %159 = llvm.mul %157, %0  : i32
            %160 = llvm.mul %158, %0  : i32
            %161 = llvm.sub %160, %159  : i32
            %162 = func.call @dma_p2p_opt(%144, %145, %150, %152, %153, %154, %159, %161, %false, %c0_i32, %c0_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<sm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<gsm>, mtfusion.multi_stage = #mtfusion.multi_stage<prelogue>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %162, %alloca_0[%140] : memref<2xi32>
            %subview_27 = memref.subview %arg23[0, 0] [%141, %97] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %163 = arith.divsi %arg20, %29 : index
            %164 = arith.remsi %163, %c3 : index
            %165 = memref.load %alloca_1[%164] : memref<3xi32>
            func.call @dma_wait_p2p(%165) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<am>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<ddr>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            %166:3 = scf.for %arg24 = %c0 to %138 step %30 iter_args(%arg25 = %subview_24, %arg26 = %subview_26, %arg27 = %subview_27) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
              %187 = arith.addi %arg24, %30 : index
              %188 = arith.cmpi slt, %187, %138 : index
              %189:3 = scf.if %188 -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
                %193 = arith.divsi %187, %30 : index
                %194 = arith.remsi %193, %c2 : index
                %195 = affine.min #map(%187)[%30, %138]
                %subview_28 = memref.subview %arg21[%187, 0] [%195, %36] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %196 = builtin.unrealized_conversion_cast %subview_28 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
                %subview_29 = memref.subview %alloca_7[0, 0, 0] [2, %195, %36] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
                %subview_30 = memref.subview %subview_29[%194, 0, 0] [1, %195, %36] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %197 = builtin.unrealized_conversion_cast %subview_30 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
                %198 = llvm.extractvalue %196[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %199 = llvm.extractvalue %196[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %200 = llvm.extractvalue %196[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %201 = llvm.extractvalue %196[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %202 = llvm.trunc %200 : i64 to i32
                %203 = llvm.trunc %201 : i64 to i32
                %204 = llvm.mul %202, %0  : i32
                %205 = llvm.mul %203, %0  : i32
                %206 = llvm.sub %205, %204  : i32
                %207 = llvm.extractvalue %197[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %208 = llvm.extractvalue %197[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %209 = llvm.extractvalue %197[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %210 = llvm.extractvalue %197[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %211 = llvm.trunc %209 : i64 to i32
                %212 = llvm.trunc %210 : i64 to i32
                %213 = llvm.mul %211, %0  : i32
                %214 = llvm.mul %212, %0  : i32
                %215 = llvm.sub %214, %213  : i32
                %216 = func.call @dma_p2p_opt(%198, %199, %204, %206, %207, %208, %213, %215, %false, %c0_i32, %c0_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<sm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<gsm>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
                memref.store %216, %alloca_0[%194] : memref<2xi32>
                %subview_31 = memref.subview %arg23[%187, 0] [%195, %97] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                scf.yield %subview_28, %subview_30, %subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              } else {
                scf.yield %arg25, %arg26, %arg27 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              }
              %190 = arith.divsi %arg24, %30 : index
              %191 = arith.remsi %190, %c2 : index
              %192 = memref.load %alloca_0[%191] : memref<2xi32>
              func.call @dma_wait_p2p(%192) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_A>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<sm>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<gsm>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
              linalg.matmul ins(%arg26, %arg18 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%arg27 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
              scf.yield %189#0, %189#1, %189#2 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
            %167 = llvm.extractvalue %133[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %168 = llvm.extractvalue %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %169 = llvm.extractvalue %133[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %170 = llvm.extractvalue %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %171 = llvm.trunc %169 : i64 to i32
            %172 = llvm.trunc %170 : i64 to i32
            %173 = llvm.mul %171, %0  : i32
            %174 = llvm.mul %172, %0  : i32
            %175 = llvm.sub %174, %173  : i32
            %176 = llvm.extractvalue %134[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %177 = llvm.extractvalue %134[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %178 = llvm.extractvalue %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %179 = llvm.extractvalue %134[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %180 = llvm.trunc %178 : i64 to i32
            %181 = llvm.trunc %179 : i64 to i32
            %182 = llvm.mul %180, %0  : i32
            %183 = llvm.mul %181, %0  : i32
            %184 = llvm.sub %183, %182  : i32
            %185 = arith.cmpi ne, %arg20, %c0 : index
            scf.if %185 {
              %187 = arith.subi %arg20, %29 : index
              %188 = arith.divsi %187, %29 : index
              %189 = arith.remsi %188, %c3 : index
              %190 = memref.load %alloca[%189] : memref<3xi32>
              func.call @dma_wait_p2p(%190) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<ddr>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<am>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            }
            %186 = func.call @dma_p2p_opt(%167, %168, %173, %175, %176, %177, %182, %184, %false, %c0_i32, %c14_i32) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<ddr>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<am>, operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            scf.yield %137#0, %137#1, %137#2 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
          }
          %127 = arith.subi %29, %c1 : index
          %128 = arith.addi %65, %127 : index
          %129 = arith.divsi %128, %29 : index
          %130 = arith.subi %129, %c1 : index
          %131 = arith.remsi %130, %c3 : index
          %132 = memref.load %alloca[%131] : memref<3xi32>
          func.call @dma_wait_p2p(%132) {mtfusion.copy_mat = #mtfusion.copy_mat<mat_C>, mtfusion.dma_copy_dst = #mtfusion.dma_copy_dst<ddr>, mtfusion.dma_copy_src = #mtfusion.dma_copy_src<am>, operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          scf.yield %96#0, %96#1, %96#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        }
        scf.yield %64#0, %64#1, %64#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
      }
    } {__tiled_for__}
    return
  }
}
