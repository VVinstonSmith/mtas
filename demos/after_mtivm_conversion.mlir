// #map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
// #map1 = affine_map<()[s0, s1] -> (s1, s0)>
// #map2 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0)>
// #map3 = affine_map<(d0)[s0, s1, s2] -> (-d0 + s2, s1, s0)>
// module {
//   func.func private @matmul_micro_kernel_r12c128(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)
//   func.func private @group_barrier(i32)
//   func.func private @dma_wait_p2p(i32)
//   func.func private @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//   func.func private @set_prir(i64)
//   func.func private @scalar_free(!llvm.ptr) -> i32
//   func.func private @scalar_malloc(i32) -> !llvm.ptr
//   func.func private @vector_free(!llvm.ptr) -> i32
//   func.func private @vector_malloc(i32) -> !llvm.ptr
//   llvm.mlir.global external @gsm_mem() {addr_space = 0 : i32} : !llvm.array<1572864 x f32>
//   func.func private @get_group_size() -> i32
//   func.func private @get_thread_id() -> i32
//   func.func @matmul_only_tiling_pointerized(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) {
//     %c6_i32 = arith.constant 6 : i32
//     %c2_i32 = arith.constant 2 : i32
//     %c8 = arith.constant 8 : index
//     %c6 = arith.constant 6 : index
//     %c4 = arith.constant 4 : index
//     %c0_i32 = arith.constant 0 : i32
//     %false = arith.constant false
//     %c4_i32 = arith.constant 4 : i32
//     %c12_i64 = arith.constant 12 : i64
//     %c2_i64 = arith.constant 2 : i64
//     %c8_i64 = arith.constant 8 : i64
//     %c0_i64 = arith.constant 0 : i64
//     %c1_i64 = arith.constant 1 : i64
//     %c1 = arith.constant 1 : index
//     %c0 = arith.constant 0 : index
//     %c2 = arith.constant 2 : index
//     %c3 = arith.constant 3 : index
//     %c3_i64 = arith.constant 3 : i64
//     %alloca = memref.alloca() : memref<2xi32>
//     %alloca_0 = memref.alloca() : memref<2xi32>
//     %alloca_1 = memref.alloca() : memref<2xi32>
//     %alloca_2 = memref.alloca() : memref<2xi32>
//     %alloca_3 = memref.alloca() : memref<2xi32>
//     call @set_prir(%c3_i64) : (i64) -> ()
//     %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//     %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %3 = llvm.insertvalue %c0_i64, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %4 = llvm.insertvalue %arg0, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %5 = llvm.insertvalue %arg2, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %6 = llvm.insertvalue %arg2, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %7 = llvm.insertvalue %c1_i64, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %9 = llvm.insertvalue %arg4, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %10 = llvm.insertvalue %arg4, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %11 = llvm.insertvalue %c0_i64, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %12 = llvm.insertvalue %arg2, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %13 = llvm.insertvalue %arg1, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %14 = llvm.insertvalue %arg1, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %15 = llvm.insertvalue %c1_i64, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %17 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %18 = llvm.insertvalue %arg5, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %19 = llvm.insertvalue %c0_i64, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %20 = llvm.insertvalue %arg0, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %21 = llvm.insertvalue %arg1, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %22 = llvm.insertvalue %arg1, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %23 = llvm.insertvalue %c1_i64, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//     %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %25 = index.casts %arg6 : i64 to index
//     %26 = index.casts %arg7 : i64 to index
//     %27 = llvm.mlir.addressof @gsm_mem : !llvm.ptr
//     %28 = llvm.getelementptr inbounds %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1572864 x f32>
//     %29 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//     %30 = llvm.insertvalue %28, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %32 = llvm.insertvalue %c0_i64, %31[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %33 = llvm.insertvalue %arg6, %32[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %34 = llvm.insertvalue %c1_i64, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %35 = llvm.insertvalue %arg7, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %36 = llvm.insertvalue %arg6, %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %37 = llvm.mul %arg6, %arg7  : i64
//     %38 = llvm.insertvalue %c2_i64, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %39 = llvm.insertvalue %37, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
//     %41 = index.casts %arg8 : i64 to index
//     %42 = llvm.mul %arg6, %c8_i64  : i64
//     %43 = llvm.mul %arg8, %42  : i64
//     %44 = llvm.trunc %43 : i64 to i32
//     %45 = call @vector_malloc(%44) : (i32) -> !llvm.ptr
//     %46 = llvm.insertvalue %45, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %47 = llvm.insertvalue %45, %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %48 = llvm.insertvalue %c0_i64, %47[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %49 = llvm.insertvalue %arg8, %48[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %50 = llvm.insertvalue %c1_i64, %49[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %51 = llvm.insertvalue %arg6, %50[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %52 = llvm.insertvalue %arg8, %51[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %53 = llvm.mul %arg8, %arg6  : i64
//     %54 = llvm.insertvalue %c2_i64, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %55 = llvm.insertvalue %53, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
//     %57 = index.casts %arg9 : i64 to index
//     %58 = llvm.mul %arg9, %c12_i64  : i64
//     %59 = llvm.mul %arg8, %58  : i64
//     %60 = llvm.trunc %59 : i64 to i32
//     %61 = call @vector_malloc(%60) : (i32) -> !llvm.ptr
//     %62 = llvm.insertvalue %61, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %63 = llvm.insertvalue %61, %62[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %64 = llvm.insertvalue %c0_i64, %63[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %65 = llvm.insertvalue %arg8, %64[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %66 = llvm.insertvalue %c1_i64, %65[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %67 = llvm.insertvalue %arg9, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %68 = llvm.insertvalue %arg8, %67[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %69 = llvm.mul %arg8, %arg9  : i64
//     %70 = llvm.insertvalue %c3_i64, %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %71 = llvm.insertvalue %69, %70[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x?x?xf32>
//     %73 = index.casts %arg10 : i64 to index
//     %74 = llvm.mul %arg10, %c8_i64  : i64
//     %75 = llvm.mul %arg6, %74  : i64
//     %76 = llvm.trunc %75 : i64 to i32
//     %77 = call @scalar_malloc(%76) : (i32) -> !llvm.ptr
//     %78 = llvm.insertvalue %77, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %79 = llvm.insertvalue %77, %78[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %80 = llvm.insertvalue %c0_i64, %79[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %81 = llvm.insertvalue %arg6, %80[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %82 = llvm.insertvalue %c1_i64, %81[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %83 = llvm.insertvalue %arg10, %82[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %84 = llvm.insertvalue %arg6, %83[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %85 = llvm.mul %arg6, %arg10  : i64
//     %86 = llvm.insertvalue %c2_i64, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %87 = llvm.insertvalue %85, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
//     %88 = builtin.unrealized_conversion_cast %87 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
//     %dim = memref.dim %8, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %dim_4 = memref.dim %8, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %dim_5 = memref.dim %16, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
//     %89 = call @get_thread_id() : () -> i32
//     %90 = index.casts %89 : i32 to index
//     %91 = arith.muli %41, %90 : index
//     %92 = call @get_group_size() : () -> i32
//     %93 = index.casts %92 : i32 to index
//     %94 = arith.muli %41, %93 : index
//     scf.for %arg11 = %c0 to %dim_4 step %25 {
//       %98 = affine.min #map(%arg11)[%25, %dim_4]
//       %subview = memref.subview %8[0, %arg11] [%dim, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//       %subview_6 = memref.subview %16[%arg11, 0] [%98, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//       %subview_7 = memref.subview %24[0, 0] [%dim, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//       %99 = arith.divsi %c0, %26 : index
//       %100 = arith.remsi %99, %c2 : index
//       %101 = affine.min #map1()[%26, %dim]
//       %subview_8 = memref.subview %subview[0, 0] [%101, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//       %102 = builtin.unrealized_conversion_cast %subview_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       %subview_9 = memref.subview %40[0, 0, 0] [2, %101, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//       %subview_10 = memref.subview %subview_9[%100, 0, 0] [1, %101, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//       %103 = builtin.unrealized_conversion_cast %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       %104 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %105 = llvm.extractvalue %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %106 = llvm.getelementptr inbounds %104[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       %107 = llvm.extractvalue %102[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %108 = llvm.extractvalue %102[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %109 = llvm.extractvalue %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %110 = llvm.trunc %108 : i64 to i32
//       %111 = llvm.trunc %109 : i64 to i32
//       %112 = llvm.mul %110, %c4_i32  : i32
//       %113 = llvm.mul %111, %c4_i32  : i32
//       %114 = llvm.sub %113, %112  : i32
//       %115 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %116 = llvm.extractvalue %103[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %117 = llvm.getelementptr inbounds %115[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       %118 = llvm.extractvalue %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %119 = llvm.extractvalue %103[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %120 = llvm.extractvalue %103[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//       %121 = llvm.trunc %119 : i64 to i32
//       %122 = llvm.trunc %120 : i64 to i32
//       %123 = llvm.mul %121, %c4_i32  : i32
//       %124 = llvm.mul %122, %c4_i32  : i32
//       %125 = llvm.sub %124, %123  : i32
//       %126 = func.call @dma_p2p_opt(%106, %107, %112, %114, %117, %118, %123, %125, %false, %c0_i32, %c2_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//       memref.store %126, %alloca_3[%c0] : memref<2xi32>
//       %subview_11 = memref.subview %subview_7[0, 0] [%101, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//       %127:3 = scf.for %arg12 = %c0 to %dim step %26 iter_args(%arg13 = %subview_8, %arg14 = %subview_10, %arg15 = %subview_11) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
//         %128 = arith.addi %arg12, %26 : index
//         %129 = arith.cmpi slt, %128, %dim : index
//         %130:3 = scf.if %129 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
//           %164 = arith.divsi %128, %26 : index
//           %165 = arith.remsi %164, %c2 : index
//           %166 = affine.min #map(%128)[%26, %dim]
//           %subview_16 = memref.subview %subview[%128, 0] [%166, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//           %167 = builtin.unrealized_conversion_cast %subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//           %subview_17 = memref.subview %40[0, 0, 0] [2, %166, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//           %subview_18 = memref.subview %subview_17[%165, 0, 0] [1, %166, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//           %168 = builtin.unrealized_conversion_cast %subview_18 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//           %169 = llvm.extractvalue %167[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %170 = llvm.extractvalue %167[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %171 = llvm.getelementptr inbounds %169[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//           %172 = llvm.extractvalue %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %173 = llvm.extractvalue %167[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %174 = llvm.extractvalue %167[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %175 = llvm.trunc %173 : i64 to i32
//           %176 = llvm.trunc %174 : i64 to i32
//           %177 = llvm.mul %175, %c4_i32  : i32
//           %178 = llvm.mul %176, %c4_i32  : i32
//           %179 = llvm.sub %178, %177  : i32
//           %180 = llvm.extractvalue %168[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %181 = llvm.extractvalue %168[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %182 = llvm.getelementptr inbounds %180[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//           %183 = llvm.extractvalue %168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %184 = llvm.extractvalue %168[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %185 = llvm.extractvalue %168[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %186 = llvm.trunc %184 : i64 to i32
//           %187 = llvm.trunc %185 : i64 to i32
//           %188 = llvm.mul %186, %c4_i32  : i32
//           %189 = llvm.mul %187, %c4_i32  : i32
//           %190 = llvm.sub %189, %188  : i32
//           %191 = arith.divsi %arg12, %26 : index
//           %192 = arith.addi %191, %c1 : index
//           %193 = arith.remsi %192, %c2 : index
//           %194 = arith.addi %193, %c2 : index
//           %195 = arith.index_castui %194 : index to i32
//           %196 = func.call @dma_p2p_opt(%171, %172, %177, %179, %182, %183, %188, %190, %false, %c0_i32, %195) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//           memref.store %196, %alloca_3[%193] : memref<2xi32>
//           %subview_19 = memref.subview %subview_7[%128, 0] [%166, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//           scf.yield %subview_16, %subview_18, %subview_19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//         } else {
//           scf.yield %arg13, %arg14, %arg15 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//         }
//         %131 = affine.min #map(%arg12)[%26, %dim]
//         %132 = arith.divsi %c0, %94 : index
//         %133 = arith.remsi %132, %c2 : index
//         %134 = affine.min #map2()[%41, %dim_5, %91]
//         %subview_12 = memref.subview %subview_6[0, %91] [%98, %134] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//         %135 = builtin.unrealized_conversion_cast %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//         %subview_13 = memref.subview %56[0, 0, 0] [2, %98, %134] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//         %subview_14 = memref.subview %subview_13[%133, 0, 0] [1, %98, %134] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//         %136 = builtin.unrealized_conversion_cast %subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//         %137 = llvm.extractvalue %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %138 = llvm.extractvalue %135[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %139 = llvm.getelementptr inbounds %137[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//         %140 = llvm.extractvalue %135[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %141 = llvm.extractvalue %135[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %142 = llvm.extractvalue %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %143 = llvm.trunc %141 : i64 to i32
//         %144 = llvm.trunc %142 : i64 to i32
//         %145 = llvm.mul %143, %c4_i32  : i32
//         %146 = llvm.mul %144, %c4_i32  : i32
//         %147 = llvm.sub %146, %145  : i32
//         %148 = llvm.extractvalue %136[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %149 = llvm.extractvalue %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %150 = llvm.getelementptr inbounds %148[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//         %151 = llvm.extractvalue %136[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %152 = llvm.extractvalue %136[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %153 = llvm.extractvalue %136[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//         %154 = llvm.trunc %152 : i64 to i32
//         %155 = llvm.trunc %153 : i64 to i32
//         %156 = llvm.mul %154, %c4_i32  : i32
//         %157 = llvm.mul %155, %c4_i32  : i32
//         %158 = llvm.sub %157, %156  : i32
//         %159 = func.call @dma_p2p_opt(%139, %140, %145, %147, %150, %151, %156, %158, %false, %c0_i32, %c4_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//         memref.store %159, %alloca_2[%c0] : memref<2xi32>
//         %subview_15 = memref.subview %arg15[0, %91] [%131, %134] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//         %160 = arith.divsi %arg12, %26 : index
//         %161 = arith.remsi %160, %c2 : index
//         %162 = memref.load %alloca_3[%161] : memref<2xi32>
//         func.call @dma_wait_p2p(%162) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//         func.call @group_barrier(%c0_i32) : (i32) -> ()
//         %163:3 = scf.for %arg16 = %91 to %dim_5 step %94 iter_args(%arg17 = %subview_12, %arg18 = %subview_14, %arg19 = %subview_15) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
//           %164 = builtin.unrealized_conversion_cast %arg18 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//           %165 = arith.addi %arg16, %94 : index
//           %166 = arith.cmpi slt, %165, %dim_5 : index
//           %167:3 = scf.if %166 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
//             %210 = arith.subi %165, %91 : index
//             %211 = arith.divsi %210, %94 : index
//             %212 = arith.remsi %211, %c2 : index
//             %213 = affine.min #map(%165)[%41, %dim_5]
//             %subview_20 = memref.subview %subview_6[0, %165] [%98, %213] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//             %214 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %subview_21 = memref.subview %56[0, 0, 0] [2, %98, %213] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//             %subview_22 = memref.subview %subview_21[%212, 0, 0] [1, %98, %213] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//             %215 = builtin.unrealized_conversion_cast %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %216 = llvm.extractvalue %214[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %217 = llvm.extractvalue %214[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %218 = llvm.getelementptr inbounds %216[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %219 = llvm.extractvalue %214[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %220 = llvm.extractvalue %214[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %221 = llvm.extractvalue %214[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %222 = llvm.trunc %220 : i64 to i32
//             %223 = llvm.trunc %221 : i64 to i32
//             %224 = llvm.mul %222, %c4_i32  : i32
//             %225 = llvm.mul %223, %c4_i32  : i32
//             %226 = llvm.sub %225, %224  : i32
//             %227 = llvm.extractvalue %215[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %228 = llvm.extractvalue %215[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %229 = llvm.getelementptr inbounds %227[%228] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %230 = llvm.extractvalue %215[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %231 = llvm.extractvalue %215[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %232 = llvm.extractvalue %215[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %233 = llvm.trunc %231 : i64 to i32
//             %234 = llvm.trunc %232 : i64 to i32
//             %235 = llvm.mul %233, %c4_i32  : i32
//             %236 = llvm.mul %234, %c4_i32  : i32
//             %237 = llvm.sub %236, %235  : i32
//             %238 = arith.subi %arg16, %91 : index
//             %239 = arith.divsi %238, %94 : index
//             %240 = arith.addi %239, %c1 : index
//             %241 = arith.remsi %240, %c2 : index
//             %242 = arith.addi %241, %c4 : index
//             %243 = arith.index_castui %242 : index to i32
//             %244 = func.call @dma_p2p_opt(%218, %219, %224, %226, %229, %230, %235, %237, %false, %c0_i32, %243) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//             memref.store %244, %alloca_2[%241] : memref<2xi32>
//             %subview_23 = memref.subview %arg15[0, %165] [%131, %213] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//             scf.yield %subview_20, %subview_22, %subview_23 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//           } else {
//             scf.yield %arg17, %arg18, %arg19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//           }
//           %168 = affine.min #map(%arg16)[%41, %dim_5]
//           %169 = arith.divsi %c0, %57 : index
//           %170 = arith.divsi %169, %c3 : index
//           %171 = arith.muli %170, %c3 : index
//           %172 = arith.subi %169, %171 : index
//           %173 = affine.min #map3(%arg12)[%57, %26, %dim]
//           %subview_16 = memref.subview %arg14[0, 0] [%173, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//           %subview_17 = memref.subview %arg19[0, 0] [%173, %168] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//           %174 = builtin.unrealized_conversion_cast %subview_17 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//           %subview_18 = memref.subview %72[0, 0, 0] [3, %173, %168] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
//           %subview_19 = memref.subview %subview_18[%172, 0, 0] [1, %173, %168] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//           %175 = builtin.unrealized_conversion_cast %subview_19 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//           %176 = llvm.extractvalue %174[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %177 = llvm.extractvalue %174[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %178 = llvm.getelementptr inbounds %176[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//           %179 = llvm.extractvalue %174[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %180 = llvm.extractvalue %174[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %181 = llvm.extractvalue %174[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %182 = llvm.trunc %180 : i64 to i32
//           %183 = llvm.trunc %181 : i64 to i32
//           %184 = llvm.mul %182, %c4_i32  : i32
//           %185 = llvm.mul %183, %c4_i32  : i32
//           %186 = llvm.sub %185, %184  : i32
//           %187 = llvm.extractvalue %175[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %188 = llvm.extractvalue %175[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %189 = llvm.getelementptr inbounds %187[%188] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//           %190 = llvm.extractvalue %175[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %191 = llvm.extractvalue %175[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %192 = llvm.extractvalue %175[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//           %193 = llvm.trunc %191 : i64 to i32
//           %194 = llvm.trunc %192 : i64 to i32
//           %195 = llvm.mul %193, %c4_i32  : i32
//           %196 = llvm.mul %194, %c4_i32  : i32
//           %197 = llvm.sub %196, %195  : i32
//           %198 = func.call @dma_p2p_opt(%178, %179, %184, %186, %189, %190, %195, %197, %false, %c0_i32, %c6_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//           memref.store %198, %alloca_1[%c0] : memref<2xi32>
//           %199 = arith.subi %arg16, %91 : index
//           %200 = arith.divsi %199, %94 : index
//           %201 = arith.remsi %200, %c2 : index
//           %202 = memref.load %alloca_2[%201] : memref<2xi32>
//           func.call @dma_wait_p2p(%202) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//           %203:3 = scf.for %arg20 = %c0 to %131 step %57 iter_args(%arg21 = %subview_16, %arg22 = %subview_17, %arg23 = %subview_19) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
//             %210 = builtin.unrealized_conversion_cast %arg23 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %211 = builtin.unrealized_conversion_cast %arg22 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %212 = arith.addi %arg20, %57 : index
//             %213 = arith.cmpi slt, %212, %131 : index
//             %214:3 = scf.if %213 -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
//               %274 = arith.divsi %212, %57 : index
//               %275 = arith.divsi %274, %c3 : index
//               %276 = arith.muli %275, %c3 : index
//               %277 = arith.subi %274, %276 : index
//               %278 = affine.min #map(%212)[%57, %131]
//               %subview_24 = memref.subview %arg14[%212, 0] [%278, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//               %subview_25 = memref.subview %arg19[%212, 0] [%278, %168] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
//               %279 = builtin.unrealized_conversion_cast %subview_25 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               %subview_26 = memref.subview %72[0, 0, 0] [3, %278, %168] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
//               %subview_27 = memref.subview %subview_26[%277, 0, 0] [1, %278, %168] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//               %280 = builtin.unrealized_conversion_cast %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               %281 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %282 = llvm.extractvalue %279[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %283 = llvm.getelementptr inbounds %281[%282] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               %284 = llvm.extractvalue %279[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %285 = llvm.extractvalue %279[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %286 = llvm.extractvalue %279[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %287 = llvm.trunc %285 : i64 to i32
//               %288 = llvm.trunc %286 : i64 to i32
//               %289 = llvm.mul %287, %c4_i32  : i32
//               %290 = llvm.mul %288, %c4_i32  : i32
//               %291 = llvm.sub %290, %289  : i32
//               %292 = llvm.extractvalue %280[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %293 = llvm.extractvalue %280[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %294 = llvm.getelementptr inbounds %292[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               %295 = llvm.extractvalue %280[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %296 = llvm.extractvalue %280[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %297 = llvm.extractvalue %280[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %298 = llvm.trunc %296 : i64 to i32
//               %299 = llvm.trunc %297 : i64 to i32
//               %300 = llvm.mul %298, %c4_i32  : i32
//               %301 = llvm.mul %299, %c4_i32  : i32
//               %302 = llvm.sub %301, %300  : i32
//               %303 = arith.divsi %arg20, %57 : index
//               %304 = arith.addi %303, %c1 : index
//               %305 = arith.remsi %304, %c2 : index
//               %306 = arith.addi %305, %c6 : index
//               %307 = arith.index_castui %306 : index to i32
//               %308 = func.call @dma_p2p_opt(%283, %284, %289, %291, %294, %295, %300, %302, %false, %c0_i32, %307) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//               memref.store %308, %alloca_1[%305] : memref<2xi32>
//               scf.yield %subview_24, %subview_25, %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
//             } else {
//               scf.yield %arg21, %arg22, %arg23 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
//             }
//             %215 = affine.min #map(%arg20)[%57, %131]
//             %216 = arith.divsi %c0, %73 : index
//             %217 = arith.remsi %216, %c2 : index
//             // %218 = affine.min #map3(%arg20)[%73, %57, %131]
//             %subview_20 = memref.subview %arg21[0, 0] [%73, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//             %219 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %subview_21 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//             %subview_22 = memref.subview %subview_21[%217, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//             %220 = builtin.unrealized_conversion_cast %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//             %221 = llvm.extractvalue %219[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %222 = llvm.extractvalue %219[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %223 = llvm.getelementptr inbounds %221[%222] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %224 = llvm.extractvalue %219[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %225 = llvm.extractvalue %219[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %226 = llvm.extractvalue %219[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %227 = llvm.trunc %225 : i64 to i32
//             %228 = llvm.trunc %226 : i64 to i32
//             %229 = llvm.mul %227, %c4_i32  : i32
//             %230 = llvm.mul %228, %c4_i32  : i32
//             %231 = llvm.sub %230, %229  : i32
//             %232 = llvm.extractvalue %220[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %233 = llvm.extractvalue %220[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %234 = llvm.getelementptr inbounds %232[%233] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %235 = llvm.extractvalue %220[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %236 = llvm.extractvalue %220[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %237 = llvm.extractvalue %220[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %238 = llvm.trunc %236 : i64 to i32
//             %239 = llvm.trunc %237 : i64 to i32
//             %240 = llvm.mul %238, %c4_i32  : i32
//             %241 = llvm.mul %239, %c4_i32  : i32
//             %242 = llvm.sub %241, %240  : i32
//             %243 = func.call @dma_p2p_opt(%223, %224, %229, %231, %234, %235, %240, %242, %false, %c0_i32, %c0_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//             memref.store %243, %alloca_0[%c0] : memref<2xi32>
//             %subview_23 = memref.subview %arg23[0, 0] [%73, %168] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//             %244 = arith.divsi %arg20, %57 : index
//             %245 = arith.remsi %244, %c2 : index
//             %246 = memref.load %alloca_1[%245] : memref<2xi32>
//             func.call @dma_wait_p2p(%246) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//             scf.for %arg24 = %c0 to %215 step %73 {
//               // %274 = builtin.unrealized_conversion_cast %arg26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               // %275 = builtin.unrealized_conversion_cast %arg27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               %276 = arith.addi %arg24, %73 : index
//               %277 = arith.cmpi slt, %276, %215 : index
//               scf.if %277 {
//                 %294 = arith.divsi %276, %73 : index
//                 %295 = arith.remsi %294, %c2 : index
//                 %subview_24 = memref.subview %arg21[%276, 0] [%73, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//                 %297 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//                 %subview_25 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//                 %subview_26 = memref.subview %subview_25[%295, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//                 %298 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//                 %299 = llvm.extractvalue %297[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %300 = llvm.extractvalue %297[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %301 = llvm.getelementptr inbounds %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//                 %302 = llvm.extractvalue %297[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %303 = llvm.extractvalue %297[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %304 = llvm.extractvalue %297[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %305 = llvm.trunc %303 : i64 to i32
//                 %306 = llvm.trunc %304 : i64 to i32
//                 %307 = llvm.mul %305, %c4_i32  : i32
//                 %308 = llvm.mul %306, %c4_i32  : i32
//                 %309 = llvm.sub %308, %307  : i32
//                 %310 = llvm.extractvalue %298[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %311 = llvm.extractvalue %298[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %312 = llvm.getelementptr inbounds %310[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//                 %313 = llvm.extractvalue %298[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %314 = llvm.extractvalue %298[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %315 = llvm.extractvalue %298[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//                 %316 = llvm.trunc %314 : i64 to i32
//                 %317 = llvm.trunc %315 : i64 to i32
//                 %318 = llvm.mul %316, %c4_i32  : i32
//                 %319 = llvm.mul %317, %c4_i32  : i32
//                 %320 = llvm.sub %319, %318  : i32
//                 %321 = arith.divsi %arg24, %73 : index
//                 %322 = arith.addi %321, %c1 : index
//                 %323 = arith.remsi %322, %c2 : index
//                 %324 = arith.index_castui %323 : index to i32
//                 %325 = func.call @dma_p2p_opt(%301, %302, %307, %309, %312, %313, %318, %320, %false, %c0_i32, %324) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//                 memref.store %325, %alloca_0[%323] : memref<2xi32>
//                 // %subview_27 = memref.subview %arg23[%276, 0] [%73, %168] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//               }

              
//               // new added
//               %294 = arith.divsi %arg24, %73 : index
//               %295 = arith.remsi %294, %c2 : index
//               // %subview_24 = memref.subview %arg21[%arg24, 0] [%73, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//               // %297 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               %subview_25 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
//               %subview_26 = memref.subview %subview_25[%295, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//               // %298 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               // %299 = llvm.extractvalue %297[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %300 = llvm.extractvalue %297[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %301 = llvm.getelementptr inbounds %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               // %302 = llvm.extractvalue %297[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %303 = llvm.extractvalue %297[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %304 = llvm.extractvalue %297[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %305 = llvm.trunc %303 : i64 to i32
//               // %306 = llvm.trunc %304 : i64 to i32
//               // %307 = llvm.mul %305, %c4_i32  : i32
//               // %308 = llvm.mul %306, %c4_i32  : i32
//               // %309 = llvm.sub %308, %307  : i32
//               // %310 = llvm.extractvalue %298[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %311 = llvm.extractvalue %298[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %312 = llvm.getelementptr inbounds %310[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               // %313 = llvm.extractvalue %298[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %314 = llvm.extractvalue %298[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %315 = llvm.extractvalue %298[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               // %316 = llvm.trunc %314 : i64 to i32
//               // %317 = llvm.trunc %315 : i64 to i32
//               // %318 = llvm.mul %316, %c4_i32  : i32
//               // %319 = llvm.mul %317, %c4_i32  : i32
//               // %320 = llvm.sub %319, %318  : i32
//               // %321 = arith.divsi %arg24, %73 : index
//               // %322 = arith.addi %321, %c1 : index
//               // %323 = arith.remsi %322, %c2 : index
//               // %324 = arith.index_castui %323 : index to i32
//               // %325 = func.call @dma_p2p_opt(%301, %302, %307, %309, %312, %313, %318, %320, %false, %c0_i32, %324) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//               // memref.store %325, %alloca_0[%323] : memref<2xi32>
//               %subview_27 = memref.subview %arg23[%arg24, 0] [%73, %168] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>

//               %274 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//               %275 = builtin.unrealized_conversion_cast %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

//               %279 = arith.divsi %arg24, %73 : index
//               %280 = arith.remsi %279, %c2 : index
//               %281 = memref.load %alloca_0[%280] : memref<2xi32>
//               func.call @dma_wait_p2p(%281) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//               %282 = llvm.extractvalue %274[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %283 = llvm.extractvalue %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %284 = llvm.extractvalue %275[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %285 = llvm.extractvalue %274[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %286 = llvm.extractvalue %164[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %287 = llvm.extractvalue %275[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %288 = llvm.getelementptr inbounds %282[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               %289 = llvm.getelementptr inbounds %283[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               %290 = llvm.getelementptr inbounds %284[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//               %291 = llvm.extractvalue %274[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//               %292 = arith.index_castui %25 : index to i64
//               %293 = arith.index_castui %41 : index to i64
//               func.call @matmul_micro_kernel_r12c128(%288, %289, %290, %291, %292, %293) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()

//             } {__tiled_for___4}
//             %248 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %249 = llvm.extractvalue %210[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %250 = llvm.getelementptr inbounds %248[%249] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %251 = llvm.extractvalue %210[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %252 = llvm.extractvalue %210[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %253 = llvm.extractvalue %210[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %254 = llvm.trunc %252 : i64 to i32
//             %255 = llvm.trunc %253 : i64 to i32
//             %256 = llvm.mul %254, %c4_i32  : i32
//             %257 = llvm.mul %255, %c4_i32  : i32
//             %258 = llvm.sub %257, %256  : i32
//             %259 = llvm.extractvalue %211[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %260 = llvm.extractvalue %211[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %261 = llvm.getelementptr inbounds %259[%260] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//             %262 = llvm.extractvalue %211[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %263 = llvm.extractvalue %211[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %264 = llvm.extractvalue %211[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
//             %265 = llvm.trunc %263 : i64 to i32
//             %266 = llvm.trunc %264 : i64 to i32
//             %267 = llvm.mul %265, %c4_i32  : i32
//             %268 = llvm.mul %266, %c4_i32  : i32
//             %269 = llvm.sub %268, %267  : i32
//             %270 = arith.addi %245, %c8 : index
//             %271 = arith.index_castui %270 : index to i32
            
//             %273 = arith.cmpi ne, %arg20, %c0 : index
//             scf.if %273 {
//               %274 = arith.subi %arg20, %57 : index
//               %275 = arith.divsi %274, %57 : index
//               %276 = arith.remsi %275, %c2 : index
//               %277 = memref.load %alloca[%276] : memref<2xi32>
//               func.call @dma_wait_p2p(%277) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//             }

//             %272 = func.call @dma_p2p_opt(%250, %251, %256, %258, %261, %262, %267, %269, %false, %c0_i32, %271) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
//             memref.store %272, %alloca[%245] : memref<2xi32>
            
//             scf.yield %214#0, %214#1, %214#2 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
//           } {__tiled_for___3}
//           %204 = arith.subi %57, %c1 : index
//           %205 = arith.addi %131, %204 : index
//           %206 = arith.divsi %205, %57 : index
//           %207 = arith.subi %206, %c1 : index
//           %208 = arith.remsi %207, %c2 : index
//           %209 = memref.load %alloca[%208] : memref<2xi32>
//           func.call @dma_wait_p2p(%209) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
//           scf.yield %167#0, %167#1, %167#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//         } {__tiled_for___2}
//         func.call @group_barrier(%c0_i32) : (i32) -> ()
//         scf.yield %130#0, %130#1, %130#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
//       } {__tiled_for___1}
//     } {__tiled_for__}
//     %95 = call @vector_free(%45) : (!llvm.ptr) -> i32
//     %96 = call @vector_free(%61) : (!llvm.ptr) -> i32
//     %97 = call @scalar_free(%77) : (!llvm.ptr) -> i32
//     return
//   }
// }



#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (-d0 + s2, s1, s0)>
module {
  func.func private @matmul_micro_kernel_r12c128(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)
  func.func private @group_barrier(i32)
  func.func private @dma_wait_p2p(i32)
  func.func private @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
  func.func private @set_prir(i64)
  func.func private @scalar_free(!llvm.ptr) -> i32
  func.func private @scalar_malloc(i32) -> !llvm.ptr
  func.func private @vector_free(!llvm.ptr) -> i32
  func.func private @vector_malloc(i32) -> !llvm.ptr
  llvm.mlir.global external @gsm_mem() {addr_space = 0 : i32} : !llvm.array<1572864 x f32>
  func.func private @get_group_size() -> i32
  func.func private @get_thread_id() -> i32
  func.func @matmul_only_tiling_pointerized(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) {
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c4_i32 = arith.constant 4 : i32
    %c12_i64 = arith.constant 12 : i64
    %c2_i64 = arith.constant 2 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c3_i64 = arith.constant 3 : i64
    %alloca = memref.alloca() : memref<2xi32>
    %alloca_0 = memref.alloca() : memref<2xi32>
    %alloca_1 = memref.alloca() : memref<2xi32>
    %alloca_2 = memref.alloca() : memref<2xi32>
    %alloca_3 = memref.alloca() : memref<2xi32>
    call @set_prir(%c3_i64) : (i64) -> ()
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %c0_i64, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg0, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg2, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg2, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %c1_i64, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %9 = llvm.insertvalue %arg4, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg4, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %c0_i64, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg1, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %c1_i64, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %17 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg5, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %c0_i64, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg0, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg1, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg1, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %c1_i64, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %25 = index.casts %arg6 : i64 to index
    %26 = index.casts %arg7 : i64 to index
    %27 = llvm.mlir.addressof @gsm_mem : !llvm.ptr
    %28 = llvm.getelementptr inbounds %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1572864 x f32>
    %29 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %30 = llvm.insertvalue %28, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %c0_i64, %31[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.insertvalue %arg6, %32[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.insertvalue %c1_i64, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.insertvalue %arg7, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.insertvalue %arg6, %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.mul %arg6, %arg7  : i64
    %38 = llvm.insertvalue %c2_i64, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.insertvalue %37, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %41 = index.casts %arg8 : i64 to index
    %42 = llvm.mul %arg6, %c8_i64  : i64
    %43 = llvm.mul %arg8, %42  : i64
    %44 = llvm.trunc %43 : i64 to i32
    %45 = call @vector_malloc(%44) : (i32) -> !llvm.ptr
    %46 = llvm.insertvalue %45, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = llvm.insertvalue %45, %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.insertvalue %c0_i64, %47[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.insertvalue %arg8, %48[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.insertvalue %c1_i64, %49[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %arg6, %50[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.insertvalue %arg8, %51[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.mul %arg8, %arg6  : i64
    %54 = llvm.insertvalue %c2_i64, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %53, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %57 = index.casts %arg9 : i64 to index
    %58 = llvm.mul %arg9, %c12_i64  : i64
    %59 = llvm.mul %arg8, %58  : i64
    %60 = llvm.trunc %59 : i64 to i32
    %61 = call @vector_malloc(%60) : (i32) -> !llvm.ptr
    %62 = llvm.insertvalue %61, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.insertvalue %61, %62[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %c0_i64, %63[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %arg8, %64[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %c1_i64, %65[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %arg9, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %arg8, %67[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.mul %arg8, %arg9  : i64
    %70 = llvm.insertvalue %c3_i64, %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.insertvalue %69, %70[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x?x?xf32>
    %73 = index.casts %arg10 : i64 to index
    %74 = llvm.mul %arg10, %c8_i64  : i64
    %75 = llvm.mul %arg6, %74  : i64
    %76 = llvm.trunc %75 : i64 to i32
    %77 = call @scalar_malloc(%76) : (i32) -> !llvm.ptr
    %78 = llvm.insertvalue %77, %29[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %79 = llvm.insertvalue %77, %78[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %80 = llvm.insertvalue %c0_i64, %79[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %81 = llvm.insertvalue %arg6, %80[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %82 = llvm.insertvalue %c1_i64, %81[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %83 = llvm.insertvalue %arg10, %82[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %84 = llvm.insertvalue %arg6, %83[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.mul %arg6, %arg10  : i64
    %86 = llvm.insertvalue %c2_i64, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %87 = llvm.insertvalue %85, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %88 = builtin.unrealized_conversion_cast %87 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %dim = memref.dim %8, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_4 = memref.dim %8, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_5 = memref.dim %16, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %89 = call @get_thread_id() : () -> i32
    %90 = index.casts %89 : i32 to index
    %91 = arith.muli %41, %90 : index
    %92 = call @get_group_size() : () -> i32
    %93 = index.casts %92 : i32 to index
    %94 = arith.muli %41, %93 : index
    scf.for %arg11 = %c0 to %dim_4 step %25 {
      %98 = affine.min #map(%arg11)[%25, %dim_4]
      %subview = memref.subview %8[0, %arg11] [%dim, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %16[%arg11, 0] [%98, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_7 = memref.subview %24[0, 0] [%dim, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %99 = arith.divsi %c0, %26 : index
      %100 = arith.remsi %99, %c2 : index
      %101 = affine.min #map1()[%26, %dim]
      %subview_8 = memref.subview %subview[0, 0] [%101, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %102 = builtin.unrealized_conversion_cast %subview_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %subview_9 = memref.subview %40[0, 0, 0] [2, %101, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
      %subview_10 = memref.subview %subview_9[%100, 0, 0] [1, %101, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %103 = builtin.unrealized_conversion_cast %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %104 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %105 = llvm.extractvalue %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %106 = llvm.getelementptr inbounds %104[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %107 = llvm.extractvalue %102[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %108 = llvm.extractvalue %102[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %109 = llvm.extractvalue %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %110 = llvm.trunc %108 : i64 to i32
      %111 = llvm.trunc %109 : i64 to i32
      %112 = llvm.mul %110, %c4_i32  : i32
      %113 = llvm.mul %111, %c4_i32  : i32
      %114 = llvm.sub %113, %112  : i32
      %115 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %116 = llvm.extractvalue %103[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %117 = llvm.getelementptr inbounds %115[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %118 = llvm.extractvalue %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %119 = llvm.extractvalue %103[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %120 = llvm.extractvalue %103[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %121 = llvm.trunc %119 : i64 to i32
      %122 = llvm.trunc %120 : i64 to i32
      %123 = llvm.mul %121, %c4_i32  : i32
      %124 = llvm.mul %122, %c4_i32  : i32
      %125 = llvm.sub %124, %123  : i32
      %126 = func.call @dma_p2p_opt(%106, %107, %112, %114, %117, %118, %123, %125, %false, %c0_i32, %c2_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
      memref.store %126, %alloca_3[%c0] : memref<2xi32>
      %subview_11 = memref.subview %subview_7[0, 0] [%101, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %127:3 = scf.for %arg12 = %c0 to %dim step %26 iter_args(%arg13 = %subview_8, %arg14 = %subview_10, %arg15 = %subview_11) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
        %128 = arith.addi %arg12, %26 : index
        %129 = arith.cmpi slt, %128, %dim : index
        %130:3 = scf.if %129 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
          %164 = arith.divsi %128, %26 : index
          %165 = arith.remsi %164, %c2 : index
          %166 = affine.min #map(%128)[%26, %dim]
          %subview_16 = memref.subview %subview[%128, 0] [%166, %98] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %167 = builtin.unrealized_conversion_cast %subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_17 = memref.subview %40[0, 0, 0] [2, %166, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_18 = memref.subview %subview_17[%165, 0, 0] [1, %166, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %168 = builtin.unrealized_conversion_cast %subview_18 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %169 = llvm.extractvalue %167[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %170 = llvm.extractvalue %167[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %171 = llvm.getelementptr inbounds %169[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %172 = llvm.extractvalue %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %173 = llvm.extractvalue %167[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %174 = llvm.extractvalue %167[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %175 = llvm.trunc %173 : i64 to i32
          %176 = llvm.trunc %174 : i64 to i32
          %177 = llvm.mul %175, %c4_i32  : i32
          %178 = llvm.mul %176, %c4_i32  : i32
          %179 = llvm.sub %178, %177  : i32
          %180 = llvm.extractvalue %168[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %181 = llvm.extractvalue %168[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %182 = llvm.getelementptr inbounds %180[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %183 = llvm.extractvalue %168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %184 = llvm.extractvalue %168[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %185 = llvm.extractvalue %168[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %186 = llvm.trunc %184 : i64 to i32
          %187 = llvm.trunc %185 : i64 to i32
          %188 = llvm.mul %186, %c4_i32  : i32
          %189 = llvm.mul %187, %c4_i32  : i32
          %190 = llvm.sub %189, %188  : i32
          %191 = arith.divsi %arg12, %26 : index
          %192 = arith.addi %191, %c1 : index
          %193 = arith.remsi %192, %c2 : index
          %194 = arith.addi %193, %c2 : index
          %195 = arith.index_castui %194 : index to i32
          %196 = func.call @dma_p2p_opt(%171, %172, %177, %179, %182, %183, %188, %190, %false, %c0_i32, %195) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %196, %alloca_3[%193] : memref<2xi32>
          %subview_19 = memref.subview %subview_7[%128, 0] [%166, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          scf.yield %subview_16, %subview_18, %subview_19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        } else {
          scf.yield %arg13, %arg14, %arg15 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        }
        %131 = affine.min #map(%arg12)[%26, %dim]
        %132 = arith.divsi %c0, %94 : index
        %133 = arith.remsi %132, %c2 : index
        %134 = affine.min #map2()[%41, %dim_5, %91]
        %subview_12 = memref.subview %subview_6[0, %91] [%98, %134] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %135 = builtin.unrealized_conversion_cast %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %subview_13 = memref.subview %56[0, 0, 0] [2, %98, %134] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %subview_14 = memref.subview %subview_13[%133, 0, 0] [1, %98, %134] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %136 = builtin.unrealized_conversion_cast %subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %137 = llvm.extractvalue %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %138 = llvm.extractvalue %135[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %139 = llvm.getelementptr inbounds %137[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %140 = llvm.extractvalue %135[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %141 = llvm.extractvalue %135[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %142 = llvm.extractvalue %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %143 = llvm.trunc %141 : i64 to i32
        %144 = llvm.trunc %142 : i64 to i32
        %145 = llvm.mul %143, %c4_i32  : i32
        %146 = llvm.mul %144, %c4_i32  : i32
        %147 = llvm.sub %146, %145  : i32
        %148 = llvm.extractvalue %136[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %149 = llvm.extractvalue %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %150 = llvm.getelementptr inbounds %148[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %151 = llvm.extractvalue %136[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %152 = llvm.extractvalue %136[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %153 = llvm.extractvalue %136[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %154 = llvm.trunc %152 : i64 to i32
        %155 = llvm.trunc %153 : i64 to i32
        %156 = llvm.mul %154, %c4_i32  : i32
        %157 = llvm.mul %155, %c4_i32  : i32
        %158 = llvm.sub %157, %156  : i32
        %159 = func.call @dma_p2p_opt(%139, %140, %145, %147, %150, %151, %156, %158, %false, %c0_i32, %c4_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
        memref.store %159, %alloca_2[%c0] : memref<2xi32>
        %subview_15 = memref.subview %arg15[0, %91] [%131, %134] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %160 = arith.divsi %arg12, %26 : index
        %161 = arith.remsi %160, %c2 : index
        %162 = memref.load %alloca_3[%161] : memref<2xi32>
        func.call @dma_wait_p2p(%162) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
        func.call @group_barrier(%c0_i32) : (i32) -> ()
        %163:3 = scf.for %arg16 = %91 to %dim_5 step %94 iter_args(%arg17 = %subview_12, %arg18 = %subview_14, %arg19 = %subview_15) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
          %164 = builtin.unrealized_conversion_cast %arg18 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %165 = arith.addi %arg16, %94 : index
          %166 = arith.cmpi slt, %165, %dim_5 : index
          %167:3 = scf.if %166 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
            %210 = arith.subi %165, %91 : index
            %211 = arith.divsi %210, %94 : index
            %212 = arith.remsi %211, %c2 : index
            %213 = affine.min #map(%165)[%41, %dim_5]
            %subview_20 = memref.subview %subview_6[0, %165] [%98, %213] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %214 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %subview_21 = memref.subview %56[0, 0, 0] [2, %98, %213] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_22 = memref.subview %subview_21[%212, 0, 0] [1, %98, %213] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %215 = builtin.unrealized_conversion_cast %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %216 = llvm.extractvalue %214[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %217 = llvm.extractvalue %214[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %218 = llvm.getelementptr inbounds %216[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %219 = llvm.extractvalue %214[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %220 = llvm.extractvalue %214[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %221 = llvm.extractvalue %214[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %222 = llvm.trunc %220 : i64 to i32
            %223 = llvm.trunc %221 : i64 to i32
            %224 = llvm.mul %222, %c4_i32  : i32
            %225 = llvm.mul %223, %c4_i32  : i32
            %226 = llvm.sub %225, %224  : i32
            %227 = llvm.extractvalue %215[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %228 = llvm.extractvalue %215[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %229 = llvm.getelementptr inbounds %227[%228] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %230 = llvm.extractvalue %215[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %231 = llvm.extractvalue %215[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %232 = llvm.extractvalue %215[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %233 = llvm.trunc %231 : i64 to i32
            %234 = llvm.trunc %232 : i64 to i32
            %235 = llvm.mul %233, %c4_i32  : i32
            %236 = llvm.mul %234, %c4_i32  : i32
            %237 = llvm.sub %236, %235  : i32
            %238 = arith.subi %arg16, %91 : index
            %239 = arith.divsi %238, %94 : index
            %240 = arith.addi %239, %c1 : index
            %241 = arith.remsi %240, %c2 : index
            %242 = arith.addi %241, %c4 : index
            %243 = arith.index_castui %242 : index to i32
            %244 = func.call @dma_p2p_opt(%218, %219, %224, %226, %229, %230, %235, %237, %false, %c0_i32, %243) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %244, %alloca_2[%241] : memref<2xi32>
            %subview_23 = memref.subview %arg15[0, %165] [%131, %213] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            scf.yield %subview_20, %subview_22, %subview_23 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
          } else {
            scf.yield %arg17, %arg18, %arg19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
          }
          %168 = affine.min #map(%arg16)[%41, %dim_5]
          %169 = arith.divsi %c0, %57 : index
          %170 = arith.divsi %169, %c3 : index
          %171 = arith.muli %170, %c3 : index
          %172 = arith.subi %169, %171 : index
          %173 = affine.min #map3(%arg12)[%57, %26, %dim]
          // %subview_16 = memref.subview %arg14[0, 0] [%173, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_17 = memref.subview %arg19[0, 0] [%173, %168] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %174 = builtin.unrealized_conversion_cast %subview_17 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_18 = memref.subview %72[0, 0, 0] [3, %173, %168] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_19 = memref.subview %subview_18[%172, 0, 0] [1, %173, %168] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %175 = builtin.unrealized_conversion_cast %subview_19 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %176 = llvm.extractvalue %174[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %177 = llvm.extractvalue %174[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %178 = llvm.getelementptr inbounds %176[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %179 = llvm.extractvalue %174[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %180 = llvm.extractvalue %174[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %181 = llvm.extractvalue %174[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %182 = llvm.trunc %180 : i64 to i32
          %183 = llvm.trunc %181 : i64 to i32
          %184 = llvm.mul %182, %c4_i32  : i32
          %185 = llvm.mul %183, %c4_i32  : i32
          %186 = llvm.sub %185, %184  : i32
          %187 = llvm.extractvalue %175[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %188 = llvm.extractvalue %175[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %189 = llvm.getelementptr inbounds %187[%188] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %190 = llvm.extractvalue %175[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %191 = llvm.extractvalue %175[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %192 = llvm.extractvalue %175[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %193 = llvm.trunc %191 : i64 to i32
          %194 = llvm.trunc %192 : i64 to i32
          %195 = llvm.mul %193, %c4_i32  : i32
          %196 = llvm.mul %194, %c4_i32  : i32
          %197 = llvm.sub %196, %195  : i32
          %198 = func.call @dma_p2p_opt(%178, %179, %184, %186, %189, %190, %195, %197, %false, %c0_i32, %c6_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %198, %alloca_1[%c0] : memref<2xi32>
          %199 = arith.subi %arg16, %91 : index
          %200 = arith.divsi %199, %94 : index
          %201 = arith.remsi %200, %c2 : index
          %202 = memref.load %alloca_2[%201] : memref<2xi32>
          func.call @dma_wait_p2p(%202) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          scf.for %arg20 = %c0 to %131 step %57 {
            %212 = arith.addi %arg20, %57 : index
            %213 = arith.cmpi slt, %212, %131 : index
            scf.if %213 {
              %274 = arith.divsi %212, %57 : index
              %275 = arith.divsi %274, %c3 : index
              %276 = arith.muli %275, %c3 : index
              %277 = arith.subi %274, %276 : index
              %278 = affine.min #map(%212)[%57, %131]
              // %subview_24 = memref.subview %arg14[%212, 0] [%278, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %subview_25 = memref.subview %arg19[%212, 0] [%278, %168] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %279 = builtin.unrealized_conversion_cast %subview_25 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %subview_26 = memref.subview %72[0, 0, 0] [3, %278, %168] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
              %subview_27 = memref.subview %subview_26[%277, 0, 0] [1, %278, %168] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %280 = builtin.unrealized_conversion_cast %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %281 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %282 = llvm.extractvalue %279[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %283 = llvm.getelementptr inbounds %281[%282] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %284 = llvm.extractvalue %279[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %285 = llvm.extractvalue %279[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %286 = llvm.extractvalue %279[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %287 = llvm.trunc %285 : i64 to i32
              %288 = llvm.trunc %286 : i64 to i32
              %289 = llvm.mul %287, %c4_i32  : i32
              %290 = llvm.mul %288, %c4_i32  : i32
              %291 = llvm.sub %290, %289  : i32
              %292 = llvm.extractvalue %280[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %293 = llvm.extractvalue %280[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %294 = llvm.getelementptr inbounds %292[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %295 = llvm.extractvalue %280[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %296 = llvm.extractvalue %280[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %297 = llvm.extractvalue %280[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %298 = llvm.trunc %296 : i64 to i32
              %299 = llvm.trunc %297 : i64 to i32
              %300 = llvm.mul %298, %c4_i32  : i32
              %301 = llvm.mul %299, %c4_i32  : i32
              %302 = llvm.sub %301, %300  : i32
              %303 = arith.divsi %arg20, %57 : index
              %304 = arith.addi %303, %c1 : index
              %305 = arith.remsi %304, %c2 : index
              %306 = arith.addi %305, %c6 : index
              %307 = arith.index_castui %306 : index to i32
              %308 = func.call @dma_p2p_opt(%283, %284, %289, %291, %294, %295, %300, %302, %false, %c0_i32, %307) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
              memref.store %308, %alloca_1[%305] : memref<2xi32>
            }
            %215 = affine.min #map(%arg20)[%57, %131]

            %idx_ma = arith.divsi %arg20, %57 : index
            %ch_ma = arith.remsi %idx_ma, %c3 : index
            %ma = affine.min #map(%arg20)[%57, %131]
            %subview_124 = memref.subview %arg14[%arg20, 0] [%ma, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %subview_125 = memref.subview %arg19[%arg20, 0] [%ma, %168] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %subview_126 = memref.subview %72[0, 0, 0] [3, %ma, %168] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_127 = memref.subview %subview_126[%ch_ma, 0, 0] [1, %ma, %168] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %210 = builtin.unrealized_conversion_cast %subview_127 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %211 = builtin.unrealized_conversion_cast %subview_125 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

            %216 = arith.divsi %c0, %73 : index
            %217 = arith.remsi %216, %c2 : index
            // %218 = affine.min #map3(%arg20)[%73, %57, %131]
            %subview_20 = memref.subview %subview_124[0, 0] [%73, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %219 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %subview_21 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_22 = memref.subview %subview_21[%217, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %220 = builtin.unrealized_conversion_cast %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %221 = llvm.extractvalue %219[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %222 = llvm.extractvalue %219[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %223 = llvm.getelementptr inbounds %221[%222] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %224 = llvm.extractvalue %219[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %225 = llvm.extractvalue %219[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %226 = llvm.extractvalue %219[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %227 = llvm.trunc %225 : i64 to i32
            %228 = llvm.trunc %226 : i64 to i32
            %229 = llvm.mul %227, %c4_i32  : i32
            %230 = llvm.mul %228, %c4_i32  : i32
            %231 = llvm.sub %230, %229  : i32
            %232 = llvm.extractvalue %220[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %233 = llvm.extractvalue %220[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %234 = llvm.getelementptr inbounds %232[%233] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %235 = llvm.extractvalue %220[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %236 = llvm.extractvalue %220[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %237 = llvm.extractvalue %220[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %238 = llvm.trunc %236 : i64 to i32
            %239 = llvm.trunc %237 : i64 to i32
            %240 = llvm.mul %238, %c4_i32  : i32
            %241 = llvm.mul %239, %c4_i32  : i32
            %242 = llvm.sub %241, %240  : i32
            %243 = func.call @dma_p2p_opt(%223, %224, %229, %231, %234, %235, %240, %242, %false, %c0_i32, %c0_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %243, %alloca_0[%c0] : memref<2xi32>
            // %subview_23 = memref.subview %subview_127[0, 0] [%73, %168] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %244 = arith.divsi %arg20, %57 : index
            %245 = arith.remsi %244, %c2 : index
            %246 = memref.load %alloca_1[%245] : memref<2xi32>
            func.call @dma_wait_p2p(%246) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            scf.for %arg24 = %c0 to %215 step %73 {
              // %274 = builtin.unrealized_conversion_cast %arg26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              // %275 = builtin.unrealized_conversion_cast %arg27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %276 = arith.addi %arg24, %73 : index
              %277 = arith.cmpi slt, %276, %215 : index
              scf.if %277 {
                %294 = arith.divsi %276, %73 : index
                %295 = arith.remsi %294, %c2 : index
                %subview_24 = memref.subview %subview_124[%276, 0] [%73, %98] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %297 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
                %subview_25 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
                %subview_26 = memref.subview %subview_25[%295, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %298 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
                %299 = llvm.extractvalue %297[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %300 = llvm.extractvalue %297[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %301 = llvm.getelementptr inbounds %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
                %302 = llvm.extractvalue %297[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %303 = llvm.extractvalue %297[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %304 = llvm.extractvalue %297[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %305 = llvm.trunc %303 : i64 to i32
                %306 = llvm.trunc %304 : i64 to i32
                %307 = llvm.mul %305, %c4_i32  : i32
                %308 = llvm.mul %306, %c4_i32  : i32
                %309 = llvm.sub %308, %307  : i32
                %310 = llvm.extractvalue %298[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %311 = llvm.extractvalue %298[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %312 = llvm.getelementptr inbounds %310[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f32
                %313 = llvm.extractvalue %298[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %314 = llvm.extractvalue %298[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %315 = llvm.extractvalue %298[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
                %316 = llvm.trunc %314 : i64 to i32
                %317 = llvm.trunc %315 : i64 to i32
                %318 = llvm.mul %316, %c4_i32  : i32
                %319 = llvm.mul %317, %c4_i32  : i32
                %320 = llvm.sub %319, %318  : i32
                %321 = arith.divsi %arg24, %73 : index
                %322 = arith.addi %321, %c1 : index
                %323 = arith.remsi %322, %c2 : index
                %324 = arith.index_castui %323 : index to i32
                %325 = func.call @dma_p2p_opt(%301, %302, %307, %309, %312, %313, %318, %320, %false, %c0_i32, %324) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
                memref.store %325, %alloca_0[%323] : memref<2xi32>
              }

              // new added
              %279 = arith.divsi %arg24, %73 : index
              %280 = arith.remsi %279, %c2 : index
              %subview_25 = memref.subview %88[0, 0, 0] [2, %73, %98] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
              %subview_26 = memref.subview %subview_25[%280, 0, 0] [1, %73, %98] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %subview_27 = memref.subview %subview_127[%arg24, 0] [%73, %168] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>

              %274 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %275 = builtin.unrealized_conversion_cast %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

              
              %281 = memref.load %alloca_0[%280] : memref<2xi32>
              func.call @dma_wait_p2p(%281) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
              %282 = llvm.extractvalue %274[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %283 = llvm.extractvalue %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %284 = llvm.extractvalue %275[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %285 = llvm.extractvalue %274[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %286 = llvm.extractvalue %164[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %287 = llvm.extractvalue %275[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %288 = llvm.getelementptr inbounds %282[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %289 = llvm.getelementptr inbounds %283[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %290 = llvm.getelementptr inbounds %284[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %291 = llvm.extractvalue %274[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %292 = arith.index_castui %25 : index to i64
              %293 = arith.index_castui %41 : index to i64
              func.call @matmul_micro_kernel_r12c128(%288, %289, %290, %291, %292, %293) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()

            } {__tiled_for___4}
            %248 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %249 = llvm.extractvalue %210[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %250 = llvm.getelementptr inbounds %248[%249] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %251 = llvm.extractvalue %210[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %252 = llvm.extractvalue %210[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %253 = llvm.extractvalue %210[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %254 = llvm.trunc %252 : i64 to i32
            %255 = llvm.trunc %253 : i64 to i32
            %256 = llvm.mul %254, %c4_i32  : i32
            %257 = llvm.mul %255, %c4_i32  : i32
            %258 = llvm.sub %257, %256  : i32
            %259 = llvm.extractvalue %211[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %260 = llvm.extractvalue %211[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %261 = llvm.getelementptr inbounds %259[%260] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %262 = llvm.extractvalue %211[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %263 = llvm.extractvalue %211[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %264 = llvm.extractvalue %211[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %265 = llvm.trunc %263 : i64 to i32
            %266 = llvm.trunc %264 : i64 to i32
            %267 = llvm.mul %265, %c4_i32  : i32
            %268 = llvm.mul %266, %c4_i32  : i32
            %269 = llvm.sub %268, %267  : i32
            %270 = arith.addi %245, %c8 : index
            %271 = arith.index_castui %270 : index to i32
            
            %273 = arith.cmpi ne, %arg20, %c0 : index
            scf.if %273 {
              %274 = arith.subi %arg20, %57 : index
              %275 = arith.divsi %274, %57 : index
              %276 = arith.remsi %275, %c2 : index
              %277 = memref.load %alloca[%276] : memref<2xi32>
              func.call @dma_wait_p2p(%277) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            }

            %272 = func.call @dma_p2p_opt(%250, %251, %256, %258, %261, %262, %267, %269, %false, %c0_i32, %271) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %272, %alloca[%245] : memref<2xi32>
            
          } {__tiled_for___3}
          %204 = arith.subi %57, %c1 : index
          %205 = arith.addi %131, %204 : index
          %206 = arith.divsi %205, %57 : index
          %207 = arith.subi %206, %c1 : index
          %208 = arith.remsi %207, %c2 : index
          %209 = memref.load %alloca[%208] : memref<2xi32>
          func.call @dma_wait_p2p(%209) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          scf.yield %167#0, %167#1, %167#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        } {__tiled_for___2}
        func.call @group_barrier(%c0_i32) : (i32) -> ()
        scf.yield %130#0, %130#1, %130#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
      } {__tiled_for___1}
    } {__tiled_for__}
    %95 = call @vector_free(%45) : (!llvm.ptr) -> i32
    %96 = call @vector_free(%61) : (!llvm.ptr) -> i32
    %97 = call @scalar_free(%77) : (!llvm.ptr) -> i32
    return
  }
}
