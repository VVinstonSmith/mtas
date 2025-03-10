#map = affine_map<()[s0, s1, s2] -> (s1 - s2, s0)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map2 = affine_map<(d0)[s0, s1, s2] -> (-d0 + s2, s1, s0)>
module {
  func.func private @group_barrier(i32)
  func.func private @matmul_micro_kernel_r12c128(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)
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
    %c1_i32 = arith.constant 1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %false = arith.constant false
    %c4_i32 = arith.constant 4 : i32
    %c12_i64 = arith.constant 12 : i64
    %c2_i64 = arith.constant 2 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i64 = arith.constant 3 : i64
    %alloca = memref.alloca() : memref<2xi32>
    %alloca_0 = memref.alloca() : memref<2xi32>
    %alloca_1 = memref.alloca() : memref<2xi32>
    %alloca_2 = memref.alloca() : memref<2xi32>
    %alloca_3 = memref.alloca() : memref<2xi32>
    call @set_prir(%c3_i64) : (i64) -> ()
    %0 = call @get_thread_id() : () -> i32
    %1 = arith.cmpi eq, %0, %c0_i32 {__is_tid_eq_zero__} : i32
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg3, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %c0_i64, %4[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg0, %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg2, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %arg2, %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %c1_i64, %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = builtin.unrealized_conversion_cast %9 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %11 = llvm.insertvalue %arg4, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg4, %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %c0_i64, %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg2, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg1, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %c1_i64, %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = builtin.unrealized_conversion_cast %17 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %19 = llvm.insertvalue %arg5, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg5, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %c0_i64, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg0, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg1, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %arg1, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %c1_i64, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %27 = index.casts %arg6 : i64 to index
    %28 = index.casts %arg7 : i64 to index
    %29 = llvm.mlir.addressof @gsm_mem : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1572864 x f32>
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.insertvalue %c0_i64, %33[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.insertvalue %arg6, %34[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.insertvalue %c1_i64, %35[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.insertvalue %arg7, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %arg6, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mul %arg6, %arg7  : i64
    %40 = llvm.insertvalue %c2_i64, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %39, %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = builtin.unrealized_conversion_cast %41 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %43 = index.casts %arg8 : i64 to index
    %44 = llvm.mul %arg6, %c8_i64  : i64
    %45 = llvm.mul %arg8, %44  : i64
    %46 = llvm.trunc %45 : i64 to i32
    %47 = call @vector_malloc(%46) : (i32) -> !llvm.ptr
    %48 = llvm.insertvalue %47, %31[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.insertvalue %47, %48[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.insertvalue %c0_i64, %49[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %arg8, %50[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.insertvalue %c1_i64, %51[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.insertvalue %arg6, %52[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %arg8, %53[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.mul %arg8, %arg6  : i64
    %56 = llvm.insertvalue %c2_i64, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.insertvalue %55, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = builtin.unrealized_conversion_cast %57 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %59 = index.casts %arg9 : i64 to index
    %60 = llvm.mul %arg9, %c12_i64  : i64
    %61 = llvm.mul %arg8, %60  : i64
    %62 = llvm.trunc %61 : i64 to i32
    %63 = call @vector_malloc(%62) : (i32) -> !llvm.ptr
    %64 = llvm.insertvalue %63, %31[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %63, %64[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %c0_i64, %65[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %arg8, %66[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %c1_i64, %67[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %arg9, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.insertvalue %arg8, %69[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.mul %arg8, %arg9  : i64
    %72 = llvm.insertvalue %c3_i64, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.insertvalue %71, %72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = builtin.unrealized_conversion_cast %73 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x?x?xf32>
    %75 = index.casts %arg10 : i64 to index
    %76 = llvm.mul %arg10, %c8_i64  : i64
    %77 = llvm.mul %arg6, %76  : i64
    %78 = llvm.trunc %77 : i64 to i32
    %79 = call @scalar_malloc(%78) : (i32) -> !llvm.ptr
    %80 = llvm.insertvalue %79, %31[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %81 = llvm.insertvalue %79, %80[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %82 = llvm.insertvalue %c0_i64, %81[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %83 = llvm.insertvalue %arg6, %82[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %84 = llvm.insertvalue %c1_i64, %83[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.insertvalue %arg10, %84[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %86 = llvm.insertvalue %arg6, %85[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %87 = llvm.mul %arg6, %arg10  : i64
    %88 = llvm.insertvalue %c2_i64, %86[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %89 = llvm.insertvalue %87, %88[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %90 = builtin.unrealized_conversion_cast %89 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<2x?x?xf32>
    %dim = memref.dim %10, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_4 = memref.dim %10, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_5 = memref.dim %18, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %91 = arith.subi %27, %c1 : index
    %92 = arith.addi %dim_4, %91 : index
    %93 = arith.divsi %92, %27 : index
    %94 = arith.subi %28, %c1 : index
    %95 = arith.addi %dim, %94 : index
    %96 = arith.divsi %95, %28 : index
    %97 = arith.muli %93, %96 : index
    %98 = call @get_thread_id() : () -> i32
    %99 = index.casts %98 : i32 to index
    %100 = arith.muli %43, %99 : index
    %101 = call @get_group_size() : () -> i32
    %102 = index.casts %101 : i32 to index
    %103 = arith.muli %43, %102 : index
    %104 = arith.muli %75, %c2 : index
    %105 = arith.divsi %c0, %96 : index
    %106 = arith.remsi %c0, %96 : index
    %107 = arith.muli %105, %27 : index
    %108 = arith.muli %106, %28 : index
    %109 = affine.min #map()[%27, %dim_4, %107]
    %110 = affine.min #map()[%28, %dim, %108]
    %subview = memref.subview %10[%108, %107] [%110, %109] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %111 = builtin.unrealized_conversion_cast %subview : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_6 = memref.subview %42[0, 0, 0] [2, %110, %109] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
    %subview_7 = memref.subview %subview_6[0, 0, 0] [1, %110, %109] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %112 = builtin.unrealized_conversion_cast %subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    scf.if %1 {
      %120 = llvm.extractvalue %111[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %121 = llvm.extractvalue %111[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %122 = llvm.getelementptr inbounds %120[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %123 = llvm.extractvalue %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %124 = llvm.extractvalue %111[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %125 = llvm.extractvalue %111[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %126 = llvm.trunc %124 : i64 to i32
      %127 = llvm.trunc %125 : i64 to i32
      %128 = llvm.mul %126, %c4_i32  : i32
      %129 = llvm.mul %127, %c4_i32  : i32
      %130 = llvm.sub %129, %128  : i32
      %131 = llvm.extractvalue %112[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %132 = llvm.extractvalue %112[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %133 = llvm.getelementptr inbounds %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %134 = llvm.extractvalue %112[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %135 = llvm.extractvalue %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %136 = llvm.extractvalue %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %137 = llvm.trunc %135 : i64 to i32
      %138 = llvm.trunc %136 : i64 to i32
      %139 = llvm.mul %137, %c4_i32  : i32
      %140 = llvm.mul %138, %c4_i32  : i32
      %141 = llvm.sub %140, %139  : i32
      %142 = func.call @dma_p2p_opt(%122, %123, %128, %130, %133, %134, %139, %141, %false, %c0_i32, %c2_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
      memref.store %142, %alloca_3[%c0] : memref<2xi32>
    }
    %113 = affine.min #map()[%43, %dim_5, %100]
    %114 = arith.index_castui %27 : index to i64
    %115 = arith.index_castui %43 : index to i64
    %116 = arith.subi %59, %c1 : index
    scf.for %arg11 = %c0 to %97 step %c1 {
      %120 = arith.addi %arg11, %c1 : index
      %121 = arith.cmpi slt, %120, %97 : index
      scf.if %121 {
        %182 = arith.remsi %120, %c2 : index
        %183 = arith.divsi %120, %96 : index
        %184 = arith.remsi %120, %96 : index
        %185 = arith.muli %183, %27 : index
        %186 = arith.muli %184, %28 : index
        %187 = affine.min #map1(%185)[%27, %dim_4]
        %188 = affine.min #map1(%186)[%28, %dim]
        %subview_18 = memref.subview %10[%186, %185] [%188, %187] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %189 = builtin.unrealized_conversion_cast %subview_18 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %subview_19 = memref.subview %42[0, 0, 0] [2, %188, %187] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %subview_20 = memref.subview %subview_19[%182, 0, 0] [1, %188, %187] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %190 = builtin.unrealized_conversion_cast %subview_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        scf.if %1 {
          %191 = llvm.extractvalue %189[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %192 = llvm.extractvalue %189[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %193 = llvm.getelementptr inbounds %191[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %194 = llvm.extractvalue %189[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %195 = llvm.extractvalue %189[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %196 = llvm.extractvalue %189[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %197 = llvm.trunc %195 : i64 to i32
          %198 = llvm.trunc %196 : i64 to i32
          %199 = llvm.mul %197, %c4_i32  : i32
          %200 = llvm.mul %198, %c4_i32  : i32
          %201 = llvm.sub %200, %199  : i32
          %202 = llvm.extractvalue %190[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %203 = llvm.extractvalue %190[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %204 = llvm.getelementptr inbounds %202[%203] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %205 = llvm.extractvalue %190[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %206 = llvm.extractvalue %190[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %207 = llvm.extractvalue %190[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %208 = llvm.trunc %206 : i64 to i32
          %209 = llvm.trunc %207 : i64 to i32
          %210 = llvm.mul %208, %c4_i32  : i32
          %211 = llvm.mul %209, %c4_i32  : i32
          %212 = llvm.sub %211, %210  : i32
          %213 = arith.addi %182, %c2 : index
          %214 = arith.index_castui %213 : index to i32
          %215 = func.call @dma_p2p_opt(%193, %194, %199, %201, %204, %205, %210, %212, %false, %c0_i32, %214) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %215, %alloca_3[%182] : memref<2xi32>
        }
      }
      %122 = arith.remsi %arg11, %c2 : index
      %123 = arith.divsi %arg11, %96 : index
      %124 = arith.remsi %arg11, %96 : index
      %125 = arith.muli %123, %27 : index
      %126 = arith.muli %124, %28 : index
      %127 = affine.min #map1(%125)[%27, %dim_4]
      %128 = affine.min #map1(%126)[%28, %dim]
      %subview_8 = memref.subview %42[0, 0, 0] [2, %128, %127] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
      %subview_9 = memref.subview %subview_8[%122, 0, 0] [1, %128, %127] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %18[%125, 0] [%127, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %26[%126, 0] [%128, %dim_5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_12 = memref.subview %90[0, 0, 0] [2, %75, %127] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
      %subview_13 = memref.subview %subview_12[0, 0, 0] [1, %75, %127] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %129 = builtin.unrealized_conversion_cast %subview_13 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %subview_14 = memref.subview %subview_12[1, 0, 0] [1, %75, %127] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %130 = builtin.unrealized_conversion_cast %subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %131 = arith.divsi %c0, %103 : index
      %132 = arith.remsi %131, %c2 : index
      %subview_15 = memref.subview %subview_10[0, %100] [%127, %113] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %133 = builtin.unrealized_conversion_cast %subview_15 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %subview_16 = memref.subview %58[0, 0, 0] [2, %127, %113] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
      %subview_17 = memref.subview %subview_16[%132, 0, 0] [1, %127, %113] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %134 = builtin.unrealized_conversion_cast %subview_17 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %135 = llvm.extractvalue %133[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %136 = llvm.extractvalue %133[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %137 = llvm.getelementptr inbounds %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %138 = llvm.extractvalue %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %139 = llvm.extractvalue %133[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %140 = llvm.extractvalue %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %141 = llvm.trunc %139 : i64 to i32
      %142 = llvm.trunc %140 : i64 to i32
      %143 = llvm.mul %141, %c4_i32  : i32
      %144 = llvm.mul %142, %c4_i32  : i32
      %145 = llvm.sub %144, %143  : i32
      %146 = llvm.extractvalue %134[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %147 = llvm.extractvalue %134[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %148 = llvm.getelementptr inbounds %146[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %149 = llvm.extractvalue %134[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %150 = llvm.extractvalue %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %151 = llvm.extractvalue %134[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %152 = llvm.trunc %150 : i64 to i32
      %153 = llvm.trunc %151 : i64 to i32
      %154 = llvm.mul %152, %c4_i32  : i32
      %155 = llvm.mul %153, %c4_i32  : i32
      %156 = llvm.sub %155, %154  : i32
      %157 = func.call @dma_p2p_opt(%137, %138, %143, %145, %148, %149, %154, %156, %false, %c0_i32, %c4_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
      memref.store %157, %alloca_2[%c0] : memref<2xi32>
      scf.if %1 {
        %182 = memref.load %alloca_3[%122] : memref<2xi32>
        func.call @dma_wait_p2p(%182) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
      }
      func.call @group_barrier(%c0_i32) : (i32) -> ()
      %158 = affine.min #map2(%126)[%59, %28, %dim]
      %159 = llvm.extractvalue %129[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %160 = llvm.extractvalue %129[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %161 = llvm.getelementptr inbounds %159[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %162 = llvm.extractvalue %129[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %163 = llvm.extractvalue %129[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %164 = llvm.extractvalue %129[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %165 = llvm.trunc %163 : i64 to i32
      %166 = llvm.trunc %164 : i64 to i32
      %167 = llvm.mul %165, %c4_i32  : i32
      %168 = llvm.mul %166, %c4_i32  : i32
      %169 = llvm.sub %168, %167  : i32
      %170 = llvm.extractvalue %130[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %171 = llvm.extractvalue %130[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %172 = llvm.getelementptr inbounds %170[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %173 = llvm.extractvalue %130[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %174 = llvm.extractvalue %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %175 = llvm.extractvalue %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %176 = llvm.trunc %174 : i64 to i32
      %177 = llvm.trunc %175 : i64 to i32
      %178 = llvm.mul %176, %c4_i32  : i32
      %179 = llvm.mul %177, %c4_i32  : i32
      %180 = llvm.sub %179, %178  : i32
      %181 = arith.addi %128, %116 : index
      scf.for %arg12 = %100 to %dim_5 step %103 {
        %182 = arith.addi %arg12, %103 : index
        %183 = arith.cmpi slt, %182, %dim_5 : index
        scf.if %183 {
          %224 = arith.subi %182, %100 : index
          %225 = arith.divsi %224, %103 : index
          %226 = arith.remsi %225, %c2 : index
          %227 = affine.min #map1(%182)[%43, %dim_5]
          %subview_24 = memref.subview %subview_10[0, %182] [%127, %227] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %228 = builtin.unrealized_conversion_cast %subview_24 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_25 = memref.subview %58[0, 0, 0] [2, %127, %227] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_26 = memref.subview %subview_25[%226, 0, 0] [1, %127, %227] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %229 = builtin.unrealized_conversion_cast %subview_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %230 = llvm.extractvalue %228[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %231 = llvm.extractvalue %228[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %232 = llvm.getelementptr inbounds %230[%231] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %233 = llvm.extractvalue %228[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %234 = llvm.extractvalue %228[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %235 = llvm.extractvalue %228[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %236 = llvm.trunc %234 : i64 to i32
          %237 = llvm.trunc %235 : i64 to i32
          %238 = llvm.mul %236, %c4_i32  : i32
          %239 = llvm.mul %237, %c4_i32  : i32
          %240 = llvm.sub %239, %238  : i32
          %241 = llvm.extractvalue %229[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %242 = llvm.extractvalue %229[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %243 = llvm.getelementptr inbounds %241[%242] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %244 = llvm.extractvalue %229[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %245 = llvm.extractvalue %229[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %246 = llvm.extractvalue %229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %247 = llvm.trunc %245 : i64 to i32
          %248 = llvm.trunc %246 : i64 to i32
          %249 = llvm.mul %247, %c4_i32  : i32
          %250 = llvm.mul %248, %c4_i32  : i32
          %251 = llvm.sub %250, %249  : i32
          %252 = arith.subi %arg12, %100 : index
          %253 = arith.divsi %252, %103 : index
          %254 = arith.addi %253, %c1 : index
          %255 = arith.remsi %254, %c2 : index
          %256 = arith.addi %255, %c4 : index
          %257 = arith.index_castui %256 : index to i32
          %258 = func.call @dma_p2p_opt(%232, %233, %238, %240, %243, %244, %249, %251, %false, %c0_i32, %257) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %258, %alloca_2[%255] : memref<2xi32>
        }
        %184 = arith.subi %arg12, %100 : index
        %185 = arith.divsi %184, %103 : index
        %186 = arith.remsi %185, %c2 : index
        %187 = affine.min #map1(%arg12)[%43, %dim_5]
        %subview_18 = memref.subview %58[0, 0, 0] [2, %127, %187] [1, 1, 1] : memref<2x?x?xf32> to memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %subview_19 = memref.subview %subview_18[%186, 0, 0] [1, %127, %187] [1, 1, 1] : memref<2x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %188 = builtin.unrealized_conversion_cast %subview_19 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %subview_20 = memref.subview %subview_11[0, %arg12] [%128, %187] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %189 = arith.divsi %c0, %59 : index
        %190 = arith.remsi %189, %c3 : index
        %subview_21 = memref.subview %subview_20[0, 0] [%158, %187] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %191 = builtin.unrealized_conversion_cast %subview_21 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %subview_22 = memref.subview %74[0, 0, 0] [3, %158, %187] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %subview_23 = memref.subview %subview_22[%190, 0, 0] [1, %158, %187] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %192 = builtin.unrealized_conversion_cast %subview_23 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %193 = llvm.extractvalue %191[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %194 = llvm.extractvalue %191[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %195 = llvm.getelementptr inbounds %193[%194] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %196 = llvm.extractvalue %191[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %197 = llvm.extractvalue %191[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %198 = llvm.extractvalue %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %199 = llvm.trunc %197 : i64 to i32
        %200 = llvm.trunc %198 : i64 to i32
        %201 = llvm.mul %199, %c4_i32  : i32
        %202 = llvm.mul %200, %c4_i32  : i32
        %203 = llvm.sub %202, %201  : i32
        %204 = llvm.extractvalue %192[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %205 = llvm.extractvalue %192[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %206 = llvm.getelementptr inbounds %204[%205] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %207 = llvm.extractvalue %192[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %208 = llvm.extractvalue %192[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %209 = llvm.extractvalue %192[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %210 = llvm.trunc %208 : i64 to i32
        %211 = llvm.trunc %209 : i64 to i32
        %212 = llvm.mul %210, %c4_i32  : i32
        %213 = llvm.mul %211, %c4_i32  : i32
        %214 = llvm.sub %213, %212  : i32
        %215 = func.call @dma_p2p_opt(%195, %196, %201, %203, %206, %207, %212, %214, %false, %c0_i32, %c6_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
        memref.store %215, %alloca_1[%c0] : memref<2xi32>
        %216 = memref.load %alloca_2[%186] : memref<2xi32>
        func.call @dma_wait_p2p(%216) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
        %217 = llvm.extractvalue %188[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %218 = llvm.extractvalue %188[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %219 = llvm.getelementptr inbounds %217[%218] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        scf.for %arg13 = %c0 to %128 step %59 {
          %224 = arith.addi %arg13, %59 : index
          %225 = arith.cmpi slt, %224, %128 : index
          scf.if %225 {
            %272 = arith.divsi %224, %59 : index
            %273 = arith.remsi %272, %c3 : index
            %274 = affine.min #map1(%224)[%59, %128]
            %subview_29 = memref.subview %subview_20[%224, 0] [%274, %187] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %275 = builtin.unrealized_conversion_cast %subview_29 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %subview_30 = memref.subview %74[0, 0, 0] [3, %274, %187] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
            %subview_31 = memref.subview %subview_30[%273, 0, 0] [1, %274, %187] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %276 = builtin.unrealized_conversion_cast %subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %277 = llvm.extractvalue %275[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %278 = llvm.extractvalue %275[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %279 = llvm.getelementptr inbounds %277[%278] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %280 = llvm.extractvalue %275[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %281 = llvm.extractvalue %275[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %282 = llvm.extractvalue %275[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %283 = llvm.trunc %281 : i64 to i32
            %284 = llvm.trunc %282 : i64 to i32
            %285 = llvm.mul %283, %c4_i32  : i32
            %286 = llvm.mul %284, %c4_i32  : i32
            %287 = llvm.sub %286, %285  : i32
            %288 = llvm.extractvalue %276[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %289 = llvm.extractvalue %276[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %290 = llvm.getelementptr inbounds %288[%289] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %291 = llvm.extractvalue %276[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %292 = llvm.extractvalue %276[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %293 = llvm.extractvalue %276[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %294 = llvm.trunc %292 : i64 to i32
            %295 = llvm.trunc %293 : i64 to i32
            %296 = llvm.mul %294, %c4_i32  : i32
            %297 = llvm.mul %295, %c4_i32  : i32
            %298 = llvm.sub %297, %296  : i32
            %299 = arith.divsi %arg13, %59 : index
            %300 = arith.addi %299, %c1 : index
            %301 = arith.remsi %300, %c2 : index
            %302 = arith.addi %301, %c6 : index
            %303 = arith.index_castui %302 : index to i32
            %304 = func.call @dma_p2p_opt(%279, %280, %285, %287, %290, %291, %296, %298, %false, %c0_i32, %303) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %304, %alloca_1[%301] : memref<2xi32>
          }
          %226 = arith.divsi %arg13, %59 : index
          %227 = arith.remsi %226, %c3 : index
          %228 = affine.min #map1(%arg13)[%59, %128]
          %subview_24 = memref.subview %subview_9[%arg13, 0] [%228, %127] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_25 = memref.subview %subview_20[%arg13, 0] [%228, %187] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %229 = builtin.unrealized_conversion_cast %subview_25 : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_26 = memref.subview %74[0, 0, 0] [3, %228, %187] [1, 1, 1] : memref<3x?x?xf32> to memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>>
          %subview_27 = memref.subview %subview_26[%227, 0, 0] [1, %228, %187] [1, 1, 1] : memref<3x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %230 = builtin.unrealized_conversion_cast %subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %subview_28 = memref.subview %subview_24[0, 0] [%75, %127] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %231 = builtin.unrealized_conversion_cast %subview_28 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %232 = llvm.extractvalue %231[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %233 = llvm.extractvalue %231[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %234 = llvm.getelementptr inbounds %232[%233] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %235 = llvm.extractvalue %231[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %236 = llvm.extractvalue %231[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %237 = llvm.extractvalue %231[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %238 = llvm.trunc %236 : i64 to i32
          %239 = llvm.trunc %237 : i64 to i32
          %240 = llvm.mul %238, %c4_i32  : i32
          %241 = llvm.mul %239, %c4_i32  : i32
          %242 = llvm.sub %241, %240  : i32
          %243 = func.call @dma_p2p_opt(%234, %235, %240, %242, %161, %162, %167, %169, %false, %c0_i32, %c0_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %243, %alloca_0[%c0] : memref<2xi32>
          %244 = arith.remsi %226, %c2 : index
          %245 = memref.load %alloca_1[%244] : memref<2xi32>
          func.call @dma_wait_p2p(%245) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          scf.for %arg14 = %c0 to %228 step %104 {
            %272 = arith.addi %arg14, %75 : index
            %subview_29 = memref.subview %subview_24[%272, 0] [%75, %127] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %273 = builtin.unrealized_conversion_cast %subview_29 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %274 = llvm.extractvalue %273[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %275 = llvm.extractvalue %273[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %276 = llvm.getelementptr inbounds %274[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %277 = llvm.extractvalue %273[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %278 = llvm.extractvalue %273[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %279 = llvm.extractvalue %273[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %280 = llvm.trunc %278 : i64 to i32
            %281 = llvm.trunc %279 : i64 to i32
            %282 = llvm.mul %280, %c4_i32  : i32
            %283 = llvm.mul %281, %c4_i32  : i32
            %284 = llvm.sub %283, %282  : i32
            %285 = func.call @dma_p2p_opt(%276, %277, %282, %284, %172, %173, %178, %180, %false, %c0_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
            memref.store %285, %alloca_0[%c1] : memref<2xi32>
            %subview_30 = memref.subview %subview_27[%arg14, 0] [%75, %187] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %286 = builtin.unrealized_conversion_cast %subview_30 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %287 = memref.load %alloca_0[%c0] : memref<2xi32>
            func.call @dma_wait_p2p(%287) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            %288 = llvm.extractvalue %286[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %289 = llvm.extractvalue %286[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %290 = llvm.getelementptr inbounds %288[%289] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            func.call @matmul_micro_kernel_r12c128(%161, %219, %290, %163, %114, %115) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
            %291 = arith.addi %arg14, %104 : index
            %292 = arith.cmpi slt, %291, %228 : index
            scf.if %292 {
              %subview_32 = memref.subview %subview_24[%291, 0] [%75, %127] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %298 = builtin.unrealized_conversion_cast %subview_32 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
              %299 = llvm.extractvalue %298[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %300 = llvm.extractvalue %298[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %301 = llvm.getelementptr inbounds %299[%300] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %302 = llvm.extractvalue %298[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %303 = llvm.extractvalue %298[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %304 = llvm.extractvalue %298[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
              %305 = llvm.trunc %303 : i64 to i32
              %306 = llvm.trunc %304 : i64 to i32
              %307 = llvm.mul %305, %c4_i32  : i32
              %308 = llvm.mul %306, %c4_i32  : i32
              %309 = llvm.sub %308, %307  : i32
              %310 = func.call @dma_p2p_opt(%301, %302, %307, %309, %161, %162, %167, %169, %false, %c0_i32, %c0_i32) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
              memref.store %310, %alloca_0[%c0] : memref<2xi32>
            }
            %subview_31 = memref.subview %subview_27[%272, 0] [%75, %187] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %293 = builtin.unrealized_conversion_cast %subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %294 = memref.load %alloca_0[%c1] : memref<2xi32>
            func.call @dma_wait_p2p(%294) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
            %295 = llvm.extractvalue %293[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %296 = llvm.extractvalue %293[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
            %297 = llvm.getelementptr inbounds %295[%296] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            func.call @matmul_micro_kernel_r12c128(%172, %219, %297, %174, %114, %115) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
          } {__tiled_for___4}
          %246 = llvm.extractvalue %230[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %247 = llvm.extractvalue %230[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %248 = llvm.getelementptr inbounds %246[%247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %249 = llvm.extractvalue %230[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %250 = llvm.extractvalue %230[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %251 = llvm.extractvalue %230[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %252 = llvm.trunc %250 : i64 to i32
          %253 = llvm.trunc %251 : i64 to i32
          %254 = llvm.mul %252, %c4_i32  : i32
          %255 = llvm.mul %253, %c4_i32  : i32
          %256 = llvm.sub %255, %254  : i32
          %257 = llvm.extractvalue %229[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %258 = llvm.extractvalue %229[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %259 = llvm.getelementptr inbounds %257[%258] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %260 = llvm.extractvalue %229[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %261 = llvm.extractvalue %229[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %262 = llvm.extractvalue %229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %263 = llvm.trunc %261 : i64 to i32
          %264 = llvm.trunc %262 : i64 to i32
          %265 = llvm.mul %263, %c4_i32  : i32
          %266 = llvm.mul %264, %c4_i32  : i32
          %267 = llvm.sub %266, %265  : i32
          %268 = arith.addi %244, %c8 : index
          %269 = arith.index_castui %268 : index to i32
          %270 = func.call @dma_p2p_opt(%248, %249, %254, %256, %259, %260, %265, %267, %false, %c0_i32, %269) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
          memref.store %270, %alloca[%244] : memref<2xi32>
          %271 = arith.cmpi ne, %arg13, %c0 : index
          scf.if %271 {
            %272 = arith.subi %arg13, %59 : index
            %273 = arith.divsi %272, %59 : index
            %274 = arith.remsi %273, %c2 : index
            %275 = memref.load %alloca[%274] : memref<2xi32>
            func.call @dma_wait_p2p(%275) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
          }
        } {__tiled_for___3}
        %220 = arith.divsi %181, %59 : index
        %221 = arith.subi %220, %c1 : index
        %222 = arith.remsi %221, %c2 : index
        %223 = memref.load %alloca[%222] : memref<2xi32>
        func.call @dma_wait_p2p(%223) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
      } {__tiled_for___2}
      func.call @group_barrier(%c0_i32) : (i32) -> ()
    } {__coalesced_loop__, __tiled_for___1}
    %117 = call @vector_free(%47) : (!llvm.ptr) -> i32
    %118 = call @vector_free(%63) : (!llvm.ptr) -> i32
    %119 = call @scalar_free(%79) : (!llvm.ptr) -> i32
    return
  }
}

