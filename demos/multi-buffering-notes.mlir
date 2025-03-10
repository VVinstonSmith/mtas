// after lowering innermost loop
#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @test_inner_loop(%arg_A: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg_B: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg_C: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
        %arg3: i64, %arg4: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 1 : index
    %len_M = memref.dim %arg_A, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %len_N = memref.dim %arg_B, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %len_K = memref.dim %arg_B, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %tile_ma = arith.index_cast %arg3 : i64 to index
    %tile_ms = arith.index_cast %arg4 : i64 to index
    %buffer_Ca = memref.alloc(%tile_ma, %len_N) {alignment = 64 : i64} : memref<?x?xf32>
    
    scf.for %arg5 = %c0 to %len_M step %tile_ma {
      %tile_ma_cur = affine.min #map(%arg5)[%tile_ma, %len_M]
      %subview_Aa = memref.subview %arg_A[%arg5, 0] [%tile_ma_cur, %len_K] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_Ca = memref.subview %arg_C[%arg5, 0] [%tile_ma_cur, %len_N] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      linalg.copy ins(%subview_Ca : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%buffer_Ca : memref<?x?xf32>)
      
      %buffer_As = memref.alloc(%tile_ms, %len_K) {alignment = 64 : i64} : memref<?x?xf32>
      scf.for %pos = %c0 to %tile_ma_cur step %tile_ms {
        %tile_ms_cur = affine.min #map(%pos)[%tile_ms, %tile_ma_cur]
        %subview_As = memref.subview %subview_Aa[%pos, 0] [%tile_ms_cur, %len_K] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        linalg.copy ins(%subview_As : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%buffer_As : memref<?x?xf32>)
        %subview_Cs = memref.subview %buffer_Ca[%pos, 0] [%tile_ms_cur, %len_N] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        linalg.matmul ins(%buffer_As, %arg_B : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_Cs : memref<?x?xf32, strided<[?, 1], offset: ?>>)
      }

      linalg.copy ins(%buffer_Ca : memref<?x?xf32>) outs(%subview_Ca : memref<?x?xf32, strided<[?, ?], offset: ?>>)

      // %subview_Ba = memref.subview %arg_B[%arg5, 0] [%tile_ma_cur, %len_M] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      // linalg.copy ins(%buffer_Ca : memref<?x?xf32>) outs(%subview_Ba : memref<?x?xf32, strided<[?, ?], offset: ?>>)


        // %buffer_As_end, %subview_Cs_end = scf.for %pos_cur = %pos_0 to %pos_end step %tile_ms 
        //         iter_args(%buffer_As_cur = %buffer_As_0, %subview_Cs_cur = %subview_Cs_0) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
        //     %pos_next = arith.addi %pos_cur, %tile_ms : index
        //     %idx_next = arith.divsi %pos_next, %tile_ms : index
        //     %backlen_next = arith.subi %tile_ma_cur, %pos_next : index
        //     %tile_ms_next = arith.minsi %tile_ms, %backlen_next : index
        //     %bufnum_next = arith.remsi %idx_next, %c2 : index
        //     %buffer_As_next = memref.subview %alloc_As[%bufnum_next, %pos_next, 0] [1, %tile_ms_next, %len_K] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

        //     %subview_As_next = memref.subview %subview_Aa[%pos_next, 0] [%tile_ms_next, %len_K] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        //     linalg.copy ins(%subview_As_next : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%buffer_As_next : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        //     %subview_Cs_next = memref.subview %buffer_Ca[%pos_next, 0] [%tile_ms_next, %len_N] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        
        //     linalg.matmul ins(%buffer_As_cur, %arg_B : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_Cs_cur : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        
        //     scf.yield %buffer_As_next, %subview_Cs_next : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        // }
        
        /////////////// after multi-buffering ///////////////
        // %alloc_As = memref.alloc(%tile_ms, %len_K) {alignment = 64 : i64} : memref<2x?x?xf32>
    
        // %pos_0 = arith.constant 0 : index
        // %idx_0 = arith.divsi %pos_0, %tile_ms : index
        // %backlen_0 = arith.subi %tile_ma_cur, %pos_0 : index
        // %tile_ms_0 = arith.minsi %tile_ms, %backlen_0 : index
        // %bufnum_0 = arith.remsi %idx_0, %c2 : index
        // %buffer_As_0 = memref.subview %alloc_As[%bufnum_0, 0, 0] [1, %tile_ms, %len_K] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

        // %subview_As_0 = memref.subview %subview_Aa[%pos_0, 0] [%tile_ms_0, %len_K] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        // linalg.copy ins(%subview_As_0 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%buffer_As_0 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        // %subview_Cs_0 = memref.subview %buffer_Ca[%pos_0, 0] [%tile_ms_0, %len_N] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

        // %backlen_end = arith.remsi %tile_ma_cur, %tile_ms : index // len_M % arg_tile_A
        // %cmp = arith.cmpi eq, %backlen_end, %c0 : index // (len_M % arg_tile_A == 0) ?
        // %result_0 = arith.subi %tile_ma_cur, %tile_ms : index // (len_M - arg_tile_A)
        // %result_1 = arith.subi %tile_ma_cur, %backlen_end : index // (len_M - len_M % arg_tile_A)
        // %pos_end = arith.select %cmp, %result_0, %result_1 : index
        // // %idx_end = arith.divsi %pos_end, %tile_ms : index
        // // %tile_ms_end = arith.subi %tile_ma_cur, %pos_end : index
        // // %bufnum_end = arith.remsi %idx_end, %c2 : index
        // // %buffer_As_end = memref.subview %alloc_As[%bufnum_end, %pos_end, 0] [1, %tile_ms_end, %len_K] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32>

        // %buffer_As_end, %subview_Cs_end = scf.for %pos_cur = %pos_0 to %pos_end step %tile_ms 
        //         iter_args(%buffer_As_cur = %buffer_As_0, %subview_Cs_cur = %subview_Cs_0) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
        //     %pos_next = arith.addi %pos_cur, %tile_ms : index
        //     %idx_next = arith.divsi %pos_next, %tile_ms : index
        //     %backlen_next = arith.subi %tile_ma_cur, %pos_next : index
        //     %tile_ms_next = arith.minsi %tile_ms, %backlen_next : index
        //     %bufnum_next = arith.remsi %idx_next, %c2 : index
        //     %buffer_As_next = memref.subview %alloc_As[%bufnum_next, %pos_next, 0] [1, %tile_ms_next, %len_K] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

        //     %subview_As_next = memref.subview %subview_Aa[%pos_next, 0] [%tile_ms_next, %len_K] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        //     linalg.copy ins(%subview_As_next : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%buffer_As_next : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        //     %subview_Cs_next = memref.subview %buffer_Ca[%pos_next, 0] [%tile_ms_next, %len_N] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        
        //     linalg.matmul ins(%buffer_As_cur, %arg_B : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_Cs_cur : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        
        //     scf.yield %buffer_As_next, %subview_Cs_next : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        // }
        
        // linalg.matmul ins(%buffer_As_end, %arg_B : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_Cs_end : memref<?x?xf32, strided<[?, 1], offset: ?>>)
        /////////////// after multi-buffering ///////////////

        // linalg.copy ins(%buffer_Ca : memref<?x?xf32>) outs(%subview_Ca : memref<?x?xf32, strided<[?, ?], offset: ?>>)
    }
    return
  }
}

