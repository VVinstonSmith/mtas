module attributes {llvm.data_layout = ""} {
  llvm.func @matmul_micro_kernel_r12c128(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @group_barrier(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_wait_p2p(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @set_prir(i64) attributes {sym_visibility = "private"}
  llvm.func @scalar_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @vector_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @vector_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.mlir.global external @gsm_mem() {addr_space = 0 : i32} : !llvm.array<1572864 x f32>
  llvm.func @get_group_size() -> i32 attributes {sym_visibility = "private"}
  llvm.func @get_thread_id() -> i32 attributes {sym_visibility = "private"}
  llvm.func @matmul_only_tiling_pointerized(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) {
    %0 = llvm.mlir.constant(-1 : index) : i64
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.mlir.constant(6 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(false) : i1
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(12 : i64) : i64
    %10 = llvm.mlir.constant(8 : i64) : i64
    %11 = llvm.mlir.constant(0 : i64) : i64
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(3 : index) : i64
    %17 = llvm.mlir.constant(3 : i64) : i64
    %18 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    %19 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    %20 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    %21 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    %22 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    llvm.call @set_prir(%17) : (i64) -> ()
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %arg3, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %arg3, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %11, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg0, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %arg2, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg4, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg4, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %11, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %arg2, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %arg1, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %arg1, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.mlir.addressof @gsm_mem : !llvm.ptr
    %37 = llvm.getelementptr inbounds %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1572864 x f32>
    %38 = llvm.mul %arg6, %arg7  : i64
    %39 = llvm.mul %arg6, %10  : i64
    %40 = llvm.mul %arg8, %39  : i64
    %41 = llvm.trunc %40 : i64 to i32
    %42 = llvm.call @vector_malloc(%41) : (i32) -> !llvm.ptr
    %43 = llvm.mul %arg8, %arg6  : i64
    %44 = llvm.mul %arg9, %9  : i64
    %45 = llvm.mul %arg8, %44  : i64
    %46 = llvm.trunc %45 : i64 to i32
    %47 = llvm.call @vector_malloc(%46) : (i32) -> !llvm.ptr
    %48 = llvm.mul %arg8, %arg9  : i64
    %49 = llvm.mul %arg10, %10  : i64
    %50 = llvm.mul %arg6, %49  : i64
    %51 = llvm.trunc %50 : i64 to i32
    %52 = llvm.call @scalar_malloc(%51) : (i32) -> !llvm.ptr
    %53 = llvm.mul %arg6, %arg10  : i64
    %54 = llvm.extractvalue %29[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.alloca %13 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %54, %55 : !llvm.array<2 x i64>, !llvm.ptr
    %56 = llvm.getelementptr %55[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %57 = llvm.load %56 : !llvm.ptr -> i64
    %58 = llvm.alloca %13 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %54, %58 : !llvm.array<2 x i64>, !llvm.ptr
    %59 = llvm.getelementptr %58[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %60 = llvm.load %59 : !llvm.ptr -> i64
    %61 = llvm.extractvalue %35[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.alloca %13 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %61, %62 : !llvm.array<2 x i64>, !llvm.ptr
    %63 = llvm.getelementptr %62[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %64 = llvm.load %63 : !llvm.ptr -> i64
    %65 = llvm.call @get_thread_id() : () -> i32
    %66 = llvm.sext %65 : i32 to i64
    %67 = llvm.mul %arg8, %66  : i64
    %68 = llvm.call @get_group_size() : () -> i32
    %69 = llvm.sext %68 : i32 to i64
    %70 = llvm.mul %arg8, %69  : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%71: i64):  // 2 preds: ^bb0, ^bb28
    %72 = llvm.icmp "slt" %71, %60 : i64
    llvm.cond_br %72, ^bb2, ^bb29
  ^bb2:  // pred: ^bb1
    %73 = llvm.mul %71, %0  : i64
    %74 = llvm.add %73, %60  : i64
    %75 = llvm.icmp "slt" %74, %arg6 : i64
    %76 = llvm.select %75, %74, %arg6 : i1, i64
    %77 = llvm.mul %71, %12  : i64
    %78 = llvm.add %77, %11  : i64
    %79 = llvm.mul %71, %arg1  : i64
    %80 = llvm.add %79, %11  : i64
    %81 = llvm.sdiv %14, %arg7  : i64
    %82 = llvm.srem %81, %15  : i64
    %83 = llvm.icmp "slt" %57, %arg7 : i64
    %84 = llvm.select %83, %57, %arg7 : i1, i64
    %85 = llvm.mul %82, %38  : i64
    %86 = llvm.add %85, %14  : i64
    %87 = llvm.insertvalue %37, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %37, %87[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.insertvalue %86, %88[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.insertvalue %84, %89[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.insertvalue %arg6, %90[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.insertvalue %76, %91[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.insertvalue %13, %92[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.getelementptr inbounds %arg3[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.trunc %76 : i64 to i32
    %96 = llvm.trunc %arg2 : i64 to i32
    %97 = llvm.mul %95, %8  : i32
    %98 = llvm.mul %96, %8  : i32
    %99 = llvm.sub %98, %97  : i32
    %100 = llvm.getelementptr inbounds %37[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %101 = llvm.trunc %arg6 : i64 to i32
    %102 = llvm.mul %101, %8  : i32
    %103 = llvm.sub %102, %97  : i32
    %104 = llvm.call @dma_p2p_opt(%94, %84, %97, %99, %100, %84, %97, %103, %7, %6, %2) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.store %104, %22 : i32, !llvm.ptr
    %105 = llvm.insertvalue %arg5, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %arg5, %105[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %11, %106[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %84, %107[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %arg1, %108[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %64, %109[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %12, %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb3(%14, %93, %111 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%112: i64, %113: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %114: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb27
    %115 = llvm.icmp "slt" %112, %57 : i64
    llvm.cond_br %115, ^bb4, ^bb28
  ^bb4:  // pred: ^bb3
    %116 = llvm.add %112, %arg7  : i64
    %117 = llvm.icmp "slt" %116, %57 : i64
    llvm.cond_br %117, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %118 = llvm.sdiv %116, %arg7  : i64
    %119 = llvm.srem %118, %15  : i64
    %120 = llvm.mul %116, %0  : i64
    %121 = llvm.add %120, %57  : i64
    %122 = llvm.icmp "slt" %121, %arg7 : i64
    %123 = llvm.select %122, %121, %arg7 : i1, i64
    %124 = llvm.mul %116, %arg2  : i64
    %125 = llvm.add %78, %124  : i64
    %126 = llvm.mul %119, %38  : i64
    %127 = llvm.add %126, %14  : i64
    %128 = llvm.insertvalue %127, %88[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %123, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %arg6, %129[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %76, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %13, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.getelementptr inbounds %arg3[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.getelementptr inbounds %37[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %135 = llvm.sdiv %112, %arg7  : i64
    %136 = llvm.add %135, %13  : i64
    %137 = llvm.srem %136, %15  : i64
    %138 = llvm.add %137, %15  : i64
    %139 = llvm.trunc %138 : i64 to i32
    %140 = llvm.call @dma_p2p_opt(%133, %123, %97, %99, %134, %123, %97, %103, %7, %6, %139) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %141 = llvm.getelementptr %22[%137] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %140, %141 : i32, !llvm.ptr
    %142 = llvm.mul %116, %arg1  : i64
    %143 = llvm.add %142, %11  : i64
    %144 = llvm.insertvalue %143, %106[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %123, %144[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %arg1, %145[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.insertvalue %64, %146[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %12, %147[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%132, %148 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%113, %114 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb7(%149: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %150: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %151 = llvm.mul %112, %0  : i64
    %152 = llvm.add %151, %57  : i64
    %153 = llvm.icmp "slt" %152, %arg7 : i64
    %154 = llvm.select %153, %152, %arg7 : i1, i64
    %155 = llvm.sdiv %14, %70  : i64
    %156 = llvm.srem %155, %15  : i64
    %157 = llvm.mul %67, %0  : i64
    %158 = llvm.add %64, %157  : i64
    %159 = llvm.icmp "slt" %158, %arg8 : i64
    %160 = llvm.select %159, %158, %arg8 : i1, i64
    %161 = llvm.mul %67, %12  : i64
    %162 = llvm.add %80, %161  : i64
    %163 = llvm.mul %156, %43  : i64
    %164 = llvm.add %163, %14  : i64
    %165 = llvm.insertvalue %42, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %42, %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %164, %166[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.insertvalue %76, %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.insertvalue %arg8, %168[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.insertvalue %160, %169[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.insertvalue %13, %170[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.getelementptr inbounds %arg4[%162] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %173 = llvm.trunc %160 : i64 to i32
    %174 = llvm.trunc %arg1 : i64 to i32
    %175 = llvm.mul %173, %8  : i32
    %176 = llvm.mul %174, %8  : i32
    %177 = llvm.sub %176, %175  : i32
    %178 = llvm.getelementptr inbounds %42[%164] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %179 = llvm.trunc %arg8 : i64 to i32
    %180 = llvm.mul %179, %8  : i32
    %181 = llvm.sub %180, %175  : i32
    %182 = llvm.call @dma_p2p_opt(%172, %76, %175, %177, %178, %76, %175, %181, %7, %6, %8) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.store %182, %21 : i32, !llvm.ptr
    %183 = llvm.extractvalue %114[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.extractvalue %114[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.extractvalue %114[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.extractvalue %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.mul %67, %187  : i64
    %189 = llvm.add %185, %188  : i64
    %190 = llvm.insertvalue %183, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.insertvalue %184, %190[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %192 = llvm.insertvalue %189, %191[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.insertvalue %154, %192[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %194 = llvm.insertvalue %186, %193[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %195 = llvm.insertvalue %160, %194[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %196 = llvm.insertvalue %187, %195[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %197 = llvm.sdiv %112, %arg7  : i64
    %198 = llvm.srem %197, %15  : i64
    %199 = llvm.getelementptr %22[%198] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %200 = llvm.load %199 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%200) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.call @group_barrier(%6) : (i32) -> ()
    llvm.br ^bb9(%67, %171, %196 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb9(%201: i64, %202: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %203: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb8, ^bb26
    %204 = llvm.icmp "slt" %201, %64 : i64
    llvm.cond_br %204, ^bb10, ^bb27
  ^bb10:  // pred: ^bb9
    %205 = llvm.add %201, %70  : i64
    %206 = llvm.icmp "slt" %205, %64 : i64
    llvm.cond_br %206, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %207 = llvm.sub %205, %67  : i64
    %208 = llvm.sdiv %207, %70  : i64
    %209 = llvm.srem %208, %15  : i64
    %210 = llvm.mul %205, %0  : i64
    %211 = llvm.add %210, %64  : i64
    %212 = llvm.icmp "slt" %211, %arg8 : i64
    %213 = llvm.select %212, %211, %arg8 : i1, i64
    %214 = llvm.mul %205, %12  : i64
    %215 = llvm.add %80, %214  : i64
    %216 = llvm.mul %209, %43  : i64
    %217 = llvm.add %216, %14  : i64
    %218 = llvm.insertvalue %217, %166[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %219 = llvm.insertvalue %76, %218[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.insertvalue %arg8, %219[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.insertvalue %213, %220[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %222 = llvm.insertvalue %13, %221[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.getelementptr inbounds %arg4[%215] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %224 = llvm.trunc %213 : i64 to i32
    %225 = llvm.mul %224, %8  : i32
    %226 = llvm.sub %176, %225  : i32
    %227 = llvm.getelementptr inbounds %42[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %228 = llvm.sub %180, %225  : i32
    %229 = llvm.sub %201, %67  : i64
    %230 = llvm.sdiv %229, %70  : i64
    %231 = llvm.add %230, %13  : i64
    %232 = llvm.srem %231, %15  : i64
    %233 = llvm.add %232, %5  : i64
    %234 = llvm.trunc %233 : i64 to i32
    %235 = llvm.call @dma_p2p_opt(%223, %76, %225, %226, %227, %76, %225, %228, %7, %6, %234) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %236 = llvm.getelementptr %21[%232] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %235, %236 : i32, !llvm.ptr
    %237 = llvm.mul %205, %187  : i64
    %238 = llvm.add %185, %237  : i64
    %239 = llvm.insertvalue %238, %191[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.insertvalue %154, %239[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.insertvalue %186, %240[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.insertvalue %213, %241[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.insertvalue %187, %242[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%222, %243 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb12:  // pred: ^bb10
    llvm.br ^bb13(%202, %203 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb13(%244: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %245: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    %246 = llvm.mul %201, %0  : i64
    %247 = llvm.add %246, %64  : i64
    %248 = llvm.icmp "slt" %247, %arg8 : i64
    %249 = llvm.select %248, %247, %arg8 : i1, i64
    %250 = llvm.sdiv %14, %arg9  : i64
    %251 = llvm.sdiv %250, %16  : i64
    %252 = llvm.mul %251, %16  : i64
    %253 = llvm.sub %250, %252  : i64
    %254 = llvm.icmp "slt" %154, %arg9 : i64
    %255 = llvm.select %254, %154, %arg9 : i1, i64
    %256 = llvm.extractvalue %203[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %257 = llvm.extractvalue %203[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.extractvalue %203[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %259 = llvm.mul %253, %48  : i64
    %260 = llvm.add %259, %14  : i64
    %261 = llvm.getelementptr inbounds %256[%257] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %262 = llvm.trunc %249 : i64 to i32
    %263 = llvm.trunc %258 : i64 to i32
    %264 = llvm.mul %262, %8  : i32
    %265 = llvm.mul %263, %8  : i32
    %266 = llvm.sub %265, %264  : i32
    %267 = llvm.getelementptr inbounds %47[%260] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %268 = llvm.sub %180, %264  : i32
    %269 = llvm.call @dma_p2p_opt(%261, %255, %264, %266, %267, %255, %264, %268, %7, %6, %1) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.store %269, %20 : i32, !llvm.ptr
    %270 = llvm.sub %201, %67  : i64
    %271 = llvm.sdiv %270, %70  : i64
    %272 = llvm.srem %271, %15  : i64
    %273 = llvm.getelementptr %21[%272] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %274 = llvm.load %273 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%274) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb15(%14 : i64)
  ^bb15(%275: i64):  // 2 preds: ^bb14, ^bb25
    %276 = llvm.icmp "slt" %275, %154 : i64
    llvm.cond_br %276, ^bb16, ^bb26
  ^bb16:  // pred: ^bb15
    %277 = llvm.add %275, %arg9  : i64
    %278 = llvm.icmp "slt" %277, %154 : i64
    llvm.cond_br %278, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %279 = llvm.sdiv %277, %arg9  : i64
    %280 = llvm.sdiv %279, %16  : i64
    %281 = llvm.mul %280, %16  : i64
    %282 = llvm.sub %279, %281  : i64
    %283 = llvm.mul %277, %0  : i64
    %284 = llvm.add %283, %154  : i64
    %285 = llvm.icmp "slt" %284, %arg9 : i64
    %286 = llvm.select %285, %284, %arg9 : i1, i64
    %287 = llvm.mul %277, %258  : i64
    %288 = llvm.add %257, %287  : i64
    %289 = llvm.mul %282, %48  : i64
    %290 = llvm.add %289, %14  : i64
    %291 = llvm.getelementptr inbounds %256[%288] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %292 = llvm.getelementptr inbounds %47[%290] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %293 = llvm.sdiv %275, %arg9  : i64
    %294 = llvm.add %293, %13  : i64
    %295 = llvm.srem %294, %15  : i64
    %296 = llvm.add %295, %4  : i64
    %297 = llvm.trunc %296 : i64 to i32
    %298 = llvm.call @dma_p2p_opt(%291, %286, %264, %266, %292, %286, %264, %268, %7, %6, %297) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %299 = llvm.getelementptr %20[%295] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %298, %299 : i32, !llvm.ptr
    llvm.br ^bb18
  ^bb18:  // 2 preds: ^bb16, ^bb17
    %300 = llvm.mul %275, %0  : i64
    %301 = llvm.add %300, %154  : i64
    %302 = llvm.icmp "slt" %301, %arg9 : i64
    %303 = llvm.select %302, %301, %arg9 : i1, i64
    %304 = llvm.sdiv %275, %arg9  : i64
    %305 = llvm.srem %304, %16  : i64
    %306 = llvm.extractvalue %113[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %307 = llvm.extractvalue %113[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %308 = llvm.extractvalue %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %309 = llvm.mul %275, %308  : i64
    %310 = llvm.add %307, %309  : i64
    %311 = llvm.mul %275, %258  : i64
    %312 = llvm.add %257, %311  : i64
    %313 = llvm.mul %305, %48  : i64
    %314 = llvm.add %313, %14  : i64
    %315 = llvm.sdiv %14, %arg10  : i64
    %316 = llvm.srem %315, %15  : i64
    %317 = llvm.mul %316, %53  : i64
    %318 = llvm.add %317, %14  : i64
    %319 = llvm.getelementptr inbounds %306[%310] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %320 = llvm.trunc %308 : i64 to i32
    %321 = llvm.mul %320, %8  : i32
    %322 = llvm.sub %321, %97  : i32
    %323 = llvm.getelementptr inbounds %52[%318] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %324 = llvm.call @dma_p2p_opt(%319, %arg10, %97, %322, %323, %arg10, %97, %103, %7, %6, %6) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.store %324, %19 : i32, !llvm.ptr
    %325 = llvm.srem %304, %15  : i64
    %326 = llvm.getelementptr %20[%325] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %327 = llvm.load %326 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%327) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb19(%14 : i64)
  ^bb19(%328: i64):  // 2 preds: ^bb18, ^bb22
    %329 = llvm.icmp "slt" %328, %303 : i64
    llvm.cond_br %329, ^bb20, ^bb23
  ^bb20:  // pred: ^bb19
    %330 = llvm.add %328, %arg10  : i64
    %331 = llvm.icmp "slt" %330, %303 : i64
    llvm.cond_br %331, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %332 = llvm.sdiv %330, %arg10  : i64
    %333 = llvm.srem %332, %15  : i64
    %334 = llvm.mul %330, %308  : i64
    %335 = llvm.add %310, %334  : i64
    %336 = llvm.mul %333, %53  : i64
    %337 = llvm.add %336, %14  : i64
    %338 = llvm.getelementptr inbounds %306[%335] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %339 = llvm.getelementptr inbounds %52[%337] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %340 = llvm.sdiv %328, %arg10  : i64
    %341 = llvm.add %340, %13  : i64
    %342 = llvm.srem %341, %15  : i64
    %343 = llvm.trunc %342 : i64 to i32
    %344 = llvm.call @dma_p2p_opt(%338, %arg10, %97, %322, %339, %arg10, %97, %103, %7, %6, %343) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %345 = llvm.getelementptr %19[%342] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %344, %345 : i32, !llvm.ptr
    llvm.br ^bb22
  ^bb22:  // 2 preds: ^bb20, ^bb21
    %346 = llvm.sdiv %328, %arg10  : i64
    %347 = llvm.srem %346, %15  : i64
    %348 = llvm.mul %347, %53  : i64
    %349 = llvm.add %348, %14  : i64
    %350 = llvm.mul %328, %arg8  : i64
    %351 = llvm.add %314, %350  : i64
    %352 = llvm.getelementptr %19[%347] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %353 = llvm.load %352 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%353) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    %354 = llvm.extractvalue %202[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %355 = llvm.extractvalue %202[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %356 = llvm.getelementptr inbounds %52[%349] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %357 = llvm.getelementptr inbounds %354[%355] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %358 = llvm.getelementptr inbounds %47[%351] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @matmul_micro_kernel_r12c128(%356, %357, %358, %76, %arg6, %arg8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.br ^bb19(%330 : i64)
  ^bb23:  // pred: ^bb19
    %359 = llvm.getelementptr inbounds %47[%314] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %360 = llvm.getelementptr inbounds %256[%312] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %361 = llvm.add %325, %3  : i64
    %362 = llvm.trunc %361 : i64 to i32
    %363 = llvm.icmp "ne" %275, %14 : i64
    llvm.cond_br %363, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %364 = llvm.sub %275, %arg9  : i64
    %365 = llvm.sdiv %364, %arg9  : i64
    %366 = llvm.srem %365, %15  : i64
    %367 = llvm.getelementptr %18[%366] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %368 = llvm.load %367 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%368) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb25
  ^bb25:  // 2 preds: ^bb23, ^bb24
    %369 = llvm.call @dma_p2p_opt(%359, %303, %264, %268, %360, %303, %264, %266, %7, %6, %362) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %370 = llvm.getelementptr %18[%325] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %369, %370 : i32, !llvm.ptr
    llvm.br ^bb15(%277 : i64)
  ^bb26:  // pred: ^bb15
    %371 = llvm.sub %arg9, %13  : i64
    %372 = llvm.add %154, %371  : i64
    %373 = llvm.sdiv %372, %arg9  : i64
    %374 = llvm.sub %373, %13  : i64
    %375 = llvm.srem %374, %15  : i64
    %376 = llvm.getelementptr %18[%375] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %377 = llvm.load %376 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%377) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb9(%205, %244, %245 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb27:  // pred: ^bb9
    llvm.call @group_barrier(%6) : (i32) -> ()
    llvm.br ^bb3(%116, %149, %150 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb28:  // pred: ^bb3
    %378 = llvm.add %71, %arg6  : i64
    llvm.br ^bb1(%378 : i64)
  ^bb29:  // pred: ^bb1
    %379 = llvm.call @vector_free(%42) : (!llvm.ptr) -> i32
    %380 = llvm.call @vector_free(%47) : (!llvm.ptr) -> i32
    %381 = llvm.call @scalar_free(%52) : (!llvm.ptr) -> i32
    llvm.return
  }
}

