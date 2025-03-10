module {
  llvm.func @matmul_micro_kenrel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @dma_wait_p2p(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @vector_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @vector_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.mlir.global external @__gsm__() {addr_space = 0 : i32} : !llvm.array<2 x array<1536 x array<512 x f32>>>
  llvm.func @get_group_size() -> i32 attributes {sym_visibility = "private"}
  llvm.func @get_thread_id() -> i32 attributes {sym_visibility = "private"}
  llvm.func @matmul_only_tiling_pointerized(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) {
    %0 = llvm.mlir.constant(-1 : index) : i64
    %1 = llvm.mlir.constant(3735928559 : index) : i64
    %2 = llvm.mlir.constant(786432 : index) : i64
    %3 = llvm.mlir.constant(512 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(6 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.mlir.constant(12 : i64) : i64
    %11 = llvm.mlir.constant(8 : i64) : i64
    %12 = llvm.mlir.constant(0 : i64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(2 : index) : i64
    %17 = llvm.mlir.constant(3 : index) : i64
    %18 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %19 = llvm.alloca %16 x i32 : (i64) -> !llvm.ptr
    %20 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %21 = llvm.alloca %16 x i32 : (i64) -> !llvm.ptr
    %22 = llvm.alloca %16 x i32 : (i64) -> !llvm.ptr
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %arg3, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %arg3, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %12, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg0, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %arg2, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg4, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg4, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %12, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %arg2, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %arg1, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %arg1, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.mlir.addressof @__gsm__ : !llvm.ptr
    %37 = llvm.getelementptr %36[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<1536 x array<512 x f32>>>
    %38 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %39 = llvm.mul %arg6, %11  : i64
    %40 = llvm.mul %arg8, %39  : i64
    %41 = llvm.trunc %40 : i64 to i32
    %42 = llvm.call @vector_malloc(%41) : (i32) -> !llvm.ptr
    %43 = llvm.mul %arg8, %arg6  : i64
    %44 = llvm.mul %arg9, %10  : i64
    %45 = llvm.mul %arg8, %44  : i64
    %46 = llvm.trunc %45 : i64 to i32
    %47 = llvm.call @vector_malloc(%46) : (i32) -> !llvm.ptr
    %48 = llvm.mul %arg8, %arg9  : i64
    %49 = llvm.mul %arg10, %11  : i64
    %50 = llvm.mul %arg6, %49  : i64
    %51 = llvm.trunc %50 : i64 to i32
    %52 = llvm.call @scalar_malloc(%51) : (i32) -> !llvm.ptr
    %53 = llvm.mul %arg6, %arg10  : i64
    %54 = llvm.extractvalue %29[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.alloca %14 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %54, %55 : !llvm.array<2 x i64>, !llvm.ptr
    %56 = llvm.getelementptr %55[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %57 = llvm.load %56 : !llvm.ptr -> i64
    %58 = llvm.alloca %14 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %54, %58 : !llvm.array<2 x i64>, !llvm.ptr
    %59 = llvm.getelementptr %58[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %60 = llvm.load %59 : !llvm.ptr -> i64
    %61 = llvm.extractvalue %35[3] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.alloca %14 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
    llvm.store %61, %62 : !llvm.array<2 x i64>, !llvm.ptr
    %63 = llvm.getelementptr %62[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
    %64 = llvm.load %63 : !llvm.ptr -> i64
    %65 = llvm.call @get_thread_id() : () -> i32
    %66 = llvm.sext %65 : i32 to i64
    %67 = llvm.call @get_group_size() : () -> i32
    %68 = llvm.sext %67 : i32 to i64
    %69 = llvm.mul %arg8, %68  : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%70: i64):  // 2 preds: ^bb0, ^bb32
    %71 = llvm.icmp "slt" %70, %60 : i64
    llvm.cond_br %71, ^bb2, ^bb33
  ^bb2:  // pred: ^bb1
    %72 = llvm.mul %70, %0  : i64
    %73 = llvm.add %72, %60  : i64
    %74 = llvm.icmp "slt" %73, %arg6 : i64
    %75 = llvm.select %74, %73, %arg6 : i1, i64
    %76 = llvm.sdiv %15, %arg7  : i64
    %77 = llvm.srem %76, %16  : i64
    %78 = llvm.icmp "slt" %57, %arg7 : i64
    %79 = llvm.select %78, %57, %arg7 : i1, i64
    %80 = llvm.mul %77, %2  : i64
    %81 = llvm.add %80, %15  : i64
    %82 = llvm.insertvalue %38, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.insertvalue %37, %82[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %81, %83[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %79, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.insertvalue %3, %85[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %75, %86[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %14, %87[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.trunc %75 : i64 to i32
    %90 = llvm.trunc %arg2 : i64 to i32
    %91 = llvm.mul %89, %9  : i32
    %92 = llvm.mul %90, %9  : i32
    %93 = llvm.sub %92, %91  : i32
    %94 = llvm.trunc %3 : i64 to i32
    %95 = llvm.mul %94, %9  : i32
    %96 = llvm.sub %95, %91  : i32
    %97 = llvm.add %77, %16  : i64
    %98 = llvm.trunc %97 : i64 to i32
    %99 = llvm.call @dma_p2p_opt(%arg3, %79, %91, %93, %37, %79, %91, %96, %8, %7, %98) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %100 = llvm.getelementptr %22[%77] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %99, %100 : i32, !llvm.ptr
    %101 = llvm.insertvalue %arg5, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %arg5, %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %12, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %79, %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %arg1, %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %64, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %13, %106[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb3(%15, %88, %107 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%108: i64, %109: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %110: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb31
    %111 = llvm.icmp "slt" %108, %57 : i64
    llvm.cond_br %111, ^bb4, ^bb32
  ^bb4:  // pred: ^bb3
    %112 = llvm.add %108, %arg7  : i64
    %113 = llvm.icmp "slt" %112, %57 : i64
    llvm.cond_br %113, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %114 = llvm.sdiv %112, %arg7  : i64
    %115 = llvm.srem %114, %16  : i64
    %116 = llvm.mul %112, %0  : i64
    %117 = llvm.add %116, %57  : i64
    %118 = llvm.icmp "slt" %117, %arg7 : i64
    %119 = llvm.select %118, %117, %arg7 : i1, i64
    %120 = llvm.mul %115, %2  : i64
    %121 = llvm.add %120, %15  : i64
    %122 = llvm.insertvalue %121, %83[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.insertvalue %119, %122[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.insertvalue %3, %123[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.insertvalue %75, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %14, %125[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.add %115, %16  : i64
    %128 = llvm.trunc %127 : i64 to i32
    %129 = llvm.call @dma_p2p_opt(%arg3, %119, %91, %93, %37, %119, %91, %96, %8, %7, %128) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %130 = llvm.getelementptr %22[%115] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %129, %130 : i32, !llvm.ptr
    %131 = llvm.mul %112, %arg1  : i64
    %132 = llvm.add %131, %12  : i64
    %133 = llvm.insertvalue %132, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %119, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %arg1, %134[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %64, %135[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.insertvalue %13, %136[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%126, %137 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%109, %110 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb7(%138: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %139: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %140 = llvm.mul %108, %0  : i64
    %141 = llvm.add %140, %57  : i64
    %142 = llvm.icmp "slt" %141, %arg7 : i64
    %143 = llvm.select %142, %141, %arg7 : i1, i64
    %144 = llvm.sdiv %15, %69  : i64
    %145 = llvm.srem %144, %16  : i64
    %146 = llvm.mul %66, %0  : i64
    %147 = llvm.add %64, %146  : i64
    %148 = llvm.icmp "slt" %147, %arg8 : i64
    %149 = llvm.select %148, %147, %arg8 : i1, i64
    %150 = llvm.mul %145, %43  : i64
    %151 = llvm.add %150, %15  : i64
    %152 = llvm.insertvalue %42, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %42, %152[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.insertvalue %151, %153[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.insertvalue %75, %154[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.insertvalue %arg8, %155[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %149, %156[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %14, %157[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.trunc %149 : i64 to i32
    %160 = llvm.trunc %arg1 : i64 to i32
    %161 = llvm.mul %159, %9  : i32
    %162 = llvm.mul %160, %9  : i32
    %163 = llvm.sub %162, %161  : i32
    %164 = llvm.trunc %arg8 : i64 to i32
    %165 = llvm.mul %164, %9  : i32
    %166 = llvm.sub %165, %161  : i32
    %167 = llvm.add %145, %6  : i64
    %168 = llvm.trunc %167 : i64 to i32
    %169 = llvm.call @dma_p2p_opt(%arg4, %75, %161, %163, %42, %75, %161, %166, %8, %7, %168) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %170 = llvm.getelementptr %21[%145] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %169, %170 : i32, !llvm.ptr
    %171 = llvm.extractvalue %110[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.extractvalue %110[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.extractvalue %110[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %174 = llvm.extractvalue %110[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.extractvalue %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mul %66, %175  : i64
    %177 = llvm.add %173, %176  : i64
    %178 = llvm.insertvalue %171, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.insertvalue %172, %178[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.insertvalue %177, %179[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %143, %180[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %174, %181[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %149, %182[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.insertvalue %175, %183[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.sdiv %108, %arg7  : i64
    %186 = llvm.srem %185, %16  : i64
    %187 = llvm.getelementptr %22[%186] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %188 = llvm.load %187 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%188) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb9(%66, %158, %184 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb9(%189: i64, %190: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %191: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb8, ^bb30
    %192 = llvm.icmp "slt" %189, %64 : i64
    llvm.cond_br %192, ^bb10, ^bb31
  ^bb10:  // pred: ^bb9
    %193 = llvm.add %189, %69  : i64
    %194 = llvm.icmp "slt" %193, %64 : i64
    llvm.cond_br %194, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %195 = llvm.sub %193, %66  : i64
    %196 = llvm.sdiv %195, %69  : i64
    %197 = llvm.srem %196, %16  : i64
    %198 = llvm.mul %193, %0  : i64
    %199 = llvm.add %198, %64  : i64
    %200 = llvm.icmp "slt" %199, %arg8 : i64
    %201 = llvm.select %200, %199, %arg8 : i1, i64
    %202 = llvm.mul %197, %43  : i64
    %203 = llvm.add %202, %15  : i64
    %204 = llvm.insertvalue %203, %153[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %205 = llvm.insertvalue %75, %204[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.insertvalue %arg8, %205[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.insertvalue %201, %206[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.insertvalue %14, %207[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.trunc %201 : i64 to i32
    %210 = llvm.mul %209, %9  : i32
    %211 = llvm.sub %162, %210  : i32
    %212 = llvm.sub %165, %210  : i32
    %213 = llvm.add %197, %6  : i64
    %214 = llvm.trunc %213 : i64 to i32
    %215 = llvm.call @dma_p2p_opt(%arg4, %75, %210, %211, %42, %75, %210, %212, %8, %7, %214) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %216 = llvm.getelementptr %21[%197] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %215, %216 : i32, !llvm.ptr
    %217 = llvm.mul %193, %175  : i64
    %218 = llvm.add %173, %217  : i64
    %219 = llvm.insertvalue %218, %179[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.insertvalue %143, %219[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.insertvalue %174, %220[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %222 = llvm.insertvalue %201, %221[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.insertvalue %175, %222[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%208, %223 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb12:  // pred: ^bb10
    llvm.br ^bb13(%190, %191 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb13(%224: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %225: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    %226 = llvm.mul %189, %0  : i64
    %227 = llvm.add %226, %64  : i64
    %228 = llvm.icmp "slt" %227, %arg8 : i64
    %229 = llvm.select %228, %227, %arg8 : i1, i64
    %230 = llvm.sdiv %15, %arg9  : i64
    %231 = llvm.srem %230, %17  : i64
    %232 = llvm.icmp "slt" %143, %arg9 : i64
    %233 = llvm.select %232, %143, %arg9 : i1, i64
    %234 = llvm.extractvalue %109[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %235 = llvm.extractvalue %109[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.extractvalue %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.extractvalue %109[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %238 = llvm.insertvalue %234, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %239 = llvm.insertvalue %235, %238[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.insertvalue %236, %239[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.insertvalue %233, %240[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.insertvalue %237, %241[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.insertvalue %75, %242[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.insertvalue %14, %243[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.extractvalue %191[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %246 = llvm.extractvalue %191[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.extractvalue %191[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.extractvalue %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %249 = llvm.extractvalue %191[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.insertvalue %245, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.insertvalue %246, %250[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %252 = llvm.insertvalue %247, %251[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %253 = llvm.insertvalue %233, %252[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.insertvalue %248, %253[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %255 = llvm.insertvalue %229, %254[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %256 = llvm.insertvalue %249, %255[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %257 = llvm.mul %231, %48  : i64
    %258 = llvm.add %257, %15  : i64
    %259 = llvm.insertvalue %47, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %260 = llvm.insertvalue %47, %259[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %261 = llvm.insertvalue %258, %260[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %262 = llvm.insertvalue %233, %261[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %263 = llvm.insertvalue %arg8, %262[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %264 = llvm.insertvalue %229, %263[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %265 = llvm.insertvalue %14, %264[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %266 = llvm.trunc %229 : i64 to i32
    %267 = llvm.trunc %248 : i64 to i32
    %268 = llvm.mul %266, %9  : i32
    %269 = llvm.mul %267, %9  : i32
    %270 = llvm.sub %269, %268  : i32
    %271 = llvm.sub %165, %268  : i32
    %272 = llvm.add %231, %5  : i64
    %273 = llvm.trunc %272 : i64 to i32
    %274 = llvm.call @dma_p2p_opt(%246, %233, %268, %270, %47, %233, %268, %271, %8, %7, %273) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %275 = llvm.getelementptr %20[%231] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %274, %275 : i32, !llvm.ptr
    %276 = llvm.sub %189, %66  : i64
    %277 = llvm.sdiv %276, %69  : i64
    %278 = llvm.srem %277, %16  : i64
    %279 = llvm.getelementptr %21[%278] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %280 = llvm.load %279 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%280) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb15(%15, %244, %256, %265 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb15(%281: i64, %282: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %283: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %284: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb14, ^bb29
    %285 = llvm.icmp "slt" %281, %143 : i64
    llvm.cond_br %285, ^bb16, ^bb30
  ^bb16:  // pred: ^bb15
    %286 = llvm.add %281, %arg9  : i64
    %287 = llvm.icmp "slt" %286, %143 : i64
    llvm.cond_br %287, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %288 = llvm.sdiv %286, %arg9  : i64
    %289 = llvm.srem %288, %17  : i64
    %290 = llvm.mul %286, %0  : i64
    %291 = llvm.add %290, %143  : i64
    %292 = llvm.icmp "slt" %291, %arg9 : i64
    %293 = llvm.select %292, %291, %arg9 : i1, i64
    %294 = llvm.mul %286, %237  : i64
    %295 = llvm.add %236, %294  : i64
    %296 = llvm.insertvalue %295, %239[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %297 = llvm.insertvalue %293, %296[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %298 = llvm.insertvalue %237, %297[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.insertvalue %75, %298[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %300 = llvm.insertvalue %14, %299[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.mul %286, %248  : i64
    %302 = llvm.add %247, %301  : i64
    %303 = llvm.insertvalue %302, %251[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %304 = llvm.insertvalue %293, %303[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %305 = llvm.insertvalue %248, %304[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.insertvalue %229, %305[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %307 = llvm.insertvalue %249, %306[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %308 = llvm.mul %289, %48  : i64
    %309 = llvm.add %308, %15  : i64
    %310 = llvm.insertvalue %309, %260[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %311 = llvm.insertvalue %293, %310[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %312 = llvm.insertvalue %arg8, %311[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %313 = llvm.insertvalue %229, %312[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %314 = llvm.insertvalue %14, %313[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %315 = llvm.add %289, %5  : i64
    %316 = llvm.trunc %315 : i64 to i32
    %317 = llvm.call @dma_p2p_opt(%246, %293, %268, %270, %47, %293, %268, %271, %8, %7, %316) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %318 = llvm.getelementptr %20[%289] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %317, %318 : i32, !llvm.ptr
    llvm.br ^bb19(%300, %307, %314 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb18:  // pred: ^bb16
    llvm.br ^bb19(%282, %283, %284 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb19(%319: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %320: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %321: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb17, ^bb18
    llvm.br ^bb20
  ^bb20:  // pred: ^bb19
    %322 = llvm.mul %281, %0  : i64
    %323 = llvm.add %322, %143  : i64
    %324 = llvm.icmp "slt" %323, %arg9 : i64
    %325 = llvm.select %324, %323, %arg9 : i1, i64
    %326 = llvm.sdiv %15, %arg10  : i64
    %327 = llvm.srem %326, %16  : i64
    %328 = llvm.icmp "slt" %325, %arg10 : i64
    %329 = llvm.select %328, %325, %arg10 : i1, i64
    %330 = llvm.extractvalue %282[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %331 = llvm.extractvalue %282[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %332 = llvm.mul %327, %53  : i64
    %333 = llvm.add %332, %15  : i64
    %334 = llvm.insertvalue %52, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %335 = llvm.insertvalue %52, %334[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %336 = llvm.insertvalue %333, %335[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %337 = llvm.insertvalue %329, %336[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %338 = llvm.insertvalue %arg6, %337[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %339 = llvm.insertvalue %75, %338[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %340 = llvm.insertvalue %14, %339[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %341 = llvm.trunc %331 : i64 to i32
    %342 = llvm.mul %341, %9  : i32
    %343 = llvm.sub %342, %91  : i32
    %344 = llvm.trunc %arg6 : i64 to i32
    %345 = llvm.mul %344, %9  : i32
    %346 = llvm.sub %345, %91  : i32
    %347 = llvm.trunc %327 : i64 to i32
    %348 = llvm.call @dma_p2p_opt(%330, %329, %91, %343, %52, %329, %91, %346, %8, %7, %347) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %349 = llvm.getelementptr %19[%327] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %348, %349 : i32, !llvm.ptr
    %350 = llvm.extractvalue %284[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %351 = llvm.extractvalue %284[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %352 = llvm.extractvalue %284[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %353 = llvm.extractvalue %284[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.insertvalue %350, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %355 = llvm.insertvalue %351, %354[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %356 = llvm.insertvalue %352, %355[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %357 = llvm.insertvalue %329, %356[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %358 = llvm.insertvalue %353, %357[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %359 = llvm.insertvalue %229, %358[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %360 = llvm.insertvalue %14, %359[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %361 = llvm.sdiv %281, %arg9  : i64
    %362 = llvm.srem %361, %17  : i64
    %363 = llvm.getelementptr %20[%362] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %364 = llvm.load %363 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%364) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb21(%15, %340, %360 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb21(%365: i64, %366: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %367: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb20, ^bb26
    %368 = llvm.icmp "slt" %365, %325 : i64
    llvm.cond_br %368, ^bb22, ^bb27
  ^bb22:  // pred: ^bb21
    %369 = llvm.add %365, %arg10  : i64
    %370 = llvm.icmp "slt" %369, %325 : i64
    llvm.cond_br %370, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %371 = llvm.sdiv %369, %arg10  : i64
    %372 = llvm.srem %371, %16  : i64
    %373 = llvm.mul %369, %0  : i64
    %374 = llvm.add %373, %325  : i64
    %375 = llvm.icmp "slt" %374, %arg10 : i64
    %376 = llvm.select %375, %374, %arg10 : i1, i64
    %377 = llvm.mul %372, %53  : i64
    %378 = llvm.add %377, %15  : i64
    %379 = llvm.insertvalue %378, %335[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %380 = llvm.insertvalue %376, %379[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %381 = llvm.insertvalue %arg6, %380[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %382 = llvm.insertvalue %75, %381[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %383 = llvm.insertvalue %14, %382[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %384 = llvm.trunc %372 : i64 to i32
    %385 = llvm.call @dma_p2p_opt(%330, %376, %91, %343, %52, %376, %91, %346, %8, %7, %384) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %386 = llvm.getelementptr %19[%372] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %385, %386 : i32, !llvm.ptr
    %387 = llvm.mul %369, %353  : i64
    %388 = llvm.add %352, %387  : i64
    %389 = llvm.insertvalue %388, %355[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %390 = llvm.insertvalue %376, %389[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %391 = llvm.insertvalue %353, %390[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %392 = llvm.insertvalue %229, %391[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %393 = llvm.insertvalue %14, %392[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb25(%383, %393 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb24:  // pred: ^bb22
    llvm.br ^bb25(%366, %367 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb25(%394: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, %395: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb23, ^bb24
    llvm.br ^bb26
  ^bb26:  // pred: ^bb25
    %396 = llvm.sdiv %365, %arg10  : i64
    %397 = llvm.srem %396, %16  : i64
    %398 = llvm.getelementptr %19[%397] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %399 = llvm.load %398 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%399) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    %400 = llvm.extractvalue %366[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %401 = llvm.extractvalue %190[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %402 = llvm.extractvalue %367[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %403 = llvm.extractvalue %366[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @matmul_micro_kenrel(%400, %401, %402, %403, %arg6, %arg8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.br ^bb21(%369, %394, %395 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb27:  // pred: ^bb21
    %404 = llvm.extractvalue %284[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %405 = llvm.extractvalue %284[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %406 = llvm.trunc %405 : i64 to i32
    %407 = llvm.trunc %353 : i64 to i32
    %408 = llvm.mul %406, %9  : i32
    %409 = llvm.mul %407, %9  : i32
    %410 = llvm.sub %409, %408  : i32
    %411 = llvm.extractvalue %283[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %412 = llvm.extractvalue %283[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %413 = llvm.extractvalue %283[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %414 = llvm.extractvalue %283[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %415 = llvm.trunc %413 : i64 to i32
    %416 = llvm.trunc %414 : i64 to i32
    %417 = llvm.mul %415, %9  : i32
    %418 = llvm.mul %416, %9  : i32
    %419 = llvm.sub %418, %417  : i32
    %420 = llvm.srem %361, %16  : i64
    %421 = llvm.add %420, %4  : i64
    %422 = llvm.trunc %421 : i64 to i32
    %423 = llvm.icmp "ne" %281, %15 : i64
    llvm.cond_br %423, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %424 = llvm.sub %281, %arg9  : i64
    %425 = llvm.sdiv %424, %arg9  : i64
    %426 = llvm.srem %425, %17  : i64
    %427 = llvm.getelementptr %18[%426] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %428 = llvm.load %427 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%428) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb29
  ^bb29:  // 2 preds: ^bb27, ^bb28
    %429 = llvm.call @dma_p2p_opt(%351, %404, %408, %410, %411, %412, %417, %419, %8, %7, %422) {operandSegmentSizes = array<i32: 1, 1>} : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %430 = llvm.getelementptr %18[%362] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %429, %430 : i32, !llvm.ptr
    llvm.br ^bb15(%286, %319, %320, %321 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb30:  // pred: ^bb15
    %431 = llvm.sub %arg9, %14  : i64
    %432 = llvm.add %143, %431  : i64
    %433 = llvm.sdiv %432, %arg9  : i64
    %434 = llvm.sub %433, %14  : i64
    %435 = llvm.srem %434, %17  : i64
    %436 = llvm.getelementptr %18[%435] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %437 = llvm.load %436 : !llvm.ptr -> i32
    llvm.call @dma_wait_p2p(%437) {operandSegmentSizes = array<i32: 1, 1>} : (i32) -> ()
    llvm.br ^bb9(%193, %224, %225 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb31:  // pred: ^bb9
    llvm.br ^bb3(%112, %138, %139 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb32:  // pred: ^bb3
    %438 = llvm.add %70, %arg6  : i64
    llvm.br ^bb1(%438 : i64)
  ^bb33:  // pred: ^bb1
    %439 = llvm.call @vector_free(%42) : (!llvm.ptr) -> i32
    %440 = llvm.call @vector_free(%47) : (!llvm.ptr) -> i32
    %441 = llvm.call @scalar_free(%52) : (!llvm.ptr) -> i32
    llvm.return
  }
}

