; ModuleID = 'sgemm_mtir_kernel.ll'
source_filename = "LLVMDialectModule"

@gsm_mem = external global [1572864 x float], section ".gsm", section ".gsm"

declare void @matmul_micro_kernel_r12c128(ptr, ptr, ptr, i64, i64, i64) local_unnamed_addr

declare void @group_barrier(i32) local_unnamed_addr

declare void @dma_wait_p2p(i32) local_unnamed_addr

declare i32 @dma_p2p_opt(ptr, i64, i32, i32, ptr, i64, i32, i32, i1, i32, i32) local_unnamed_addr

declare void @set_prir(i64) local_unnamed_addr

declare i32 @scalar_free(ptr) local_unnamed_addr

declare ptr @scalar_malloc(i32) local_unnamed_addr

declare i32 @vector_free(ptr) local_unnamed_addr

declare ptr @vector_malloc(i32) local_unnamed_addr

declare i32 @get_group_size() local_unnamed_addr

declare i32 @get_thread_id() local_unnamed_addr

define void @matmul_only_tiling_pointerized(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) local_unnamed_addr {
  %12 = alloca [2 x i32], align 4
  %13 = alloca [2 x i32], align 4
  %14 = alloca [2 x i32], align 4
  %15 = alloca [2 x i32], align 4
  %16 = alloca [2 x i32], align 4
  tail call void @set_prir(i64 3)
  %17 = mul i64 %7, %6
  %18 = shl i64 %6, 3
  %19 = mul i64 %18, %8
  %20 = trunc i64 %19 to i32
  %21 = tail call ptr @vector_malloc(i32 %20)
  %22 = mul i64 %8, %6
  %23 = mul i64 %9, %8
  %24 = trunc i64 %23 to i32
  %25 = mul i32 %24, 12
  %26 = tail call ptr @vector_malloc(i32 %25)
  %27 = mul i64 %18, %10
  %28 = trunc i64 %27 to i32
  %29 = tail call ptr @scalar_malloc(i32 %28)
  %30 = mul i64 %10, %6
  %31 = tail call i32 @get_thread_id()
  %32 = sext i32 %31 to i64
  %33 = mul i64 %32, %8
  %34 = tail call i32 @get_group_size()
  %35 = sext i32 %34 to i64
  %36 = mul i64 %35, %8
  %37 = icmp sgt i64 %2, 0
  br i1 %37, label %.lr.ph28, label %._crit_edge29

.lr.ph28:                                         ; preds = %11
  %38 = tail call i64 @llvm.smin.i64(i64 %0, i64 %7)
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr @gsm_mem, ptr @gsm_mem, i64 0, [2 x i64] undef, [2 x i64] undef }, i64 %38, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %6, 4, 0
  %41 = trunc i64 %2 to i32
  %42 = trunc i64 %6 to i32
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %5, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, ptr %5, 1
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 0, 2
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %38, 3, 0
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %1, 4, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 %1, 3, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 1, 4, 1
  %50 = icmp sgt i64 %0, 0
  %51 = sub i64 %1, %33
  %52 = tail call i64 @llvm.smin.i64(i64 %51, i64 %8)
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %21, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, ptr %21, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 0, 2
  %56 = trunc i64 %52 to i32
  %57 = trunc i64 %1 to i32
  %58 = shl i32 %56, 2
  %59 = shl i32 %57, 2
  %60 = sub i32 %59, %58
  %61 = trunc i64 %8 to i32
  %62 = shl i32 %61, 2
  %63 = sub i32 %62, %58
  %64 = icmp slt i64 %33, %1
  %65 = add i64 %9, -1
  br label %66

66:                                               ; preds = %.lr.ph28, %._crit_edge26
  %67 = phi i64 [ 0, %.lr.ph28 ], [ %297, %._crit_edge26 ]
  %68 = sub i64 %2, %67
  %69 = tail call i64 @llvm.smin.i64(i64 %68, i64 %6)
  %70 = mul i64 %67, %1
  %71 = getelementptr inbounds float, ptr %3, i64 %67
  %72 = trunc i64 %69 to i32
  %73 = shl i32 %72, 2
  %74 = sub i32 %41, %72
  %75 = shl i32 %74, 2
  %76 = sub i32 %42, %72
  %77 = shl i32 %76, 2
  %78 = tail call i32 @dma_p2p_opt(ptr %71, i64 %38, i32 %73, i32 %75, ptr nonnull @gsm_mem, i64 %38, i32 %73, i32 %77, i1 false, i32 0, i32 2)
  store i32 %78, ptr %16, align 4
  br i1 %50, label %.lr.ph25, label %._crit_edge26

.lr.ph25:                                         ; preds = %66
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %69, 3, 1
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, i64 1, 4, 1
  %81 = add i64 %70, %33
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 %69, 3, 0
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 %8, 4, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 %52, 3, 1
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 1, 4, 1
  %86 = getelementptr inbounds float, ptr %4, i64 %81
  br label %87

87:                                               ; preds = %.lr.ph25, %._crit_edge23
  %88 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %49, %.lr.ph25 ], [ %123, %._crit_edge23 ]
  %89 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %80, %.lr.ph25 ], [ %122, %._crit_edge23 ]
  %90 = phi i64 [ 0, %.lr.ph25 ], [ %91, %._crit_edge23 ]
  %91 = add i64 %90, %7
  %92 = icmp slt i64 %91, %0
  br i1 %92, label %93, label %121

93:                                               ; preds = %87
  %94 = sdiv i64 %91, %7
  %95 = srem i64 %94, 2
  %96 = sub i64 %0, %91
  %97 = tail call i64 @llvm.smin.i64(i64 %96, i64 %7)
  %98 = mul i64 %91, %2
  %99 = add i64 %98, %67
  %100 = mul i64 %17, %95
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr @gsm_mem, ptr @gsm_mem, i64 undef, [2 x i64] undef, [2 x i64] undef }, i64 %100, 2
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %101, i64 %97, 3, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, i64 %6, 4, 0
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 %69, 3, 1
  %105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, i64 1, 4, 1
  %106 = getelementptr inbounds float, ptr %3, i64 %99
  %107 = getelementptr inbounds float, ptr @gsm_mem, i64 %100
  %108 = sdiv i64 %90, %7
  %109 = add i64 %108, 1
  %110 = srem i64 %109, 2
  %111 = trunc i64 %110 to i32
  %112 = add nsw i32 %111, 2
  %113 = tail call i32 @dma_p2p_opt(ptr %106, i64 %97, i32 %73, i32 %75, ptr nonnull %107, i64 %97, i32 %73, i32 %77, i1 false, i32 0, i32 %112)
  %114 = getelementptr i32, ptr %16, i64 %110
  store i32 %113, ptr %114, align 4
  %115 = mul i64 %91, %1
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %115, 2
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 %97, 3, 0
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 %1, 4, 0
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 %1, 3, 1
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 1, 4, 1
  br label %121

121:                                              ; preds = %93, %87
  %122 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %105, %93 ], [ %89, %87 ]
  %123 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %120, %93 ], [ %88, %87 ]
  %124 = sub i64 %0, %90
  %125 = tail call i64 @llvm.smin.i64(i64 %124, i64 %7)
  %126 = tail call i32 @dma_p2p_opt(ptr %86, i64 %69, i32 %58, i32 %60, ptr %21, i64 %69, i32 %58, i32 %63, i1 false, i32 0, i32 4)
  store i32 %126, ptr %15, align 4
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, 0
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, 1
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, 2
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, 4, 0
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, 4, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %127, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %128, 1
  %134 = sdiv i64 %90, %7
  %135 = srem i64 %134, 2
  %136 = getelementptr i32, ptr %16, i64 %135
  %137 = load i32, ptr %136, align 4
  tail call void @dma_wait_p2p(i32 %137)
  tail call void @group_barrier(i32 0)
  br i1 %64, label %.lr.ph22, label %._crit_edge23

.lr.ph22:                                         ; preds = %121
  %138 = mul i64 %131, %33
  %139 = add i64 %138, %129
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 %139, 2
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 %125, 3, 0
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 %130, 4, 0
  %143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, i64 %52, 3, 1
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %143, i64 %131, 4, 1
  %145 = tail call i64 @llvm.smin.i64(i64 %125, i64 %9)
  %146 = icmp sgt i64 %125, 0
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 1
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 2
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, 4, 0
  %150 = trunc i64 %149 to i32
  %151 = sub i32 %150, %72
  %152 = shl i32 %151, 2
  %153 = add i64 %65, %125
  br label %154

154:                                              ; preds = %.lr.ph22, %._crit_edge20
  %155 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %144, %.lr.ph22 ], [ %196, %._crit_edge20 ]
  %156 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %85, %.lr.ph22 ], [ %195, %._crit_edge20 ]
  %157 = phi i64 [ %33, %.lr.ph22 ], [ %158, %._crit_edge20 ]
  %158 = add i64 %157, %36
  %159 = icmp slt i64 %158, %1
  br i1 %159, label %160, label %._crit_edge31

._crit_edge31:                                    ; preds = %154
  %.pre = sub i64 %157, %33
  br label %194

160:                                              ; preds = %154
  %161 = sub i64 %158, %33
  %162 = sdiv i64 %161, %36
  %163 = srem i64 %162, 2
  %164 = sub i64 %1, %158
  %165 = tail call i64 @llvm.smin.i64(i64 %164, i64 %8)
  %166 = add i64 %158, %70
  %167 = mul i64 %22, %163
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %167, 2
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, i64 %69, 3, 0
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 %8, 4, 0
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %170, i64 %165, 3, 1
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, i64 1, 4, 1
  %173 = getelementptr inbounds float, ptr %4, i64 %166
  %174 = trunc i64 %165 to i32
  %175 = shl i32 %174, 2
  %176 = sub i32 %59, %175
  %177 = getelementptr inbounds float, ptr %21, i64 %167
  %178 = sub i32 %62, %175
  %179 = sub i64 %157, %33
  %180 = sdiv i64 %179, %36
  %181 = add i64 %180, 1
  %182 = srem i64 %181, 2
  %183 = trunc i64 %182 to i32
  %184 = add nsw i32 %183, 4
  %185 = tail call i32 @dma_p2p_opt(ptr %173, i64 %69, i32 %175, i32 %176, ptr %177, i64 %69, i32 %175, i32 %178, i1 false, i32 0, i32 %184)
  %186 = getelementptr i32, ptr %15, i64 %182
  store i32 %185, ptr %186, align 4
  %187 = mul i64 %158, %131
  %188 = add i64 %187, %129
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 %188, 2
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 %125, 3, 0
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 %130, 4, 0
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, i64 %165, 3, 1
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, i64 %131, 4, 1
  br label %194

194:                                              ; preds = %._crit_edge31, %160
  %.pre-phi = phi i64 [ %.pre, %._crit_edge31 ], [ %179, %160 ]
  %195 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %156, %._crit_edge31 ], [ %172, %160 ]
  %196 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %155, %._crit_edge31 ], [ %193, %160 ]
  %197 = sub i64 %1, %157
  %198 = tail call i64 @llvm.smin.i64(i64 %197, i64 %8)
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, 1
  %200 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, 2
  %201 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %155, 4, 0
  %202 = getelementptr inbounds float, ptr %199, i64 %200
  %203 = trunc i64 %198 to i32
  %204 = trunc i64 %201 to i32
  %205 = shl i32 %203, 2
  %206 = sub i32 %204, %203
  %207 = shl i32 %206, 2
  %208 = sub i32 %62, %205
  %209 = tail call i32 @dma_p2p_opt(ptr %202, i64 %145, i32 %205, i32 %207, ptr %26, i64 %145, i32 %205, i32 %208, i1 false, i32 0, i32 6)
  store i32 %209, ptr %14, align 4
  %210 = sdiv i64 %.pre-phi, %36
  %211 = srem i64 %210, 2
  %212 = getelementptr i32, ptr %15, i64 %211
  %213 = load i32, ptr %212, align 4
  tail call void @dma_wait_p2p(i32 %213)
  br i1 %146, label %.lr.ph19, label %._crit_edge20

.lr.ph19:                                         ; preds = %194
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %156, 1
  %215 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %156, 2
  %216 = getelementptr inbounds float, ptr %214, i64 %215
  br label %217

217:                                              ; preds = %.lr.ph19, %289
  %218 = phi i64 [ 0, %.lr.ph19 ], [ %219, %289 ]
  %219 = add i64 %218, %9
  %220 = icmp slt i64 %219, %125
  br i1 %220, label %221, label %._crit_edge30

._crit_edge30:                                    ; preds = %217
  %.pre32 = sdiv i64 %218, %9
  br label %238

221:                                              ; preds = %217
  %222 = sdiv i64 %219, %9
  %.fr = freeze i64 %222
  %223 = srem i64 %.fr, 3
  %224 = sub i64 %125, %219
  %225 = tail call i64 @llvm.smin.i64(i64 %224, i64 %9)
  %226 = mul i64 %219, %201
  %227 = add i64 %226, %200
  %228 = mul i64 %223, %23
  %229 = getelementptr inbounds float, ptr %199, i64 %227
  %230 = getelementptr inbounds float, ptr %26, i64 %228
  %231 = sdiv i64 %218, %9
  %232 = add i64 %231, 1
  %233 = srem i64 %232, 2
  %234 = trunc i64 %233 to i32
  %235 = add nsw i32 %234, 6
  %236 = tail call i32 @dma_p2p_opt(ptr %229, i64 %225, i32 %205, i32 %207, ptr %230, i64 %225, i32 %205, i32 %208, i1 false, i32 0, i32 %235)
  %237 = getelementptr i32, ptr %14, i64 %233
  store i32 %236, ptr %237, align 4
  br label %238

238:                                              ; preds = %._crit_edge30, %221
  %.pre-phi33 = phi i64 [ %.pre32, %._crit_edge30 ], [ %231, %221 ]
  %239 = sub i64 %125, %218
  %240 = tail call i64 @llvm.smin.i64(i64 %239, i64 %9)
  %241 = srem i64 %.pre-phi33, 3
  %242 = mul i64 %218, %149
  %243 = add i64 %242, %148
  %244 = mul i64 %218, %201
  %245 = add i64 %244, %200
  %246 = mul i64 %241, %23
  %247 = getelementptr inbounds float, ptr %147, i64 %243
  %248 = tail call i32 @dma_p2p_opt(ptr %247, i64 %10, i32 %73, i32 %152, ptr %29, i64 %10, i32 %73, i32 %77, i1 false, i32 0, i32 0)
  store i32 %248, ptr %13, align 4
  %249 = srem i64 %.pre-phi33, 2
  %250 = getelementptr i32, ptr %14, i64 %249
  %251 = load i32, ptr %250, align 4
  tail call void @dma_wait_p2p(i32 %251)
  %252 = icmp sgt i64 %240, 0
  br i1 %252, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %238, %270
  %253 = phi i64 [ %254, %270 ], [ 0, %238 ]
  %254 = add i64 %253, %10
  %255 = icmp slt i64 %254, %240
  br i1 %255, label %256, label %.lr.ph._crit_edge

.lr.ph._crit_edge:                                ; preds = %.lr.ph
  %.pre34 = sdiv i64 %253, %10
  br label %270

256:                                              ; preds = %.lr.ph
  %257 = sdiv i64 %254, %10
  %258 = srem i64 %257, 2
  %259 = mul i64 %254, %149
  %260 = add i64 %259, %243
  %261 = mul i64 %258, %30
  %262 = getelementptr inbounds float, ptr %147, i64 %260
  %263 = getelementptr inbounds float, ptr %29, i64 %261
  %264 = sdiv i64 %253, %10
  %265 = add i64 %264, 1
  %266 = srem i64 %265, 2
  %267 = trunc i64 %266 to i32
  %268 = tail call i32 @dma_p2p_opt(ptr %262, i64 %10, i32 %73, i32 %152, ptr %263, i64 %10, i32 %73, i32 %77, i1 false, i32 0, i32 %267)
  %269 = getelementptr i32, ptr %13, i64 %266
  store i32 %268, ptr %269, align 4
  br label %270

270:                                              ; preds = %.lr.ph._crit_edge, %256
  %.pre-phi35 = phi i64 [ %.pre34, %.lr.ph._crit_edge ], [ %264, %256 ]
  %271 = srem i64 %.pre-phi35, 2
  %272 = mul i64 %271, %30
  %273 = mul i64 %253, %8
  %274 = add i64 %273, %246
  %275 = getelementptr i32, ptr %13, i64 %271
  %276 = load i32, ptr %275, align 4
  tail call void @dma_wait_p2p(i32 %276)
  %277 = getelementptr inbounds float, ptr %29, i64 %272
  %278 = getelementptr inbounds float, ptr %26, i64 %274
  tail call void @matmul_micro_kernel_r12c128(ptr %277, ptr %216, ptr %278, i64 %69, i64 %6, i64 %8)
  br i1 %255, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %270, %238
  %279 = getelementptr inbounds float, ptr %26, i64 %246
  %280 = getelementptr inbounds float, ptr %199, i64 %245
  %281 = trunc i64 %249 to i32
  %282 = add nsw i32 %281, 8
  %.not = icmp eq i64 %218, 0
  br i1 %.not, label %289, label %283

283:                                              ; preds = %._crit_edge
  %284 = sub i64 %218, %9
  %285 = sdiv i64 %284, %9
  %286 = srem i64 %285, 2
  %287 = getelementptr i32, ptr %12, i64 %286
  %288 = load i32, ptr %287, align 4
  tail call void @dma_wait_p2p(i32 %288)
  br label %289

289:                                              ; preds = %283, %._crit_edge
  %290 = tail call i32 @dma_p2p_opt(ptr %279, i64 %240, i32 %205, i32 %208, ptr %280, i64 %240, i32 %205, i32 %207, i1 false, i32 0, i32 %282)
  %291 = getelementptr i32, ptr %12, i64 %249
  store i32 %290, ptr %291, align 4
  br i1 %220, label %217, label %._crit_edge20

._crit_edge20:                                    ; preds = %289, %194
  %292 = sdiv i64 %153, %9
  %293 = add i64 %292, -1
  %294 = srem i64 %293, 2
  %295 = getelementptr i32, ptr %12, i64 %294
  %296 = load i32, ptr %295, align 4
  tail call void @dma_wait_p2p(i32 %296)
  br i1 %159, label %154, label %._crit_edge23

._crit_edge23:                                    ; preds = %._crit_edge20, %121
  tail call void @group_barrier(i32 0)
  br i1 %92, label %87, label %._crit_edge26

._crit_edge26:                                    ; preds = %._crit_edge23, %66
  %297 = add i64 %67, %6
  %298 = icmp slt i64 %297, %2
  br i1 %298, label %66, label %._crit_edge29

._crit_edge29:                                    ; preds = %._crit_edge26, %11
  %299 = tail call i32 @vector_free(ptr %21)
  %300 = tail call i32 @vector_free(ptr %26)
  %301 = tail call i32 @scalar_free(ptr %29)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
