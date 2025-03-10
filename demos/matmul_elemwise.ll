; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__gsm__ = external global [2 x [1536 x [512 x float]]]

declare void @matmul_micro_kenrel(ptr, ptr, ptr, i64, i64, i64)

declare void @dma_wait_p2p(i32)

declare i32 @dma_p2p_opt(ptr, i64, i32, i32, ptr, i64, i32, i32, i1, i32, i32)

declare i32 @scalar_free(ptr)

declare ptr @scalar_malloc(i32)

declare i32 @vector_free(ptr)

declare ptr @vector_malloc(i32)

declare i32 @get_group_size()

declare i32 @get_thread_id()

define void @matmul_only_tiling_pointerized(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = alloca i32, i64 3, align 4
  %13 = alloca i32, i64 2, align 4
  %14 = alloca i32, i64 3, align 4
  %15 = alloca i32, i64 2, align 4
  %16 = alloca i32, i64 2, align 4
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, ptr %3, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %0, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %2, 4, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %2, 3, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, ptr %4, 1
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 0, 2
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %2, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %1, 4, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %1, 3, 1
  %29 = mul i64 %6, 8
  %30 = mul i64 %8, %29
  %31 = trunc i64 %30 to i32
  %32 = call ptr @vector_malloc(i32 %31)
  %33 = mul i64 %8, %6
  %34 = mul i64 %9, 12
  %35 = mul i64 %8, %34
  %36 = trunc i64 %35 to i32
  %37 = call ptr @vector_malloc(i32 %36)
  %38 = mul i64 %8, %9
  %39 = mul i64 %10, 8
  %40 = mul i64 %6, %39
  %41 = trunc i64 %40 to i32
  %42 = call ptr @scalar_malloc(i32 %41)
  %43 = mul i64 %6, %10
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, 3
  %45 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %44, ptr %45, align 4
  %46 = getelementptr [2 x i64], ptr %45, i32 0, i32 0
  %47 = load i64, ptr %46, align 4
  %48 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %44, ptr %48, align 4
  %49 = getelementptr [2 x i64], ptr %48, i32 0, i32 1
  %50 = load i64, ptr %49, align 4
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3
  %52 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %51, ptr %52, align 4
  %53 = getelementptr [2 x i64], ptr %52, i32 0, i32 1
  %54 = load i64, ptr %53, align 4
  %55 = call i32 @get_thread_id()
  %56 = sext i32 %55 to i64
  %57 = call i32 @get_group_size()
  %58 = sext i32 %57 to i64
  %59 = mul i64 %8, %58
  br label %60

60:                                               ; preds = %455, %11
  %61 = phi i64 [ %456, %455 ], [ 0, %11 ]
  %62 = icmp slt i64 %61, %50
  br i1 %62, label %63, label %457

63:                                               ; preds = %60
  %64 = mul i64 %61, -1
  %65 = add i64 %64, %50
  %66 = icmp slt i64 %65, %6
  %67 = select i1 %66, i64 %65, i64 %6
  %68 = sdiv i64 0, %7
  %69 = srem i64 %68, 2
  %70 = icmp slt i64 %47, %7
  %71 = select i1 %70, i64 %47, i64 %7
  %72 = mul i64 %69, 786432
  %73 = add i64 %72, 0
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr inttoptr (i64 3735928559 to ptr), ptr @__gsm__, i64 undef, [2 x i64] undef, [2 x i64] undef }, i64 %73, 2
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 %71, 3, 0
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, i64 512, 4, 0
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, i64 %67, 3, 1
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, i64 1, 4, 1
  %79 = trunc i64 %67 to i32
  %80 = trunc i64 %2 to i32
  %81 = mul i32 %79, 4
  %82 = mul i32 %80, 4
  %83 = sub i32 %82, %81
  %84 = sub i32 2048, %81
  %85 = add i64 %69, 2
  %86 = trunc i64 %85 to i32
  %87 = call i32 @dma_p2p_opt(ptr %3, i64 %71, i32 %81, i32 %83, ptr @__gsm__, i64 %71, i32 %81, i32 %84, i1 false, i32 0, i32 %86)
  %88 = getelementptr i32, ptr %16, i64 %69
  store i32 %87, ptr %88, align 4
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %5, 0
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, ptr %5, 1
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 0, 2
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 %71, 3, 0
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 %1, 4, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, i64 %54, 3, 1
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 1, 4, 1
  br label %96

96:                                               ; preds = %454, %63
  %97 = phi i64 [ %102, %454 ], [ 0, %63 ]
  %98 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %131, %454 ], [ %78, %63 ]
  %99 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %132, %454 ], [ %95, %63 ]
  %100 = icmp slt i64 %97, %47
  br i1 %100, label %101, label %455

101:                                              ; preds = %96
  %102 = add i64 %97, %7
  %103 = icmp slt i64 %102, %47
  br i1 %103, label %104, label %129

104:                                              ; preds = %101
  %105 = sdiv i64 %102, %7
  %106 = srem i64 %105, 2
  %107 = mul i64 %102, -1
  %108 = add i64 %107, %47
  %109 = icmp slt i64 %108, %7
  %110 = select i1 %109, i64 %108, i64 %7
  %111 = mul i64 %106, 786432
  %112 = add i64 %111, 0
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr inttoptr (i64 3735928559 to ptr), ptr @__gsm__, i64 undef, [2 x i64] undef, [2 x i64] undef }, i64 %112, 2
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 %110, 3, 0
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 512, 4, 0
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, i64 %67, 3, 1
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 1, 4, 1
  %118 = add i64 %106, 2
  %119 = trunc i64 %118 to i32
  %120 = call i32 @dma_p2p_opt(ptr %3, i64 %110, i32 %81, i32 %83, ptr @__gsm__, i64 %110, i32 %81, i32 %84, i1 false, i32 0, i32 %119)
  %121 = getelementptr i32, ptr %16, i64 %106
  store i32 %120, ptr %121, align 4
  %122 = mul i64 %102, %1
  %123 = add i64 %122, 0
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 %123, 2
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 %110, 3, 0
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, i64 %1, 4, 0
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, i64 %54, 3, 1
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 1, 4, 1
  br label %130

129:                                              ; preds = %101
  br label %130

130:                                              ; preds = %104, %129
  %131 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %98, %129 ], [ %117, %104 ]
  %132 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %99, %129 ], [ %128, %104 ]
  br label %133

133:                                              ; preds = %130
  %134 = mul i64 %97, -1
  %135 = add i64 %134, %47
  %136 = icmp slt i64 %135, %7
  %137 = select i1 %136, i64 %135, i64 %7
  %138 = sdiv i64 0, %59
  %139 = srem i64 %138, 2
  %140 = mul i64 %56, -1
  %141 = add i64 %54, %140
  %142 = icmp slt i64 %141, %8
  %143 = select i1 %142, i64 %141, i64 %8
  %144 = mul i64 %139, %33
  %145 = add i64 %144, 0
  %146 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %32, 0
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %146, ptr %32, 1
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, i64 %145, 2
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 %67, 3, 0
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 %8, 4, 0
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 %143, 3, 1
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 1, 4, 1
  %153 = trunc i64 %143 to i32
  %154 = trunc i64 %1 to i32
  %155 = mul i32 %153, 4
  %156 = mul i32 %154, 4
  %157 = sub i32 %156, %155
  %158 = trunc i64 %8 to i32
  %159 = mul i32 %158, 4
  %160 = sub i32 %159, %155
  %161 = add i64 %139, 4
  %162 = trunc i64 %161 to i32
  %163 = call i32 @dma_p2p_opt(ptr %4, i64 %67, i32 %155, i32 %157, ptr %32, i64 %67, i32 %155, i32 %160, i1 false, i32 0, i32 %162)
  %164 = getelementptr i32, ptr %15, i64 %139
  store i32 %163, ptr %164, align 4
  %165 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, 0
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, 1
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, 2
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, 4, 0
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, 4, 1
  %170 = mul i64 %56, %169
  %171 = add i64 %167, %170
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %165, 0
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, ptr %166, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 %171, 2
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %137, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 %168, 4, 0
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %143, 3, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 %169, 4, 1
  %179 = sdiv i64 %97, %7
  %180 = srem i64 %179, 2
  %181 = getelementptr i32, ptr %16, i64 %180
  %182 = load i32, ptr %181, align 4
  call void @dma_wait_p2p(i32 %182)
  br label %183

183:                                              ; preds = %446, %133
  %184 = phi i64 [ %189, %446 ], [ %56, %133 ]
  %185 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %223, %446 ], [ %152, %133 ]
  %186 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %224, %446 ], [ %178, %133 ]
  %187 = icmp slt i64 %184, %54
  br i1 %187, label %188, label %454

188:                                              ; preds = %183
  %189 = add i64 %184, %59
  %190 = icmp slt i64 %189, %54
  br i1 %190, label %191, label %221

191:                                              ; preds = %188
  %192 = sub i64 %189, %56
  %193 = sdiv i64 %192, %59
  %194 = srem i64 %193, 2
  %195 = mul i64 %189, -1
  %196 = add i64 %195, %54
  %197 = icmp slt i64 %196, %8
  %198 = select i1 %197, i64 %196, i64 %8
  %199 = mul i64 %194, %33
  %200 = add i64 %199, 0
  %201 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, i64 %200, 2
  %202 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %201, i64 %67, 3, 0
  %203 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %202, i64 %8, 4, 0
  %204 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %203, i64 %198, 3, 1
  %205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %204, i64 1, 4, 1
  %206 = trunc i64 %198 to i32
  %207 = mul i32 %206, 4
  %208 = sub i32 %156, %207
  %209 = sub i32 %159, %207
  %210 = add i64 %194, 4
  %211 = trunc i64 %210 to i32
  %212 = call i32 @dma_p2p_opt(ptr %4, i64 %67, i32 %207, i32 %208, ptr %32, i64 %67, i32 %207, i32 %209, i1 false, i32 0, i32 %211)
  %213 = getelementptr i32, ptr %15, i64 %194
  store i32 %212, ptr %213, align 4
  %214 = mul i64 %189, %169
  %215 = add i64 %167, %214
  %216 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 %215, 2
  %217 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %216, i64 %137, 3, 0
  %218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %217, i64 %168, 4, 0
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %218, i64 %198, 3, 1
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, i64 %169, 4, 1
  br label %222

221:                                              ; preds = %188
  br label %222

222:                                              ; preds = %191, %221
  %223 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %185, %221 ], [ %205, %191 ]
  %224 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %186, %221 ], [ %220, %191 ]
  br label %225

225:                                              ; preds = %222
  %226 = mul i64 %184, -1
  %227 = add i64 %226, %54
  %228 = icmp slt i64 %227, %8
  %229 = select i1 %228, i64 %227, i64 %8
  %230 = sdiv i64 0, %9
  %231 = srem i64 %230, 3
  %232 = icmp slt i64 %137, %9
  %233 = select i1 %232, i64 %137, i64 %9
  %234 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, 0
  %235 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, 1
  %236 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, 2
  %237 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, 4, 0
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %234, 0
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, ptr %235, 1
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 %236, 2
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, i64 %233, 3, 0
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, i64 %237, 4, 0
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, i64 %67, 3, 1
  %244 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, i64 1, 4, 1
  %245 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, 0
  %246 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, 1
  %247 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, 2
  %248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, 4, 0
  %249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, 4, 1
  %250 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %245, 0
  %251 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %250, ptr %246, 1
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %251, i64 %247, 2
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %252, i64 %233, 3, 0
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, i64 %248, 4, 0
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 %229, 3, 1
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, i64 %249, 4, 1
  %257 = mul i64 %231, %38
  %258 = add i64 %257, 0
  %259 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %37, 0
  %260 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %259, ptr %37, 1
  %261 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %260, i64 %258, 2
  %262 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %261, i64 %233, 3, 0
  %263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %262, i64 %8, 4, 0
  %264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %263, i64 %229, 3, 1
  %265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %264, i64 1, 4, 1
  %266 = trunc i64 %229 to i32
  %267 = trunc i64 %248 to i32
  %268 = mul i32 %266, 4
  %269 = mul i32 %267, 4
  %270 = sub i32 %269, %268
  %271 = sub i32 %159, %268
  %272 = add i64 %231, 6
  %273 = trunc i64 %272 to i32
  %274 = call i32 @dma_p2p_opt(ptr %246, i64 %233, i32 %268, i32 %270, ptr %37, i64 %233, i32 %268, i32 %271, i1 false, i32 0, i32 %273)
  %275 = getelementptr i32, ptr %14, i64 %231
  store i32 %274, ptr %275, align 4
  %276 = sub i64 %184, %56
  %277 = sdiv i64 %276, %59
  %278 = srem i64 %277, 2
  %279 = getelementptr i32, ptr %15, i64 %278
  %280 = load i32, ptr %279, align 4
  call void @dma_wait_p2p(i32 %280)
  br label %281

281:                                              ; preds = %443, %225
  %282 = phi i64 [ %288, %443 ], [ 0, %225 ]
  %283 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %324, %443 ], [ %244, %225 ]
  %284 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %325, %443 ], [ %256, %225 ]
  %285 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %326, %443 ], [ %265, %225 ]
  %286 = icmp slt i64 %282, %137
  br i1 %286, label %287, label %446

287:                                              ; preds = %281
  %288 = add i64 %282, %9
  %289 = icmp slt i64 %288, %137
  br i1 %289, label %290, label %322

290:                                              ; preds = %287
  %291 = sdiv i64 %288, %9
  %292 = srem i64 %291, 3
  %293 = mul i64 %288, -1
  %294 = add i64 %293, %137
  %295 = icmp slt i64 %294, %9
  %296 = select i1 %295, i64 %294, i64 %9
  %297 = mul i64 %288, %237
  %298 = add i64 %236, %297
  %299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 %298, 2
  %300 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, i64 %296, 3, 0
  %301 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %300, i64 %237, 4, 0
  %302 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, i64 %67, 3, 1
  %303 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, i64 1, 4, 1
  %304 = mul i64 %288, %248
  %305 = add i64 %247, %304
  %306 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %251, i64 %305, 2
  %307 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %306, i64 %296, 3, 0
  %308 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %307, i64 %248, 4, 0
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, i64 %229, 3, 1
  %310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %309, i64 %249, 4, 1
  %311 = mul i64 %292, %38
  %312 = add i64 %311, 0
  %313 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %260, i64 %312, 2
  %314 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, i64 %296, 3, 0
  %315 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %314, i64 %8, 4, 0
  %316 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, i64 %229, 3, 1
  %317 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %316, i64 1, 4, 1
  %318 = add i64 %292, 6
  %319 = trunc i64 %318 to i32
  %320 = call i32 @dma_p2p_opt(ptr %246, i64 %296, i32 %268, i32 %270, ptr %37, i64 %296, i32 %268, i32 %271, i1 false, i32 0, i32 %319)
  %321 = getelementptr i32, ptr %14, i64 %292
  store i32 %320, ptr %321, align 4
  br label %323

322:                                              ; preds = %287
  br label %323

323:                                              ; preds = %290, %322
  %324 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %283, %322 ], [ %303, %290 ]
  %325 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %284, %322 ], [ %310, %290 ]
  %326 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %285, %322 ], [ %317, %290 ]
  br label %327

327:                                              ; preds = %323
  %328 = mul i64 %282, -1
  %329 = add i64 %328, %137
  %330 = icmp slt i64 %329, %9
  %331 = select i1 %330, i64 %329, i64 %9
  %332 = sdiv i64 0, %10
  %333 = srem i64 %332, 2
  %334 = icmp slt i64 %331, %10
  %335 = select i1 %334, i64 %331, i64 %10
  %336 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 1
  %337 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 4, 0
  %338 = mul i64 %333, %43
  %339 = add i64 %338, 0
  %340 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %42, 0
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %340, ptr %42, 1
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, i64 %339, 2
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 %335, 3, 0
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 %6, 4, 0
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 %67, 3, 1
  %346 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %345, i64 1, 4, 1
  %347 = trunc i64 %337 to i32
  %348 = mul i32 %347, 4
  %349 = sub i32 %348, %81
  %350 = trunc i64 %6 to i32
  %351 = mul i32 %350, 4
  %352 = sub i32 %351, %81
  %353 = trunc i64 %333 to i32
  %354 = call i32 @dma_p2p_opt(ptr %336, i64 %335, i32 %81, i32 %349, ptr %42, i64 %335, i32 %81, i32 %352, i1 false, i32 0, i32 %353)
  %355 = getelementptr i32, ptr %13, i64 %333
  store i32 %354, ptr %355, align 4
  %356 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 0
  %357 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 1
  %358 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 2
  %359 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 4, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %356, 0
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, ptr %357, 1
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 %358, 2
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %362, i64 %335, 3, 0
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, i64 %359, 4, 0
  %365 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %364, i64 %229, 3, 1
  %366 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, i64 1, 4, 1
  %367 = sdiv i64 %282, %9
  %368 = srem i64 %367, 3
  %369 = getelementptr i32, ptr %14, i64 %368
  %370 = load i32, ptr %369, align 4
  call void @dma_wait_p2p(i32 %370)
  br label %371

371:                                              ; preds = %407, %327
  %372 = phi i64 [ %377, %407 ], [ 0, %327 ]
  %373 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %405, %407 ], [ %346, %327 ]
  %374 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %406, %407 ], [ %366, %327 ]
  %375 = icmp slt i64 %372, %331
  br i1 %375, label %376, label %416

376:                                              ; preds = %371
  %377 = add i64 %372, %10
  %378 = icmp slt i64 %377, %331
  br i1 %378, label %379, label %403

379:                                              ; preds = %376
  %380 = sdiv i64 %377, %10
  %381 = srem i64 %380, 2
  %382 = mul i64 %377, -1
  %383 = add i64 %382, %331
  %384 = icmp slt i64 %383, %10
  %385 = select i1 %384, i64 %383, i64 %10
  %386 = mul i64 %381, %43
  %387 = add i64 %386, 0
  %388 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, i64 %387, 2
  %389 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %388, i64 %385, 3, 0
  %390 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %389, i64 %6, 4, 0
  %391 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, i64 %67, 3, 1
  %392 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %391, i64 1, 4, 1
  %393 = trunc i64 %381 to i32
  %394 = call i32 @dma_p2p_opt(ptr %336, i64 %385, i32 %81, i32 %349, ptr %42, i64 %385, i32 %81, i32 %352, i1 false, i32 0, i32 %393)
  %395 = getelementptr i32, ptr %13, i64 %381
  store i32 %394, ptr %395, align 4
  %396 = mul i64 %377, %359
  %397 = add i64 %358, %396
  %398 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 %397, 2
  %399 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %398, i64 %385, 3, 0
  %400 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %399, i64 %359, 4, 0
  %401 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %400, i64 %229, 3, 1
  %402 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %401, i64 1, 4, 1
  br label %404

403:                                              ; preds = %376
  br label %404

404:                                              ; preds = %379, %403
  %405 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %373, %403 ], [ %392, %379 ]
  %406 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %374, %403 ], [ %402, %379 ]
  br label %407

407:                                              ; preds = %404
  %408 = sdiv i64 %372, %10
  %409 = srem i64 %408, 2
  %410 = getelementptr i32, ptr %13, i64 %409
  %411 = load i32, ptr %410, align 4
  call void @dma_wait_p2p(i32 %411)
  %412 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, 1
  %413 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 1
  %414 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, 1
  %415 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, 3, 1
  call void @matmul_micro_kenrel(ptr %412, ptr %413, ptr %414, i64 %415, i64 %6, i64 %8)
  br label %371

416:                                              ; preds = %371
  %417 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 3, 0
  %418 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %285, 3, 1
  %419 = trunc i64 %418 to i32
  %420 = trunc i64 %359 to i32
  %421 = mul i32 %419, 4
  %422 = mul i32 %420, 4
  %423 = sub i32 %422, %421
  %424 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %284, 1
  %425 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %284, 3, 0
  %426 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %284, 3, 1
  %427 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %284, 4, 0
  %428 = trunc i64 %426 to i32
  %429 = trunc i64 %427 to i32
  %430 = mul i32 %428, 4
  %431 = mul i32 %429, 4
  %432 = sub i32 %431, %430
  %433 = srem i64 %367, 2
  %434 = add i64 %433, 8
  %435 = trunc i64 %434 to i32
  %436 = icmp ne i64 %282, 0
  br i1 %436, label %437, label %443

437:                                              ; preds = %416
  %438 = sub i64 %282, %9
  %439 = sdiv i64 %438, %9
  %440 = srem i64 %439, 3
  %441 = getelementptr i32, ptr %12, i64 %440
  %442 = load i32, ptr %441, align 4
  call void @dma_wait_p2p(i32 %442)
  br label %443

443:                                              ; preds = %437, %416
  %444 = call i32 @dma_p2p_opt(ptr %357, i64 %417, i32 %421, i32 %423, ptr %424, i64 %425, i32 %430, i32 %432, i1 false, i32 0, i32 %435)
  %445 = getelementptr i32, ptr %12, i64 %368
  store i32 %444, ptr %445, align 4
  br label %281

446:                                              ; preds = %281
  %447 = sub i64 %9, 1
  %448 = add i64 %137, %447
  %449 = sdiv i64 %448, %9
  %450 = sub i64 %449, 1
  %451 = srem i64 %450, 3
  %452 = getelementptr i32, ptr %12, i64 %451
  %453 = load i32, ptr %452, align 4
  call void @dma_wait_p2p(i32 %453)
  br label %183

454:                                              ; preds = %183
  br label %96

455:                                              ; preds = %96
  %456 = add i64 %61, %6
  br label %60

457:                                              ; preds = %60
  %458 = call i32 @vector_free(ptr %32)
  %459 = call i32 @vector_free(ptr %37)
  %460 = call i32 @scalar_free(ptr %42)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
