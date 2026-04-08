module {
  func.func @matmul_only(
    %A: memref<6x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
    %B: memref<512x128xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
    %C: memref<6x128xf32> {ftm.memory_level = #ftm.memory_level<am>})
  {
    linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x512xf32>, memref<512x128xf32>) outs(%C : memref<6x128xf32>)
    return
  }
}