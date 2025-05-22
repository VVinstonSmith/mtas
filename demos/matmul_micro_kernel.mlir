// module {
//   func.func @matmul_only(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
//   {
//     linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x?xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<?x3xvector<32xf32>> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x3xvector<32xf32>> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul ins(%A, %B : memref<6x?xf32>, memref<?x3xvector<32xf32>>) outs(%C : memref<6x3xvector<32xf32>>)
//     return
//   }
// }


// module {
//   func.func @matmul_only(
//     %A: memref<6x?xf32>,
//     %B: memref<?x?xf32>,
//     %C: memref<6x?xf32>,
//     %D: memref<?x?xf32>,
//     %E: memref<6x?xf32>)
//   {
    
//     linalg.matmul ins(%A, %B : memref<6x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
//     linalg.matmul ins(%C, %D : memref<6x?xf32>, memref<?x96xf32>) outs(%E : memref<6x96xf32>)
//     return
//   }
// }



// module {
//   func.func @matmul_only(
//     %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x64xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x64xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x512xf32>, memref<512x64xf32>) outs(%C : memref<3x64xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x512xf32>, memref<512x96xf32>) outs(%C : memref<3x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<1x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x288xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<1x288xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<1x512xf32>, memref<512x288xf32>) outs(%C : memref<1x288xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<1x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x576xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<1x576xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<1x256xf32>, memref<256x576xf32>) outs(%C : memref<1x576xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<1x336xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<336x576xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<1x576xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<1x336xf32>, memref<336x576xf32>) outs(%C : memref<1x576xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<2x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x288xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<2x288xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<2x256xf32>, memref<256x288xf32>) outs(%C : memref<2x288xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<2x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x288xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<2x288xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<2x512xf32>, memref<512x288xf32>) outs(%C : memref<2x288xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<2x400xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<400x480xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<2x480xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<2x400xf32>, memref<400x480xf32>) outs(%C : memref<2x480xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x128xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<128x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x128xf32>, memref<128x192xf32>) outs(%C : memref<3x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x256xf32>, memref<256x192xf32>) outs(%C : memref<3x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x512xf32>, memref<512x192xf32>) outs(%C : memref<3x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x224xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x224xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x512xf32>, memref<512x224xf32>) outs(%C : memref<3x224xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<3x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x352xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<3x352xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<3x512xf32>, memref<512x352xf32>) outs(%C : memref<3x352xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x256xf32>, memref<256x192xf32>) outs(%C : memref<4x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x512xf32>, memref<512x192xf32>) outs(%C : memref<4x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x500xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<500x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x500xf32>, memref<500x192xf32>) outs(%C : memref<4x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x256xf32>, memref<256x192xf32>) outs(%C : memref<4x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x256xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x256xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x512xf32>, memref<512x256xf32>) outs(%C : memref<4x256xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<4x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x288xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<4x288xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<4x512xf32>, memref<512x288xf32>) outs(%C : memref<4x288xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<5x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<5x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<5x256xf32>, memref<256x192xf32>) outs(%C : memref<5x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<5x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<5x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<5x512xf32>, memref<512x192xf32>) outs(%C : memref<5x192xf32>)
//     return
//   }
// }

module {
  func.func @matmul_only(
    %A: memref<6x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
    %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
    %C: memref<6x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
  {
    linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x256xf32>, memref<256x96xf32>) outs(%C : memref<6x96xf32>)
    return
  }
}

// module {
//   func.func @matmul_only(
//     %A: memref<6x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x128xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x128xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x256xf32>, memref<256x128xf32>) outs(%C : memref<6x128xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x256xf32>, memref<256x192xf32>) outs(%C : memref<6x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x512xf32>, memref<512x96xf32>) outs(%C : memref<6x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x128xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x128xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x512xf32>, memref<512x128xf32>) outs(%C : memref<6x128xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x160xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x160xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x512xf32>, memref<512x160xf32>) outs(%C : memref<6x160xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<6x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x192xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<6x192xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<6x512xf32>, memref<512x192xf32>) outs(%C : memref<6x192xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<7x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<7x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<7x256xf32>, memref<256x96xf32>) outs(%C : memref<7x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<7x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<7x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<7x512xf32>, memref<512x96xf32>) outs(%C : memref<7x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<8x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<8x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<8x256xf32>, memref<256x96xf32>) outs(%C : memref<8x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<9x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<9x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<9x256xf32>, memref<256x96xf32>) outs(%C : memref<9x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<9x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x128xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<9x128xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<9x512xf32>, memref<512x128xf32>) outs(%C : memref<9x128xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<10x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<10x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<10x256xf32>, memref<256x96xf32>) outs(%C : memref<10x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<10x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<10x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<10x512xf32>, memref<512x96xf32>) outs(%C : memref<10x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<11x256xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<256x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<11x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<11x256xf32>, memref<256x96xf32>) outs(%C : memref<11x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<11x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x96xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<11x96xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<11x512xf32>, memref<512x96xf32>) outs(%C : memref<11x96xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(
//     %A: memref<12x512xf32> {ftm.memory_level = #ftm.memory_level<sm>}, 
//     %B: memref<512x64xf32> {ftm.memory_level = #ftm.memory_level<am>}, 
//     %C: memref<12x64xf32> {ftm.memory_level = #ftm.memory_level<am>})
//   {
//     linalg.matmul {ftm.unroll_loop_number = #ftm.unroll_loop_number<2>} ins(%A, %B : memref<12x512xf32>, memref<512x64xf32>) outs(%C : memref<12x64xf32>)
//     return
//   }
// }

// module {
//   func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %0 : tensor<?x?xf32>
//   }
// }

// module {
//   func.func @matmul_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>}
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %0 : tensor<?x?xf32>
//   }
// }

// module {
//   func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> 
//   attributes {mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>}
//   {
//     %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
//     return %2 : tensor<?x?xf32>
//   }
// }
