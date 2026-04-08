#!/usr/bin/env python3
"""
MLIR 矩阵乘法 IR 生成器
用法: python generate_mlir.py M N K [output_file]
"""

import sys

def generate_matmul_ir(M, N, K, output_file="input.mlir"):
    ir_template = f"""module {{
  func.func @matmul_only(
    %A: memref<{M}x{K}xf32> {{ftm.memory_level = #ftm.memory_level<sm>}}, 
    %B: memref<{K}x{N}xf32> {{ftm.memory_level = #ftm.memory_level<am>}}, 
    %C: memref<{M}x{N}xf32> {{ftm.memory_level = #ftm.memory_level<am>}})
  {{
    linalg.matmul {{ftm.unroll_loop_number = #ftm.unroll_loop_number<2>}} ins(%A, %B : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>) outs(%C : memref<{M}x{N}xf32>)
    return
  }}
}}"""
    
    with open(output_file, 'w') as f:
        f.write(ir_template)
    
    print(f"Generated {output_file} with dimensions: A({M}x{K}) * B({K}x{N}) = C({M}x{N})")

if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("Usage: python generate_mlir.py M K N [output_file]")
    #     print("Example: python generate_mlir.py 6 512 128")
    #     sys.exit(1)
    
    # M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    # output_file = sys.argv[4] if len(sys.argv) > 4 else "matmul_micro_kernel.mlir"

    M, K, N = 6, 512, 4 * 32
    output_file = "demos/matmul_micro_kernel.mlir"
    
    generate_matmul_ir(M, N, K, output_file)