
../build/bin/mtas-opt matmul_micro_kernel.mlir \
    -tile-dynamic-dims \
    -cse -canonicalize \
    -loop-unrolling="unrolling-factor=3" \
    -cse -canonicalize \

    
  