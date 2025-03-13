
../build/bin/mtas-opt matmul_micro_kernel.mlir \
    -split-matmul \
    -tile-dynamic-dims \
    -fold-memref-alias-ops \
    -cse -canonicalize \
    -loop-unrolling \
    -cse -canonicalize \
    # -fold-memref-alias-ops \
    # -cse -canonicalize \
    # -lower-matmul-to-fma \
    # -fold-memref-alias-ops \

    
    # -cse -canonicalize \

    
  