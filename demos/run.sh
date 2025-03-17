
../build/bin/mtas-opt matmul_micro_kernel.mlir \
    -split-matmul \
    -lower-kernel-arguments \
    -cse -canonicalize -cse \
    -tile-dynamic-dims \
    -cse -canonicalize \
    -loop-unrolling \
    -lower-matmul-to-fma \
    -cse -canonicalize -cse \
    -fold-memref-alias-ops \
    -convert-memref-to-ptr \
    -expand-strided-metadata \
    -finalize-memref-to-llvm \
    -lower-affine \
    -cse -canonicalize -cse \
    -loop-invariant-code-motion \
    -cse -canonicalize -cse \
    -cast-ptr-to-int64 \
    -loop-folding \
    -fold-register-alloca \
    -cse -canonicalize -cse \
    -allocate-offset-registers \
    -cse -canonicalize -cse \
    
    # -cse -canonicalize -cse \

    # -lower-load-and-store-memref-to-ptr \
    # -lower-matmul-to-fma \
    # -cse -canonicalize \
    # -cse -canonicalize -cse \
    # -fold-memref-alias-ops \

    
    # -cse -canonicalize \

    
  