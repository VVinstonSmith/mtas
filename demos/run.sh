
../build/bin/mtas-opt matmul_micro_kernel.mlir \
    -split-matmul \
    -cse -canonicalize -cse \
    -tile-linalg-dims \
    -cse -canonicalize \
    -loop-unrolling \
    -cse -canonicalize -cse \
    -lower-linalg-ops \
    -cse -canonicalize -cse \
    -lower-kernel-arguments \
    -fold-memref-alias-ops \
    -convert-memref-to-ptr \
    -expand-strided-metadata \
    -finalize-memref-to-llvm \
    -lower-affine \
    -cse -canonicalize -cse \
    -loop-invariant-code-motion \
    -cse -canonicalize -cse \
    -cast-ptr-to-int64 \
    -cse -canonicalize -cse \
    -loop-folding \
    -fold-register-alloca \
    -cse -canonicalize -cse \
    -allocate-offset-registers \
    -cse -canonicalize -cse \
    -loop-strength-reduce \
    
    # -cse -canonicalize -cse \

    # -lower-load-and-store-memref-to-ptr \
    # -lower-matmul-to-fma \
    # -cse -canonicalize \
    # -cse -canonicalize -cse \
    # -fold-memref-alias-ops \

    
    # -cse -canonicalize \

    
  