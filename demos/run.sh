# ../build/bin/mtir-opt test_one_shot.mlir \
#     -one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops" \
#     -debug-only="one-shot-analysis"

# mlir-opt test_one_shot.mlir -debug-only="one-shot-analysis" --mlir-print-ir-after-all

# ../build/bin/mtir-opt bufferize_issue.mlir \
#     -one-shot-bufferize="bufferize-function-boundaries=true"
#     -cse -canonicalize -debug-only="one-shot-analysis"

# ../build/bin/mtir-opt matmul_elemwise.mlir \ 
#     -one-shot-bufferize="bufferize-function-boundaries=true" \
#     -cse -canonicalize

../build/bin/mtir-opt matmul_elemwise.mlir \
    -mtfusion-predefined-tiling="fusion-mode=MIX_CV tiling-seq={-axis=k},{-axis=m -copy-mat=A -copy-dst=GSM},{-axis=n -nthreads=8 -copy-mat=B copy-dst=AM},{-axis=m -copy-mat=C -copy-dst=AM},{-axis=m -copy-mat=A -copy-dst=SM}" \
    -cse -canonicalize -cse \
    -one-shot-bufferize="bufferize-function-boundaries=true" \
    -cse -canonicalize \
    -mtfusion-buffer-normalize \
    -cse -canonicalize \
    -mtfusion-thread-parallelization \
    -mtfusion-dual-pipelining \
    -loop-invariant-code-motion \
    -cse -canonicalize -cse \
    -mtfusion-multi-buffering="enable-post-store=false" \
    -cse -canonicalize -cse \
    -fuse-elemwise-ops \
    -cse -canonicalize -cse \
    -lowering-device-func-args="remove-return-value=true" \
    -cse -canonicalize -cse \
    -convert-mtfusion-to-mtivm \
    -loop-invariant-code-motion \
    -cse -canonicalize -cse \
    # > mtivm_mlir_18_tmp.mlir
    
    # -mtfusion-buffer-normalize \
    # -cse -canonicalize \
    # -loop-invariant-code-motion \
    # -cse -canonicalize \
    # -mtfusion-multi-buffering="enable-post-store=false" \
    # -cse -canonicalize \
    # -cse -canonicalize
    # -mtfusion-buffer-normalize \
    # -cse -canonicalize \
    # -redundant-extract-removal \
    # -cse -canonicalize \
    # -mtfusion-thread-parallelization \
    # -cse -canonicalize \
    # -lowering-device-func-args="remove-return-value=true" \
    # -cse -canonicalize \
    # -mtfusion-multi-buffering="enable-post-store=false" \
    # -cse -canonicalize \
    # -convert-mtfusion-to-mtivm="post-multi-buffering=true" \
    # -cse -canonicalize \
    # > mtivm_mlir_18_tmp.mlir

# ../../llvm_17_install/bin/mlir-opt mtivm_mlir_18_tmp.mlir \
#     -expand-strided-metadata \
#     -cse -canonicalize \
#     -lower-affine \
#     -cse -canonicalize \
#     -convert-scf-to-cf \
#     -convert-func-to-llvm="use-bare-ptr-memref-call-conv=true" \
#     -finalize-memref-to-llvm \
#     -cse -canonicalize \
#     -convert-cf-to-llvm \
#     -convert-index-to-llvm \
#     -convert-arith-to-llvm \
#     -cse -canonicalize \
#     -cse -canonicalize > sgemm_mtir_kernel.mlir

# ../../llvm_17_install/bin/mlir-translate sgemm_mtir_kernel.mlir \
#     --mlir-to-llvmir \
#     -o sgemm_mtir_kernel.ll

# ../../llvm_17_install/bin/opt sgemm_mtir_kernel.ll \
#     -O2 \
#     -S -o sgemm_mtir_kernel.ll

# sh add_gsm_global.sh
    
# cp sgemm_mtir_kernel.ll ../../sgemm_mlir_codgen/device_code/



    # -mtfusion-multi-buffering="enable-post-store=false" \
    # -cse -canonicalize \
    # -convert-mtfusion-to-mtivm="post-multi-buffering=true" \
    # -cse -canonicalize \
    # -expand-strided-metadata \
    # -cse -canonicalize \


    # -convert-mtfusion-to-mtivm="post-multi-buffering=true" \
    # -cse -canonicalize
    
    
    
    # -lowering-device-func-args

    # -mtfusion-thread-parallelization
    # -one-shot-bufferize="bufferize-function-boundaries" \
    # -cse -canonicalize
    # -mtfusion-auto-schedule
    # -mlir-print-ir-after-all
    
## Available Passes:
# -convert-generic-to-named \
# -mtfusion-normalize-ops \
# -legalize-bool \
# -mtfusion-simplify-ops \
# -mtfusion-inline-brc \
# -mtfusion-infer-func-fusion-kind \
# -mtfusion-fuse-ops="always-inline=false move-out-to-param=true output-mode=multi fusion-mode=PURE_ELEMWISE max-horizontal-fusion-size=-1" \
# -mtfusion-single-op-outline \
# -mtfusion-cache-io \
# -mtfusion-auto-schedule \
# -mtfusion-tensor-results-to-out-params \
# -decompose-multi \
# -mtfusion-reorder-ops \
# -mtfusion-constantize-tiling-data \
# -mtfusion-pack-tiling-data \
# -loop-canonicalize \
# -redundant-copy-removal \



# ../build/bin/mtir-opt multi-buffering-notes.mlir \
#     -canonicalize \
#     -mtfusion-multi-buffering

