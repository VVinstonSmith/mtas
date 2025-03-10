../../llvm_17_install/bin/mlir-opt after_mtivm_conversion.mlir \
    -expand-strided-metadata \
    -lower-affine \
    -convert-scf-to-cf \
    -convert-func-to-llvm="use-bare-ptr-memref-call-conv=true" \
    -finalize-memref-to-llvm \
    -cse -canonicalize \
    -convert-cf-to-llvm \
    -convert-index-to-llvm \
    -convert-arith-to-llvm \
    -cse -canonicalize \
    -cse -canonicalize > mtivm_mlir_17_tmp.mlir

../../llvm_17_install/bin/mlir-translate mtivm_mlir_17_tmp.mlir \
    --mlir-to-llvmir \
    -o sgemm_mtir_kernel.ll

../../llvm_17_install/bin/opt sgemm_mtir_kernel.ll \
    -O2 \
    -S -o sgemm_mtir_kernel.ll
    
sh add_gsm_global.sh
    
cp sgemm_mtir_kernel.ll ../../sgemm_mlir_codgen/device_code/