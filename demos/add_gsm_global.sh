#!/bin/bash

# 定义文件名和要插入的内容
file="sgemm_mtir_kernel.ll"
insert_string=", section \".gsm\""

# 检查文件是否存在
if [ ! -f "$file" ]; then
    echo "文件 $file 不存在！"
    exit 1
fi

# 使用 awk 在指定字符串后插入内容
# awk -v insert="$insert_string" '/@gsm_mem = internal global \[1572864 x float\] undef/ { print $0 insert; next }1' "$file" > temp_file
awk -v insert="$insert_string" '/@gsm_mem = external global \[1572864 x float\]/ { print $0 insert; next }1' "$file" > temp_file


# 检查是否成功修改
if [ $? -eq 0 ]; then
    # 将临时文件覆盖原文件
    mv temp_file "$file"
    echo "gsm修改完成！"
else
    echo "gsm修改失败！"
    rm temp_file
fi

#######################################################################

# # 定义文件名
# file="sgemm_mtir_kernel.ll"

# # 检查文件是否存在
# if [ ! -f "$file" ]; then
#     echo "文件 $file 不存在！"
#     exit 1
# fi

# # 要插入的内容
# # target_string='/@matmul_only_tiling_pointerized(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)'
# target_string='@matmul_only_tiling_pointerized'

# insert_string=" section \".global\""


# awk -v insert="$insert_string" '/@matmul_only_tiling_pointerized/ { print $0 insert; next }1' "$file" > temp_file


# # 检查是否成功修改
# if [ $? -eq 0 ]; then
#     # 将临时文件覆盖原文件
#     mv temp_file "$file"
#     echo "global修改完成！"
# else
#     echo "global修改失败！"
#     rm temp_file
# fi


#!/bin/bash

# 定义文件名
file="sgemm_mtir_kernel.ll"

# 检查文件是否存在
if [ ! -f "$file" ]; then
    echo "Error: File $file does not exist."
    exit 1
fi

# 使用 sed 替换字符串
sed -i 's/define void @matmul_only_tiling_pointerized(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {/define void @matmul_only_tiling_pointerized(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) section ".global" {/' "$file"

echo "global修改完成！"