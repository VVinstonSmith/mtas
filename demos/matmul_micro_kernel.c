#include <compiler/m3000.h>
#include "hthread_device.h"

void matmul_micro_kernel(float* src_a, lvector float* src_b, lvector float* dst_c, const long k_size){
    __asm__ __volatile__(
// [0]
"    SMVAGA.M1         %0, AR10                    \t\n"
// [1]
"    SNOP              1                           \t\n"
// [2]
"    SMVAGA.M1         %2, AR0                     \t\n"
"|   SLDW              *AR10, R28                  \t\n"
// [3]
"    SMVAGA.M1         %1, AR1                     \t\n"
"|   SLDW              *+AR10[256], R27            \t\n"
// [4]
"    SMOVI.M1          1024, R30                   \t\n"
"|   SLDW              *+AR10[512], R26            \t\n"
"|   VLDDW             *AR0, VR43:VR42             \t\n"
"|   VLDDW             *+AR0[16], VR41:VR40        \t\n"
// [5]
"    SMVAGA.M1         R30, OR8                    \t\n"
"|   SMOVI.M2          1280, R31                   \t\n"
"|   VLDDW             *+AR1[16], VR15:VR14        \t\n"
"|   VLDDW             *AR1, VR17:VR16             \t\n"
// [6]
"    SMVAGA.M1         R31, OR9                    \t\n"
"|   SLDW              *+AR10[768], R9             \t\n"
"|   VLDDW             *+AR0[32], VR39:VR38        \t\n"
// [7]
"    SLDW              *+AR10[OR8], R8             \t\n"
"|   VLDDW             *+AR0[64], VR35:VR34        \t\n"
"|   VLDDW             *+AR0[48], VR37:VR36        \t\n"
// [8]
"    SLDW              *+AR10[OR9], R7             \t\n"
"|   VLDDW             *+AR0[80], VR33:VR32        \t\n"
// [9]
"    SVBCAST.M1        R28, VR19                   \t\n"
"|   VLDDW             *+AR0[96], VR31:VR30        \t\n"
"|   VLDDW             *+AR0[112], VR29:VR28       \t\n"
// [10]
"    SVBCAST.M1        R27, VR13                   \t\n"
"|   VLDDW             *+AR0[128], VR27:VR26       \t\n"
// [11]
"    SVBCAST.M1        R26, VR11                   \t\n"
"|   VLDDW             *+AR0[160], VR23:VR22       \t\n"
"|   VLDDW             *+AR0[144], VR25:VR24       \t\n"
// [12]
"    SMOVI.M1          0, R44                      \t\n"
"|   SMOVI.M2          2, R42                      \t\n"
"|   VLDDW             *+AR0[176], VR21:VR20       \t\n"
// [13]
"    SVBCAST.M1        R9, VR9                     \t\n"
"|   SMOVI.M2          512, R43                    \t\n"
"|   SMOVI             8, R29                      \t\n"
"|   VBALE2            VR19, VR19, VR18            \t\n"
"|   VLDDW             *+AR1[32], VR3:VR2          \t\n"
"|   VLDDW             *+AR1[48], VR1:VR0          \t\n"
"loop_k: \t\n"
// [0]
"    SVBCAST.M1        R8, VR7                     \t\n"
"|   VFMULAS32.M1      VR18, VR16, VR42, VR42      \t\n"
"|   VFMULAS32.M2      VR18, VR17, VR43, VR43      \t\n"
"|   VFMULAS32.M3      VR18, VR14, VR40, VR40      \t\n"
"|   VBALE2            VR13, VR13, VR12            \t\n"
// [1]
"    SVBCAST.M1        R7, VR5                     \t\n"
"|   VFMULAS32.M1      VR18, VR15, VR41, VR41      \t\n"
"|   VFMULAS32.M2      VR12, VR16, VR38, VR38      \t\n"
"|   VFMULAS32.M3      VR12, VR17, VR39, VR39      \t\n"
"|   VBALE2            VR11, VR11, VR10            \t\n"
// [2]
"    SADDA             R29, AR10, AR10             \t\n"
"|   VFMULAS32.M1      VR12, VR15, VR37, VR37      \t\n"
"|   VFMULAS32.M2      VR12, VR14, VR36, VR36      \t\n"
"|   VFMULAS32.M3      VR10, VR16, VR34, VR34      \t\n"
// [3]
"    VFMULAS32.M1      VR10, VR17, VR35, VR35      \t\n"
"|   VFMULAS32.M2      VR10, VR14, VR32, VR32      \t\n"
"|   VFMULAS32.M3      VR10, VR15, VR33, VR33      \t\n"
"|   VBALE2            VR9, VR9, VR8               \t\n"
// [4]
"    SADDA             R30, AR1, AR1               \t\n"
"|   SLDW              *AR10, R28                  \t\n"
"|   VFMULAS32.M1      VR8, VR16, VR30, VR30       \t\n"
"|   VFMULAS32.M2      VR8, VR17, VR31, VR31       \t\n"
"|   VFMULAS32.M3      VR8, VR14, VR28, VR28       \t\n"
"|   VBALE2            VR7, VR7, VR6               \t\n"
// [5]
"    SLDW              *+AR10[256], R27            \t\n"
"|   VFMULAS32.M1      VR8, VR15, VR29, VR29       \t\n"
"|   VFMULAS32.M2      VR6, VR16, VR26, VR26       \t\n"
"|   VFMULAS32.M3      VR6, VR17, VR27, VR27       \t\n"
"|   VBALE2            VR5, VR5, VR4               \t\n"
// [6]
"    SLDW              *+AR10[512], R26            \t\n"
"|   VFMULAS32.M1      VR6, VR14, VR24, VR24       \t\n"
"|   VFMULAS32.M2      VR6, VR15, VR25, VR25       \t\n"
"|   VFMULAS32.M3      VR4, VR16, VR22, VR22       \t\n"
// [7]
"    SADD.M1           R44, R42, R44               \t\n"
"|   VFMULAS32.M1      VR4, VR17, VR23, VR23       \t\n"
"|   VFMULAS32.M2      VR4, VR14, VR20, VR20       \t\n"
"|   VFMULAS32.M3      VR4, VR15, VR21, VR21       \t\n"
"|   VBALE2h           VR19, VR19, VR18            \t\n"
"|   VLDDW             *+AR1[16], VR15:VR14        \t\n"
"|   VLDDW             *AR1, VR17:VR16             \t\n"
// [8]
"    SLT               R44, R43, R0                \t\n"
"|   SLDW              *+AR10[768], R9             \t\n"
"|   VFMULAS32.M1      VR18, VR2, VR42, VR42       \t\n"
"|   VFMULAS32.M2      VR18, VR3, VR43, VR43       \t\n"
"|   VFMULAS32.M3      VR18, VR0, VR40, VR40       \t\n"
"|   VBALE2h           VR13, VR13, VR12            \t\n"
// [9]
"    SLDW              *+AR10[OR8], R8             \t\n"
"|   [R0] SBR          loop_k                      \t\n"
"|   VFMULAS32.M1      VR18, VR1, VR41, VR41       \t\n"
"|   VFMULAS32.M2      VR12, VR2, VR38, VR38       \t\n"
"|   VFMULAS32.M3      VR12, VR3, VR39, VR39       \t\n"
"|   VBALE2h           VR11, VR11, VR10            \t\n"
// [10]
"    SLDW              *+AR10[OR9], R7             \t\n"
"|   VFMULAS32.M1      VR12, VR1, VR37, VR37       \t\n"
"|   VFMULAS32.M2      VR10, VR2, VR34, VR34       \t\n"
"|   VFMULAS32.M3      VR12, VR0, VR36, VR36       \t\n"
// [11]
"    SVBCAST.M1        R28, VR19                   \t\n"
"|   VFMULAS32.M1      VR10, VR3, VR35, VR35       \t\n"
"|   VFMULAS32.M2      VR10, VR0, VR32, VR32       \t\n"
"|   VFMULAS32.M3      VR10, VR1, VR33, VR33       \t\n"
"|   VBALE2h           VR9, VR9, VR8               \t\n"
// [12]
"    SVBCAST.M1        R27, VR13                   \t\n"
"|   VFMULAS32.M1      VR8, VR2, VR30, VR30        \t\n"
"|   VFMULAS32.M2      VR8, VR3, VR31, VR31        \t\n"
"|   VFMULAS32.M3      VR8, VR0, VR28, VR28        \t\n"
"|   VBALE2h           VR7, VR7, VR6               \t\n"
// [13]
"    SVBCAST.M1        R26, VR11                   \t\n"
"|   VFMULAS32.M1      VR8, VR1, VR29, VR29        \t\n"
"|   VFMULAS32.M2      VR6, VR2, VR26, VR26        \t\n"
"|   VFMULAS32.M3      VR6, VR3, VR27, VR27        \t\n"
"|   VBALE2h           VR5, VR5, VR4               \t\n"
// [14]
"    VFMULAS32.M1      VR6, VR0, VR24, VR24        \t\n"
"|   VFMULAS32.M2      VR6, VR1, VR25, VR25        \t\n"
"|   VFMULAS32.M3      VR4, VR2, VR22, VR22        \t\n"
// [15]
"    SVBCAST.M1        R9, VR9                     \t\n"
"|   VFMULAS32.M1      VR4, VR3, VR23, VR23        \t\n"
"|   VFMULAS32.M2      VR4, VR0, VR20, VR20        \t\n"
"|   VFMULAS32.M3      VR4, VR1, VR21, VR21        \t\n"
"|   VBALE2            VR19, VR19, VR18            \t\n"
"|   VLDDW             *+AR1[32], VR3:VR2          \t\n"
"|   VLDDW             *+AR1[48], VR1:VR0          \t\n"
// [0]
"    SBR               R63                         \t\n"
"|   VSTDW             VR39:VR38, *+AR0[32]        \t\n"
"|   VSTDW             VR41:VR40, *+AR0[16]        \t\n"
// [1]
"    VSTDW             VR43:VR42, *AR0             \t\n"
"|   VSTDW             VR37:VR36, *+AR0[48]        \t\n"
// [2]
"    VSTDW             VR35:VR34, *+AR0[64]        \t\n"
"|   VSTDW             VR33:VR32, *+AR0[80]        \t\n"
// [3]
"    VSTDW             VR31:VR30, *+AR0[96]        \t\n"
"|   VSTDW             VR27:VR26, *+AR0[128]       \t\n"
// [4]
"    VSTDW             VR29:VR28, *+AR0[112]       \t\n"
"|   VSTDW             VR25:VR24, *+AR0[144]       \t\n"
// [5]
"    VSTDW             VR21:VR20, *+AR0[176]       \t\n"
"|   VSTDW             VR23:VR22, *+AR0[160]       \t\n"
// [6]
"    SNOP              1                           \t\n"
    ::"r"(src_a), "r"(src_b), "r"(dst_c), "r"(k_size));
}
