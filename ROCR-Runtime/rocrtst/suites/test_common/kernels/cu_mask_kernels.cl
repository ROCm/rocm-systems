#define GETREG_IMMED(SIZE, OFFSET, REG) ((SIZE-1)<<11)|(OFFSET<<6)|REG

#if ROCRTST_GPU < 0x1000
  #define HW_ID_CU_ID_OFFSET 8
  #define HW_ID 4
  #if (ROCRTST_GPU == 0x908) || (ROCRTST_GPU == 0x90a) || (ROCRTST_GPU == 0x940)
    #define HW_ID_CU_ID_SIZE 8
  #else
    #define HW_ID_CU_ID_SIZE 7
  #endif
#else
  #define HW_ID_CU_ID_OFFSET 9 //Skips first bit of SIMD ID, could be wrong.
  #define HW_ID 23
  #define HW_ID_CU_ID_SIZE 10
#endif

__kernel void get_hw_id(__global uint* hw_ids) {
  uint idx = get_global_id(0);
  hw_ids[idx] = __builtin_amdgcn_s_getreg(GETREG_IMMED(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
}
