// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

__device__ __forceinline__ void access_lds(unsigned input, unsigned *output) {
  __shared__ unsigned words[32];
  const unsigned id = __builtin_amdgcn_workitem_id_x();
  const unsigned wave = id / warpSize;
  asm volatile("" ::: "memory");
  if (id % warpSize == 0) {
    words[wave] = input + wave;
    asm volatile("" ::: "memory");
    output[wave] = words[wave];
  }
}

extern "C" __global__ void sel(unsigned input, unsigned *output) {
  access_lds(input, output);
}

#if CONSAN_NATIVE_AGPR == 3
extern "C" __global__ void sib() {}
#else
extern "C" __global__
#if CONSAN_NATIVE_AGPR == 2
    __launch_bounds__(64)
#endif
    void sib(unsigned input, unsigned *output) {
#if CONSAN_NATIVE_AGPR == 2
  // Keep the unified count above the ordinary VGPR namespace.
  asm volatile("v_mov_b32 v191, 0\n\t"
               "v_accvgpr_write_b32 a191, v191" ::: "v191", "a191");
#elif CONSAN_NATIVE_AGPR == 1
  // Force a VGPR before the aligned accumulator region in the unified count.
  asm volatile("v_mov_b32 v0, 0\n\t"
               "v_accvgpr_write_b32 a0, v0" ::: "v0", "a0");
#else
  access_lds(input, output);
#endif
}
#endif
