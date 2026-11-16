// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

// The compiler barrier keeps the ordinary LDS accesses observable without
// adding guest synchronization. Volatile generic pointers can lower to FLAT.
extern "C" __global__ void lds_producer_consumer(unsigned input, unsigned *output) {
  __shared__ unsigned words[32];
  const unsigned id = __builtin_amdgcn_workitem_id_x();
  const unsigned wave = id / warpSize;
  asm volatile("" ::: "memory");
#if CONSAN_CASE == 1
  if (id == 0)
    words[0] = input;
  asm volatile("" ::: "memory");
  if (id == warpSize)
    output[0] = words[0];
#elif CONSAN_CASE == 2
  if (id % warpSize == 0) {
    words[wave] = input + wave;
    asm volatile("" ::: "memory");
    output[wave] = words[wave];
  }
#else
  // Keep the LDS pointer opaque to prevent folding an unspecified initial
  // value to undef. Its address-space type keeps the load native LDS.
  using LdsWord = unsigned __attribute__((address_space(3)));
  auto *local = (LdsWord *)words;
  asm volatile("" : "+v"(local));
  if (id % warpSize == 0)
    output[wave] = *local;
#endif
}
