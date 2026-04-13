// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct-to-LDS buffer load on gfx950: load to LDS, read back, compute,
// write to output.

#include <hip/hip_runtime.h>
#include <cstdint>

using i32x4 = int __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;

#if defined(__gfx950__)
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr, int size, int voffset, int soffset,
    int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
#endif

struct buffer_resource {
  uint64_t ptr;
  uint32_t range;
  uint32_t config;
};

__device__ __attribute__((always_inline)) i32x4
make_srsrc(const void *ptr, uint32_t range_bytes) {
  buffer_resource rsrc = {reinterpret_cast<uint64_t>(ptr), range_bytes,
                          0x110000};
  return *reinterpret_cast<const i32x4 *>(&rsrc);
}

constexpr int N = 256;

// DTL intrinsic: load to LDS, read back, add 1, write to output.
extern "C" __global__ void dtl_load(const float *input, float *output) {
#if defined(__gfx950__)
  __shared__ float lds_buf[N];

  uint32_t tid = threadIdx.x;

  i32x4 srsrc = make_srsrc(input, N * sizeof(float));
  as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(&lds_buf[0]);

  llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, /*size=*/4,
                                  /*voffset=*/tid * 4,
                                  /*soffset=*/0, /*offset=*/0, /*aux=*/0);

  // Missing s_waitcnt vmcnt(0) -- deliberate race.

  // Read from LDS and compute.
  float val = lds_buf[tid];
  output[tid] = val + 1.0f;
#endif
}
