// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Oversized static-LDS kernel used to validate rocjitsu's virtual-LDS sidecar.
///
/// The shared array is 108288 bytes, matching the size class of the Qwen/Tensile
/// gate-projection kernel that cannot fit in gfx942 hardware LDS.  The body uses
/// vectorized 16-byte LDS accesses at both low offsets and offsets above 64 KiB
/// so the translated sidecar must correctly remap wide DS reads/writes into the
/// per-workgroup global backing store.
extern "C" __global__ void virtual_lds_smoke(const unsigned *__restrict__ input,
                                             unsigned *__restrict__ output,
                                             unsigned workgroup_size) {
  constexpr unsigned kThreads = 256;
  constexpr unsigned kHighBase = 4096;   // 4096 * sizeof(uint4) == 64 KiB.
  constexpr unsigned kLdsVectors = 6768; // 6768 * sizeof(uint4) == 108288 bytes.
  __shared__ __align__(16) uint4 scratch[kLdsVectors];

  const unsigned tid = threadIdx.x;
  const unsigned gid = blockIdx.x * workgroup_size + tid;
  const unsigned seed = input[gid];

  uint4 low;
  low.x = seed ^ 0x10203040u;
  low.y = seed + 0x31415927u;
  low.z = seed ^ (tid * 0x01010101u);
  low.w = seed + gid;

  uint4 high;
  high.x = seed ^ 0xa5a50000u;
  high.y = seed + 0x27182818u;
  high.z = seed ^ (gid * 0x00010001u);
  high.w = ~seed;

  scratch[tid] = low;
  scratch[kHighBase + tid] = high;
  __syncthreads();

  // Cross-lane reads force the compiler to materialize the values from LDS
  // rather than forwarding each lane's own stores through registers.
  const uint4 from_low = scratch[(tid + 1u) & (kThreads - 1u)];
  const uint4 from_high = scratch[kHighBase + ((tid + 17u) & (kThreads - 1u))];

  const unsigned out = gid * 4u;
  output[out + 0] = from_low.x ^ from_high.w;
  output[out + 1] = from_low.y + from_high.z;
  output[out + 2] = from_low.z ^ from_high.y;
  output[out + 3] = from_low.w + from_high.x;
}
