// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "multikernel_indirect_branch_common.h"

/// First half of the multi-kernel fixture, built separately to check that the
/// full code object does not introduce extra CFG edges between independent
/// kernel regions.
extern "C" __global__ void multikernel_indirect_branch_a(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx];
  const unsigned first = multikernel_indirect_branch_shared_block(base + idx, 17u);
  const unsigned second = multikernel_indirect_branch_shared_block(first ^ base, 29u);
  output[idx] = first + second;
}

extern "C" __global__ void multikernel_indirect_branch_b(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] + (idx << 1);
  output[idx] = multikernel_indirect_branch_shared_block(base, 41u);
}

extern "C" __global__ void multikernel_indirect_branch_c(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] ^ (idx * 3u);
  output[idx] = multikernel_indirect_branch_shared_block(base, 53u);
}
