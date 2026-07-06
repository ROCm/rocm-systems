// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "multikernel_indirect_branch_common.h"

/// Kernel that calls the shared helper multiple times from one kernel body.
///
/// The second call consumes the first call's result so the compiler cannot turn
/// the calls into independent constant expressions.  The output store gives the
/// device-side return values an observable use in the final code object.
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

/// Second kernel that reaches the same helper body from a different descriptor.
extern "C" __global__ void multikernel_indirect_branch_b(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] + (idx << 1);
  output[idx] = multikernel_indirect_branch_shared_block(base, 41u);
}

/// Third kernel sharing the helper body.
extern "C" __global__ void multikernel_indirect_branch_c(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] ^ (idx * 3u);
  output[idx] = multikernel_indirect_branch_shared_block(base, 53u);
}

/// Fourth kernel sharing the helper body.
extern "C" __global__ void multikernel_indirect_branch_d(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] + (idx ^ 0x5a5a5a5au);
  output[idx] = multikernel_indirect_branch_shared_block(base, 67u);
}

/// Fifth kernel sharing the helper body.
extern "C" __global__ void multikernel_indirect_branch_e(const unsigned *__restrict__ input,
                                                         unsigned *__restrict__ output,
                                                         unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned base = input[idx] ^ (idx + 0x10203040u);
  output[idx] = multikernel_indirect_branch_shared_block(base, 79u);
}

/// Kernel with `s_setpc_b64` coverage unrelated to the shared helper.
extern "C" __global__ void multikernel_indirect_branch_setpc(const unsigned *__restrict__ input,
                                                             unsigned *__restrict__ output,
                                                             unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned value = input[idx] + idx;
  if (value == 0x31415926u)
    RJ_STATIC_SETPC_ISLAND(4, 5);
  if (value == 0x27182818u)
    RJ_STATIC_SETPC_ISLAND(6, 7);
  if (value == 0x16180339u)
    RJ_STATIC_SETPC_ISLAND(8, 9);

  output[idx] = value ^ 0xa5a5a5a5u;
}

/// Kernel with direct `s_call_b64` coverage unrelated to the shared helper.
extern "C" __global__ void multikernel_indirect_branch_scall(const unsigned *__restrict__ input,
                                                             unsigned *__restrict__ output,
                                                             unsigned n) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const unsigned value = input[idx] ^ (idx * 11u);
  if (value == 0x0badc0deu)
    RJ_STATIC_SCALL_ISLAND(20, 21);
  if (value == 0x00c0ffeeu)
    RJ_STATIC_SCALL_ISLAND(22, 23);
  if (value == 0x1234abcdu)
    RJ_STATIC_SCALL_ISLAND(24, 25);

  output[idx] = value + 1u;
}
