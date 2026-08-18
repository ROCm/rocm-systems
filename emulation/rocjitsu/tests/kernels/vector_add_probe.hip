// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Register-padded vector_add for the DBI probe-CALL hardware smoke test
// (tests/dbi/hsa_dbi_nop_probe_test.cpp). Functionally identical to
// tests/kernels/vector_add.hip.
//
// Why a separate kernel: the no-op probe `rj_nop_probe` is compiled with the
// standard AMDGPU function ABI and returns through the link pair s[30:31]. The
// probe-call trampoline therefore writes/reads s[30:31], which the wave must
// actually own. A normal vector_add only allocates ~12 SGPRs, so s[30:31] lie
// outside the allocation and the call faults/hangs the GPU.
//
// Until the instrumentor grows the patched kernel's SGPR allocation itself
// (deferred to a follow-up PR), this fixture forces the allocation up to >=32
// SGPRs with an inline-asm clobber of s30/s31. The clobber emits no
// instructions and the registers are never read, so they stay dead at every
// anchor (the probe-call resource planner still accepts the patch) -- it only
// bumps the descriptor's .sgpr_count so s[30:31] are granted.

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float *__restrict__ A, const float *__restrict__ B,
                                      float *__restrict__ C, unsigned N) {
  // Force the SGPR allocation up through s31 without making them live.
  asm volatile("" ::: "s30", "s31");

  constexpr unsigned BLOCK_SIZE = 64;
  unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (gid < N)
    C[gid] = A[gid] + B[gid];
}
