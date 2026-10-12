// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Block reduction fixture for the DBI runtime hook tests
// (tests/dbi/). Unlike the vector_add fixtures it has loops, a barrier, LDS and
// a global atomic, so instrumenting one site per basic block exercises more
// than one straight-line block.
//
// Two source-level knobs shape the kernel's resources, because the
// instrumentor does not grow a kernel's allocation:
//
// - The inline-asm clobber of s30/s31 raises the SGPR allocation to >= 32 so a
//   called probe's link pair s[30:31] is granted on CDNA (see
//   vector_add_probe.hip). The v29 clobber raises the VGPR allocation to 32:
//   the record-writing probe uses up to v29, and no gate checks a probe's own
//   registers against the kernel's allocation.
// - `priv` is read with a runtime index, which keeps it in scratch and gives
//   the kernel a fixed private segment. Spilling around a probe call needs
//   one. An array over 32 dwords stays in scratch (the AMDGPU promote-alloca
//   limit); 64 keeps `i % 64` a mask.
//
// The tests launch it through raw HSA, which leaves the hidden kernel arguments
// that blockDim and gridDim read unset. So the block size is a constant (launch
// with kBlockSize threads per block) and the grid stride is an argument.
// Callers that compare outputs bit-exactly must choose inputs whose float sum
// is exact in any order: integer-valued, with a total below 2^24.

#include <hip/hip_runtime.h>

constexpr unsigned kBlockSize = 256;
constexpr unsigned kPrivateCount = 64;

extern "C" __global__ void block_reduce(const float *__restrict__ in, float *__restrict__ out,
                                        unsigned n, unsigned stride) {
  // Force the allocations up without making the registers live.
  asm volatile("" ::: "s30", "s31", "v29");

  __shared__ float partial[kBlockSize];
  float priv[kPrivateCount];
  for (unsigned k = 0; k < kPrivateCount; ++k)
    priv[k] = static_cast<float>(k);

  unsigned tid = threadIdx.x;
  float sum = 0.0f;
  for (unsigned i = blockIdx.x * kBlockSize + tid; i < n; i += stride)
    sum += in[i] * priv[i % kPrivateCount];
  partial[tid] = sum;
  __syncthreads();

  for (unsigned half = kBlockSize / 2; half > 0; half >>= 1) {
    if (tid < half)
      partial[tid] += partial[tid + half];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(out, partial[0]);
}
