/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// CE DDA AllGather local-copy kernel -- vectorized load/store edition.
//
// Self-contained device-only TU: compiled with full HIP (host+device) by the
// device linker pipeline (see cmake/DeviceLinker.cmake) so the fat binary is
// embedded in ce_local_copy.o. ce_coll.cc (main target, --offload-host-only)
// only forward-declares and calls ncclCeLaunchLocalCopyKernel() below, so it
// has no __global__ call site of its own and produces no undefined
// __hip_fatbin_<hash> reference.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "nccl.h"

// Pure byte copy: no reduction is involved, so the vector width only needs to
// respect src/dst/n alignment, not the collective's logical datatype. Same
// int4-vectorized-with-scalar-tail idiom as hierarchicalAGShuffle() in
// hierarchical_ag_shuffle.h.
template <typename Vec>
__device__ __forceinline__ void ceVectorCopyBody(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                                  size_t n) {
  size_t nVec = n / sizeof(Vec);
  const Vec* srcV = reinterpret_cast<const Vec*>(src);
  Vec* dstV = reinterpret_cast<Vec*>(dst);

  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  #pragma unroll
  for (size_t i = tid; i < nVec; i += stride) dstV[i] = srcV[i];

  // Tail bytes that don't fill a full sizeof(Vec) lane.
  size_t copied = nVec * sizeof(Vec);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    #pragma unroll
    for (size_t i = copied; i < n; i++) dst[i] = src[i];
  }
}

// vecShift selects the widest pack size that src, dst, and n all share:
// 4->int4 (16B), 3->int2 (8B), 2->int (4B), else scalar byte fallback.
__global__ __launch_bounds__(256) void ncclCeLocalCopyKernel(
  const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t n, int vecShift) {
  switch (vecShift) {
  case 4: ceVectorCopyBody<int4>(src, dst, n); break;
  case 3: ceVectorCopyBody<int2>(src, dst, n); break;
  case 2: ceVectorCopyBody<int>(src, dst, n); break;
  default: ceVectorCopyBody<uint8_t>(src, dst, n); break;
  }
}

// Host launcher -- external linkage, forward-declared and called from
// ce_coll.cc (ncclCeAllGatherPipelined's copy-back loop).
ncclResult_t ncclCeLaunchLocalCopyKernel(const void* src, void* dst, size_t n, hipStream_t stream) {
  if (n == 0) return ncclSuccess;

  uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  int vecShift = ((s | d | n) % 16 == 0) ? 4
               : ((s | d | n) % 8  == 0) ? 3
               : ((s | d | n) % 4  == 0) ? 2 : 1;
  static const int vecBytes[] = {1, 4, 8, 16};
  const int threadsPerBlock = 256;
  size_t nVec = n / vecBytes[vecShift - 1];
  int blocks = (int)std::min<size_t>(16, (nVec + threadsPerBlock - 1) / threadsPerBlock);
  blocks = std::max(blocks, 1);

  (void)hipGetLastError(); // clear any stale error before checking this launch
  ncclCeLocalCopyKernel<<<blocks, threadsPerBlock, 0, stream>>>((const uint8_t*)src, (uint8_t*)dst, n, vecShift);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) return ncclUnhandledCudaError;
  return ncclSuccess;
}
