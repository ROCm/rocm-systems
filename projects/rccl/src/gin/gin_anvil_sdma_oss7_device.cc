/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#if defined(__HIPCC__) || defined(__CUDACC__)

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gin_anvil {
namespace sdma {

// Device-side OSS7 toggle for SDMA packet selection (COPY_LINEAR_PHY_MI4 vs legacy).
__device__ int gin_anvil_sdma_oss7_enabled = 1;

// [GIN-CONN-CHECK] Init-time LSA signal connectivity self-test.
//
// The GIN Anvil-SDMA path requires cuMem VMM peer mappings, which RCCL only
// auto-enables on gfx1250 and which must be force-enabled (NCCL_CUMEM_ENABLE=1)
// on gfx950. On that arch the per-peer VMM import intermittently produces a
// silently-wrong mapping for one rank: that rank's cross-rank signal stores go
// nowhere, so the first collective spins forever on an unbounded waitSignal
// (~1 run in 6). These kernels exercise the exact coherent-fabric store path
// used by the small-message SignalInc so a broken mapping is detected -- and
// the job fails loudly -- at init instead of hanging mid-run.

// Each participating thread p stores `stamp` into peer p's test slot indexed by
// this rank (selfRank), through the same peer signal base addresses the real
// SignalInc path resolves.
__global__ void ginAnvilConnWriteKernel(uintptr_t* remoteAddrs, int nRanks, int selfRank,
                                        unsigned long long stamp) {
  int p = threadIdx.x;
  if (p >= nRanks) return;
  uintptr_t base = remoteAddrs[p];
  if (base == 0) return;
  unsigned long long* dst = reinterpret_cast<unsigned long long*>(base) + selfRank;
  __atomic_store_n(dst, stamp, __ATOMIC_RELAXED);
  __threadfence_system();
}

// Each thread x checks whether source rank x's store landed in our local test
// slot x. missing[x] = 1 means source x could not reach this rank.
__global__ void ginAnvilConnCheckKernel(unsigned long long* localSignals, int nRanks,
                                        unsigned long long stamp, int* missing) {
  int x = threadIdx.x;
  if (x >= nRanks) return;
  __threadfence_system();
  unsigned long long v = __atomic_load_n(localSignals + x, __ATOMIC_RELAXED);
  missing[x] = (v == stamp) ? 0 : 1;
}

}  // namespace sdma
}  // namespace gin_anvil

extern "C" int ginAnvilConnWrite(void* remoteAddrsDev, int nRanks, int selfRank,
                                 unsigned long long stamp) {
  hipLaunchKernelGGL(gin_anvil::sdma::ginAnvilConnWriteKernel, dim3(1), dim3(nRanks), 0, 0,
                     reinterpret_cast<uintptr_t*>(remoteAddrsDev), nRanks, selfRank, stamp);
  return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

extern "C" int ginAnvilConnCheck(void* localSignals, int nRanks, unsigned long long stamp,
                                 int* missingDev) {
  hipLaunchKernelGGL(gin_anvil::sdma::ginAnvilConnCheckKernel, dim3(1), dim3(nRanks), 0, 0,
                     reinterpret_cast<unsigned long long*>(localSignals), nRanks, stamp, missingDev);
  return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

#endif  // __HIPCC__ || __CUDACC__

#endif  // ENABLE_ROCSHMEM_GIN
