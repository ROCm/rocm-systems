/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#if defined(__HIPCC__) || defined(__CUDACC__)

#include "gin/gin_anvil_conn_check.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gin_anvil {
namespace conn_check {

// HIP blockDim.x must cover nRanks threads (one per peer/source rank).
static constexpr int kMaxConnCheckRanks = 1024;

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

__global__ void ginAnvilConnCheckKernel(unsigned long long* localSignals, int nRanks,
                                        unsigned long long stamp, int* missing) {
  int x = threadIdx.x;
  if (x >= nRanks) return;
  __threadfence_system();
  unsigned long long v = __atomic_load_n(localSignals + x, __ATOMIC_RELAXED);
  missing[x] = (v == stamp) ? 0 : 1;
}

static int connCheckLaunchOk() {
  const hipError_t err = hipGetLastError();
  return (err == hipSuccess) ? 0 : -1;
}

}  // namespace conn_check
}  // namespace gin_anvil

extern "C" int ginAnvilConnWrite(void* remoteAddrsDev, int nRanks, int selfRank,
                                 unsigned long long stamp, hipStream_t stream) {
  if (nRanks <= 0 || nRanks > gin_anvil::conn_check::kMaxConnCheckRanks) return -1;
  hipLaunchKernelGGL(gin_anvil::conn_check::ginAnvilConnWriteKernel, dim3(1), dim3(nRanks), 0, stream,
                     reinterpret_cast<uintptr_t*>(remoteAddrsDev), nRanks, selfRank, stamp);
  return gin_anvil::conn_check::connCheckLaunchOk();
}

extern "C" int ginAnvilConnCheck(void* localSignals, int nRanks, unsigned long long stamp, int* missingDev,
                                 hipStream_t stream) {
  if (nRanks <= 0 || nRanks > gin_anvil::conn_check::kMaxConnCheckRanks) return -1;
  hipLaunchKernelGGL(gin_anvil::conn_check::ginAnvilConnCheckKernel, dim3(1), dim3(nRanks), 0, stream,
                     reinterpret_cast<unsigned long long*>(localSignals), nRanks, stamp, missingDev);
  return gin_anvil::conn_check::connCheckLaunchOk();
}

#else  // !__HIPCC__ && !__CUDACC__

#include "gin/gin_anvil_conn_check.h"

extern "C" int ginAnvilConnWrite(void* remoteAddrsDev, int nRanks, int selfRank,
                                 unsigned long long stamp, hipStream_t stream) {
  (void)remoteAddrsDev;
  (void)nRanks;
  (void)selfRank;
  (void)stamp;
  (void)stream;
  return -1;
}

extern "C" int ginAnvilConnCheck(void* localSignals, int nRanks, unsigned long long stamp, int* missingDev,
                                 hipStream_t stream) {
  (void)localSignals;
  (void)nRanks;
  (void)stamp;
  (void)missingDev;
  (void)stream;
  return -1;
}

#endif  // __HIPCC__ || __CUDACC__

#endif  // ENABLE_ROCSHMEM_GIN
