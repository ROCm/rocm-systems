/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#if defined(__HIPCC__) || defined(__CUDACC__)

#include "gin/gin_abce_conn_check.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gin_abce {
namespace conn_check {

__global__ void ginAbceConnWriteKernel(uintptr_t* remoteAddrs, int nRanks, int selfRank,
                                        unsigned long long stamp) {
  int p = threadIdx.x;
  if (p >= nRanks) return;
  uintptr_t base = remoteAddrs[p];
  if (base == 0) return;
  unsigned long long* dst = reinterpret_cast<unsigned long long*>(base) + selfRank;
  __atomic_store_n(dst, stamp, __ATOMIC_RELAXED);
  __threadfence_system();
}

__global__ void ginAbceConnCheckKernel(unsigned long long* localSignals, int nRanks,
                                        unsigned long long stamp, int* missing) {
  int x = threadIdx.x;
  if (x >= nRanks) return;
  __threadfence_system();
  unsigned long long v = __atomic_load_n(localSignals + x, __ATOMIC_RELAXED);
  missing[x] = (v == stamp) ? 0 : 1;
}

static int connCheckLaunchOk() {
  const hipError_t err = hipPeekAtLastError();
  return (err == hipSuccess) ? 0 : -1;
}

}  // namespace conn_check
}  // namespace gin_abce

extern "C" int ginAbceConnWrite(void* remoteAddrsDev, int nRanks, int selfRank,
                                 unsigned long long stamp, hipStream_t stream) {
  if (remoteAddrsDev == nullptr) return -1;
  if (nRanks <= 0 || nRanks > gin_abce::conn_check::kMaxConnCheckKernelRanks) return -1;
  if (selfRank < 0 || selfRank >= nRanks) return -1;
  (void)hipGetLastError();
  hipLaunchKernelGGL(gin_abce::conn_check::ginAbceConnWriteKernel, dim3(1), dim3(nRanks), 0, stream,
                     reinterpret_cast<uintptr_t*>(remoteAddrsDev), nRanks, selfRank, stamp);
  return gin_abce::conn_check::connCheckLaunchOk();
}

extern "C" int ginAbceConnCheck(void* localSignals, int nRanks, unsigned long long stamp, int* missingDev,
                                 hipStream_t stream) {
  if (localSignals == nullptr || missingDev == nullptr) return -1;
  if (nRanks <= 0 || nRanks > gin_abce::conn_check::kMaxConnCheckKernelRanks) return -1;
  (void)hipGetLastError();
  hipLaunchKernelGGL(gin_abce::conn_check::ginAbceConnCheckKernel, dim3(1), dim3(nRanks), 0, stream,
                     reinterpret_cast<unsigned long long*>(localSignals), nRanks, stamp, missingDev);
  return gin_abce::conn_check::connCheckLaunchOk();
}

#else  // !__HIPCC__ && !__CUDACC__

#error "gin_abce_conn_check_device.cc requires HIP device compilation"

#endif  // __HIPCC__ || __CUDACC__

#endif  // ENABLE_ROCSHMEM_GIN
