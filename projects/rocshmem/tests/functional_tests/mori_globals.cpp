/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/
// This file provides strong definitions of Mori device symbols
// All other test files have MORI_SHMEM_NO_STATIC_INIT defined.
// Only compiled when TEST_WITH_MORI is defined.

#ifdef TEST_WITH_MORI

#include <hip/hip_runtime.h>

// Include internal headers to get type definitions
#include <mori/shmem/internal.hpp>
#include <mori/shmem/shmem_device_api.hpp>
#include <mori/shmem/shmem_api.hpp>

namespace mori {
namespace shmem {

// Strong definition of globalGpuStates (NOT weak)
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;

namespace _static_init {

// Define and register the address provider
void* _getGpuStatesAddr() {
  void* addr = nullptr;
  (void)hipGetSymbolAddress(&addr, HIP_SYMBOL(mori::shmem::globalGpuStates));
  return addr;
}

struct _GpuStatesRegistrar {
  _GpuStatesRegistrar() { RegisterGpuStatesAddrProvider(_getGpuStatesAddr); }
};
_GpuStatesRegistrar _s_gpuStatesRegistrar;

// Define barrier kernel for static init path
__global__ void _barrier_kernel() { ShmemBarrierAllBlock(); }

void _barrierLauncher(hipStream_t stream) {
  _barrier_kernel<<<1, 1, 0, stream>>>();
}

struct _BarrierRegistrar {
  _BarrierRegistrar() { RegisterBarrierLauncher(_barrierLauncher); }
};
_BarrierRegistrar _s_barrierRegistrar;

}  // namespace _static_init
}  // namespace shmem
}  // namespace mori

#endif  // TEST_WITH_MORI
