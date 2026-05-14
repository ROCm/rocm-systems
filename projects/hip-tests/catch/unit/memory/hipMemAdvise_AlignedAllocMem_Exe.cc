/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#if __linux__
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime_api.h"
#include <hip/hip_runtime.h>

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      return -1;                                                                                   \
    }                                                                                              \
  }

// Kernel
__global__ void MemAdvise_Exe(int* Hmm, int n) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (int i = offset; i < n; i += stride) {
    Hmm[i] += 10;
  }
}

static int hipMemAdvise_AlignedAllocMem_Exe() {
  int managedMem = 0, pageMemAccess = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pageMemAccess, hipDeviceAttributePageableMemoryAccess, 0));
  std::cout << "\n hipDeviceAttributePageableMemoryAccess:" << pageMemAccess;
  HIP_CHECK(hipDeviceGetAttribute(&managedMem, hipDeviceAttributeManagedMemory, 0));
  std::cout << "\n hipDeviceAttributeManagedMemory: " << managedMem;

  if ((managedMem == 1) && (pageMemAccess == 1)) {
    int *Mllc = nullptr, MemSz = 4096 * 4, NumElms = 4096, InitVal = 123;
    Mllc = reinterpret_cast<int*>(aligned_alloc(4096, MemSz));

    for (int i = 0; i < NumElms; ++i) {
      Mllc[i] = InitVal;
    }

    hipStream_t strm;
    int DataMismatch = 0;
    HIP_CHECK(hipStreamCreate(&strm));
    // The following hipMemAdvise() call is made to know if advise on
    // aligned_alloc() is causing any issue
    HIP_CHECK(hipMemAdvise(Mllc, MemSz, hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(hipMemPrefetchAsync(Mllc, MemSz, 0, strm));
    HIP_CHECK(hipStreamSynchronize(strm));
    MemAdvise_Exe<<<(NumElms / 32), 32, 0, strm>>>(Mllc, NumElms);
    HIP_CHECK(hipStreamSynchronize(strm));
    for (int i = 0; i < NumElms; ++i) {
      if (Mllc[i] != (InitVal + 10)) {
        DataMismatch++;
      }
    }
    if (DataMismatch != 0) return -1;
  }
  return 0;
}

int main() { return hipMemAdvise_AlignedAllocMem_Exe(); }
#endif
