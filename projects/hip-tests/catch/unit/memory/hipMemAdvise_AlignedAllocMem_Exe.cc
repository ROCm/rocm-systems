/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#if __linux__
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"

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
