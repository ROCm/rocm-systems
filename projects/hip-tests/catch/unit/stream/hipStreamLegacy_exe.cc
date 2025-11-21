/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include "hip/hip_runtime.h"

static constexpr int N = 2 * 1024 * 1024;
static constexpr size_t NBYTES = N * sizeof(int);

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FUNCTION__, __LINE__);                                                      \
      return -1;                                                                                   \
    }                                                                                              \
  }

/**
 * Used hipStreamLegacy in child process and copies H2H, H2D, D2D, D2H
 */
int main() {
  constexpr int elementVal = 3456;

  int* hostArr1 = new int[N];
  if (hostArr1 == nullptr) {
    std::cout << "Invalid hostArr1" << std::endl;
    return -1;
  }

  for (int i = 0; i < N; i++) {
    hostArr1[i] = elementVal;
  }

  int* hostArr2 = new int[N];
  if (hostArr2 == nullptr) {
    std::cout << "Invalid hostArr2" << std::endl;
    return -1;
  }

  int* devArr1 = nullptr;
  HIP_CHECK(hipMalloc(&devArr1, NBYTES));
  if (devArr1 == nullptr) {
    std::cout << "Invalid devArr1" << std::endl;
    return -1;
  }

  int* devArr2 = nullptr;
  HIP_CHECK(hipMalloc(&devArr2, NBYTES));
  if (devArr2 == nullptr) {
    std::cout << "Invalid devArr2" << std::endl;
    return -1;
  }

  int* hostArr3 = new int[N];
  if (hostArr3 == nullptr) {
    std::cout << "Invalid hostArr3" << std::endl;
    return -1;
  }

  HIP_CHECK(hipMemcpyAsync(hostArr2, hostArr1, NBYTES, hipMemcpyHostToHost, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr1, hostArr2, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr2, devArr1, NBYTES, hipMemcpyDeviceToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArr3, devArr2, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    if (hostArr3[i] != elementVal) {
      std::cout << "At index : = " << i << " Got value : " << hostArr3[i]
                << " Expected value : " << elementVal << std::endl;
      return -1;
    }
  }

  delete[] hostArr1;
  delete[] hostArr2;
  HIP_CHECK(hipFree(devArr1));
  HIP_CHECK(hipFree(devArr2));
  delete[] hostArr3;

  return 0;
}
