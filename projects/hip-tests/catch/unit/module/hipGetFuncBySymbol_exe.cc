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

#include "hip/hip_runtime.h"

#define ARR_SIZE (32 * 32)
#define SIZE (ARR_SIZE * sizeof(int))

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FUNCTION__, __LINE__);                                                      \
      exit(0);                                                                                     \
    }                                                                                              \
  }

/**
 * Sample Kernel to be used for functional test cases
 */
__global__ void hipKernel(int* a) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = offset; i < ARR_SIZE; i += stride) {
    a[i] += a[i];
  }
}

/**
 * This function gets the function pointer and launches the kernel and
 * verifies the reult
 */
int main() {
  hipFunction_t funcPointer;

  if (hipGetFuncBySymbol(&funcPointer, reinterpret_cast<const void*>(hipKernel)) != hipSuccess) {
    return -1;
  }

  int* h_a = reinterpret_cast<int*>(malloc(SIZE));
  if (h_a == nullptr) {
    return -1;
  }

  int* output_ref = reinterpret_cast<int*>(malloc(SIZE));
  if (output_ref == nullptr) {
    return -1;
  }

  for (int i = 0; i < ARR_SIZE; i++) {
    h_a[i] = 2;
    output_ref[i] = 4;
  }

  int* d_a = nullptr;
  HIP_CHECK(hipMalloc(&d_a, SIZE));
  if (d_a == nullptr) {
    return -1;
  }
  HIP_CHECK(hipMemcpy(d_a, h_a, SIZE, hipMemcpyHostToDevice));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, 64);

  void* kernelParam[] = {d_a};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  if (hipModuleLaunchKernel(funcPointer, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z,
                            threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, 0, 0, nullptr,
                            kernel_parameter) != hipSuccess) {
    return -1;
  }

  HIP_CHECK(hipMemcpy(h_a, d_a, SIZE, hipMemcpyDeviceToHost));

  for (int i = 0; i < ARR_SIZE; i++) {
    if (h_a[i] != output_ref[i]) {
      return -1;
    }
  }

  free(h_a);
  free(output_ref);
  HIP_CHECK(hipFree(d_a));

  return 0;
}
