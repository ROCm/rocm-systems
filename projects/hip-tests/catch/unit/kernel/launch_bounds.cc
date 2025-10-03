/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_common.hh>


constexpr size_t N = 1024;
int p_blockSize = 256;

__global__ void __launch_bounds__(256, 2) myKern(int* C, const int* A, int N) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  if (tid < N) {
    C[tid] = A[tid];
  }
}
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check launch bounds via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/launch_bounds.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_kernel_LaunchBounds_Functional") {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *C_d, *A_h, *C_h;
  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  C_h = reinterpret_cast<int*>(malloc(Nbytes));

  for (int i = 0; i < N; i++) {
    A_h[i] = i * 10;
    C_h[i] = 0x0;
  }
  int blocks = N / p_blockSize;

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipGetLastError());
  hipLaunchKernelGGL(myKern, dim3(blocks), dim3(p_blockSize), 0, 0, C_d, A_d, N);

#ifdef __HIP_PLATFORM_NVIDIA__
  cudaFuncAttributes attrib;
  cudaFuncGetAttributes(&attrib, myKern);
  printf("binaryVersion = %d\n", attrib.binaryVersion);
  printf("cacheModeCA = %d\n", attrib.cacheModeCA);
  printf("constSizeBytes = %zu\n", attrib.constSizeBytes);
  printf("localSizeBytes = %zud\n", attrib.localSizeBytes);
  printf("maxThreadsPerBlock = %d\n", attrib.maxThreadsPerBlock);
  printf("numRegs = %d\n", attrib.numRegs);
  printf("ptxVersion = %d\n", attrib.ptxVersion);
  printf("sharedSizeBytes = %zud\n", attrib.sharedSizeBytes);
#endif

  HIPCHECK(hipDeviceSynchronize());
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipDeviceSynchronize());

  for (int i = 0; i < N; i++) {
    int goldVal = i * 10;
    REQUIRE(C_h[i] == goldVal);
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
}

/**
 * End doxygen group KernelTest.
 * @}
 */
