/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip_tests_config.hh>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>

/**
 * @addtogroup DeviceSideMallocTest DeviceSideMallocTest
 * @{
 * @ingroup DeviceSideMallocTest
 */

#define NUM_BLOCKS_DSM 20

__device__ int* dataptr_dsm[NUM_BLOCKS_DSM];

__global__ void allocmemKernel() {
  if (threadIdx.x == 0)
    dataptr_dsm[blockIdx.x] = static_cast<int*>(malloc(blockDim.x * sizeof(int)));
  __syncthreads();

  if (dataptr_dsm[blockIdx.x] == nullptr)
    return;

  dataptr_dsm[blockIdx.x][threadIdx.x] = 0;
}

__global__ void usememKernel() {
  int* ptr = dataptr_dsm[blockIdx.x];
  if (ptr != nullptr)
    ptr[threadIdx.x] += threadIdx.x;
}

__global__ void freememKernel() {
  int* ptr = dataptr_dsm[blockIdx.x];
  if (ptr != nullptr && threadIdx.x == 0) {
    free(ptr);
    dataptr_dsm[blockIdx.x] = nullptr;
  }
}

/**
 * Test Description
 * ------------------------
 *    - Allocates device memory in one kernel, uses it in multiple kernels,
 *      then frees in a separate kernel (allocation across kernels).
 * Test source
 * ------------------------
 *    - unit/sanityTests/deviceSideMalloc.cc
 */
HIP_TEST_CASE(Unit_AccessMallocAcrossKernels) {
  allocmemKernel<<<NUM_BLOCKS_DSM, 10>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  usememKernel<<<NUM_BLOCKS_DSM, 10>>>();
  usememKernel<<<NUM_BLOCKS_DSM, 10>>>();
  usememKernel<<<NUM_BLOCKS_DSM, 10>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  freememKernel<<<NUM_BLOCKS_DSM, 10>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * End doxygen group DeviceSideMallocTest.
 * @}
 */
