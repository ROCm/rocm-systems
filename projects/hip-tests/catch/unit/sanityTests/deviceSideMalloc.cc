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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
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

// Kernel: device-side new/delete
__global__ void deviceSideNew() {
  int* ptr = new int;
  if (ptr == nullptr)
    return;
  delete ptr;
}

// Kernel: device-side malloc/free (single thread)
__global__ void deviceSideMalloc() {
  int* ptr = static_cast<int*>(malloc(sizeof(int)));
  if (ptr == nullptr)
    return;
  free(ptr);
}

// Kernel: per-thread malloc, memset, free
__global__ void perThreadMalloc() {
  size_t size = 123;
  char* ptr = static_cast<char*>(malloc(size));
  if (ptr == nullptr)
    return;
  memset(ptr, 0, size);
  free(ptr);
}

// Kernel: per-block allocation
__global__ void perThreadBlockAllocation() {
  __shared__ int* data;

  if (threadIdx.x == 0) {
    size_t size = blockDim.x * 64 * sizeof(int);
    data = static_cast<int*>(malloc(size));
  }
  __syncthreads();

  if (data == nullptr)
    return;

  int* ptr = data;
  for (int i = 0; i < 64; ++i)
    ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;

  __syncthreads();

  if (threadIdx.x == 0)
    free(data);
}

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
 *    - Launches a kernel that uses device-side new/delete for a single int.
 * Test source
 * ------------------------
 *    - unit/sanityTests/deviceSideMalloc.cc
 */
TEST_CASE("Unit_deviceSideNewDelete") {
  hipLaunchKernelGGL(deviceSideNew, 1, 1, 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel that uses device-side malloc/free for a single int.
 * Test source
 * ------------------------
 *    - unit/sanityTests/deviceSideMalloc.cc
 */
TEST_CASE("Unit_deviceSideMalloc") {
  hipLaunchKernelGGL(deviceSideMalloc, 1, 1, 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel with multiple threads; each thread allocates,
 *      memset, and frees device memory (per-thread malloc).
 * Test source
 * ------------------------
 *    - unit/sanityTests/deviceSideMalloc.cc
 */
TEST_CASE("Unit_PerThreadMalloc") {
  perThreadMalloc<<<1, 5>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * Test Description
 * ------------------------
 *    - Launches blocks; first thread per block allocates shared buffer,
 *      all threads write coalesced, then first thread frees (per-block allocation).
 * Test source
 * ------------------------
 *    - unit/sanityTests/deviceSideMalloc.cc
 */
TEST_CASE("Unit_PerThreadBlockAllocation") {
  perThreadBlockAllocation<<<10, 128>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
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
TEST_CASE("Unit_AccessMallocAcrossKernels") {
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
