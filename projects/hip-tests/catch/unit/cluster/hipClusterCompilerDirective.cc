/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>

#include "ClusterHelper.hpp"

/**
 * @addtogroup cluster
 * @{
 * @ingroup ClusterTest
 * Contains unit tests for cluster launch using compiler directives and launch
 */

#define CLX 2
#define CLY 1
#define CLZ 1

__global__ void CLUSTER_DIMS(CLX, CLY, CLZ) ClusterLaunchKernelBasicCD(int* in, int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernel with compiler directives that takes cluster dimensions to launch across
 * different compute units.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterCompilerDirective.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

constexpr size_t num_elems = 64;
constexpr size_t num_size = num_elems * sizeof(int);

constexpr int nbig = 16;   // number of blocks in grid.
constexpr int ntib = 512;  // number of threads in blocks.

void SetupHostMemory(int*& hptr_in, int*& hptr_out) {
  hptr_in = (int*)malloc(num_size);
  hptr_out = (int*)malloc(num_size);

  memset(hptr_in, 0x00, num_size);
  memset(hptr_out, 0x00, num_size);

  for (size_t idx = 0; idx < num_elems; ++idx) {
    hptr_in[idx] = static_cast<int>(idx);
  }
}

void SetupDeviceMemory(int*& dptr_in, int*& dptr_out) {
  HIP_CHECK(hipMalloc(&dptr_in, num_size));
  HIP_CHECK(hipMalloc(&dptr_out, num_size));

  HIP_CHECK(hipMemset(dptr_in, 0x00, num_size));
  HIP_CHECK(hipMemset(dptr_out, 0x00, num_size));
}

void ReleaseHostAndDeviceMemory(int* hptr_in, int* hptr_out, int* dptr_in, int* dptr_out) {
  HIP_CHECK(hipFree(dptr_in));
  HIP_CHECK(hipFree(dptr_out));
  free(hptr_in);
  free(hptr_out);
}

HIP_TEST_CASE(Unit_hipClusterLaunch_CompilerDirective_Basic) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  BasicMemoryAllocator<int> bma(num_elems);

  int* hptr_in = bma.CreateAndResetHostMemory();
  int* hptr_out = bma.CreateAndResetHostMemory();
  int* dptr_in = bma.CreateAndResetDeviceMemory();
  int* dptr_out = bma.CreateAndResetDeviceMemory();

  assert(hptr_in != nullptr && hptr_out != nullptr && dptr_in != nullptr && dptr_out != nullptr);

  HIP_CHECK(hipMemcpy(dptr_in, hptr_in, num_size, hipMemcpyHostToDevice));
  ClusterLaunchKernelBasicCD<<<nbig, ntib>>>(dptr_in, dptr_out, num_elems);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hptr_out, dptr_out, num_size, hipMemcpyDeviceToHost));

  REQUIRE(bma.ValidateArrays(hptr_in, hptr_out));

  bma.DestroyHostMemory(hptr_in);
  bma.DestroyHostMemory(hptr_out);
  bma.DestroyDeviceMemory(dptr_in);
  bma.DestroyDeviceMemory(dptr_out);
}

/**
 * End doxygen group ClusterTest.
 * @}
 */
