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

__global__ void ClusterLaunchKernelBasicL (int* in, int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernel through hipLaunchKernelExC API which has cluster dimentions parametrer
 * conifgured which will launch across different Compute Units.
 *
 * Test Source
 * ------------------------
 *  - catch/unit/cluster/hipClusterLaunch.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

constexpr size_t num_elems = 64;
constexpr size_t num_size = num_elems * sizeof(int);

constexpr int nbig = 16; // number of blocks in grid.
constexpr int ntib = 512; // number of threads in blocks.

void SetupAndLaunchCluster(void** kernel_params) {
  hipLaunchConfig_t config;
  config.gridDim = nbig;
  config.blockDim = ntib;

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {2, 1, 1};
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = nullptr;
  config.dynamicSmemBytes = 64;

  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<const void *>(&ClusterLaunchKernelBasicL),
                               kernel_params));
}

HIP_TEST_CASE(Unit_hipClusterLaunch_LaunchApi_Basic) {

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

  size_t n = num_size;
  HIP_CHECK(hipMemcpy(dptr_in, hptr_in, num_size, hipMemcpyHostToDevice));
  void* kernel_params[] = {&dptr_in, &dptr_out, &n};
  SetupAndLaunchCluster(kernel_params);
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