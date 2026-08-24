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
 * Contains unit tests for cluster occupancy APIs
 */

__global__ void ClusterOccupancyKernel(int* in, int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Tests hipOccupancyMaxActiveClusters API to query the maximum number of active clusters
 * that can run concurrently on the device for a given kernel configuration.
 *
 * Test Source
 * ------------------------
 *  - catch/unit/cluster/hipClusterOccupancy.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipOccupancyMaxActiveClusters_Basic) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  hipLaunchConfig_t config;
  config.gridDim = 16;
  config.blockDim = 512;

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {2, 1, 1};
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = nullptr;
  config.dynamicSmemBytes = 0;

  int numClusters = 0;
  HIP_CHECK(hipOccupancyMaxActiveClusters(&numClusters,
            reinterpret_cast<const void*>(&ClusterOccupancyKernel), &config));

  INFO("Maximum active clusters: " << numClusters);
  REQUIRE(numClusters > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Tests hipOccupancyMaxPotentialClusterSize API to query the maximum potential cluster size
 * (in number of blocks) that can run on the device for a given kernel configuration.
 *
 * Test Source
 * ------------------------
 *  - catch/unit/cluster/hipClusterOccupancy.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipOccupancyMaxPotentialClusterSize_Basic) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  hipLaunchConfig_t config;
  config.gridDim = 16;
  config.blockDim = 512;

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {2, 1, 1};
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = nullptr;
  config.dynamicSmemBytes = 0;

  int clusterSize = 0;
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&clusterSize,
            reinterpret_cast<const void*>(&ClusterOccupancyKernel), &config));

  INFO("Maximum potential cluster size: " << clusterSize);
  REQUIRE(clusterSize > 0);
}

/**
 * Test Description
 * ------------------------
 *  - Tests cluster launch with maximum potential cluster size. Queries the maximum cluster size
 * using hipOccupancyMaxPotentialClusterSize API, then launches a kernel with that cluster size
 * and validates the results.
 *
 * Test Source
 * ------------------------
 *  - catch/unit/cluster/hipClusterOccupancy.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipClusterLaunch_MaxClusterSize) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  constexpr size_t num_elems = 1024;
  constexpr size_t num_size = num_elems * sizeof(int);
  constexpr int ntib = 256;

  hipLaunchConfig_t query_config;
  query_config.gridDim = 16;
  query_config.blockDim = ntib;
  query_config.attrs = nullptr;
  query_config.numAttrs = 0;
  query_config.stream = nullptr;
  query_config.dynamicSmemBytes = 0;

  int maxClusterSize = 0;
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&maxClusterSize,
            reinterpret_cast<const void*>(&ClusterOccupancyKernel), &query_config));

  INFO("Maximum potential cluster size: " << maxClusterSize);
  REQUIRE(maxClusterSize > 0);

  BasicMemoryAllocator<int> bma(num_elems);

  int* hptr_in = bma.CreateAndResetHostMemory();
  int* hptr_out = bma.CreateAndResetHostMemory();
  int* dptr_in = bma.CreateAndResetDeviceMemory();
  int* dptr_out = bma.CreateAndResetDeviceMemory();

  assert(hptr_in != nullptr && hptr_out != nullptr && dptr_in != nullptr && dptr_out != nullptr);

  for (size_t i = 0; i < num_elems; i++) {
    hptr_in[i] = i;
  }

  HIP_CHECK(hipMemcpy(dptr_in, hptr_in, num_size, hipMemcpyHostToDevice));

  hipLaunchConfig_t config;
  int nbig = maxClusterSize * 4;
  config.gridDim = nbig;
  config.blockDim = ntib;

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {static_cast<unsigned int>(maxClusterSize), 1, 1};
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = nullptr;
  config.dynamicSmemBytes = 0;

  void* kernel_params[] = {&dptr_in, &dptr_out, reinterpret_cast<void*>(const_cast<size_t*>(&num_elems))};

  INFO("Launching kernel with max cluster size: " << maxClusterSize);
  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<const void*>(&ClusterOccupancyKernel),
                               kernel_params));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hptr_out, dptr_out, num_size, hipMemcpyDeviceToHost));

  REQUIRE(bma.ValidateArrays(hptr_in, hptr_out));

  bma.DestroyHostMemory(hptr_in);
  bma.DestroyHostMemory(hptr_out);
  bma.DestroyDeviceMemory(dptr_in);
  bma.DestroyDeviceMemory(dptr_out);
}

/**
 * Test Description
 * ------------------------
 *  - Tests cluster launch with maximum active clusters and maximum cluster size.
 * Queries both hipOccupancyMaxActiveClusters and hipOccupancyMaxPotentialClusterSize,
 * then launches a kernel using the maximum values for both parameters to stress test
 * the cluster launch capability at full scale.
 *
 * Test Source
 * ------------------------
 *  - catch/unit/cluster/hipClusterOccupancy.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipClusterLaunch_MaxActiveClustersAndSize) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  constexpr int ntib = 256;

  hipLaunchConfig_t query_config;
  query_config.gridDim = 16;
  query_config.blockDim = ntib;
  query_config.stream = nullptr;
  query_config.dynamicSmemBytes = 0;

  int maxClusterSize = 0;
  query_config.attrs = nullptr;
  query_config.numAttrs = 0;
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&maxClusterSize,
            reinterpret_cast<const void*>(&ClusterOccupancyKernel), &query_config));

  INFO("Maximum potential cluster size: " << maxClusterSize);
  REQUIRE(maxClusterSize > 0);

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim = {static_cast<unsigned int>(maxClusterSize), 1, 1};
  query_config.attrs = attribute;
  query_config.numAttrs = 1;

  int maxActiveClusters = 0;
  HIP_CHECK(hipOccupancyMaxActiveClusters(&maxActiveClusters,
            reinterpret_cast<const void*>(&ClusterOccupancyKernel), &query_config));

  INFO("Maximum active clusters: " << maxActiveClusters);
  REQUIRE(maxActiveClusters > 0);

  int totalBlocks = maxActiveClusters * maxClusterSize;
  size_t num_elems = totalBlocks * ntib;
  size_t num_size = num_elems * sizeof(int);

  INFO("Launching with " << maxActiveClusters << " clusters, "
       << maxClusterSize << " blocks per cluster, "
       << totalBlocks << " total blocks, "
       << num_elems << " total threads");

  BasicMemoryAllocator<int> bma(num_elems);

  int* hptr_in = bma.CreateAndResetHostMemory();
  int* hptr_out = bma.CreateAndResetHostMemory();
  int* dptr_in = bma.CreateAndResetDeviceMemory();
  int* dptr_out = bma.CreateAndResetDeviceMemory();

  assert(hptr_in != nullptr && hptr_out != nullptr && dptr_in != nullptr && dptr_out != nullptr);

  for (size_t i = 0; i < num_elems; i++) {
    hptr_in[i] = i;
  }

  HIP_CHECK(hipMemcpy(dptr_in, hptr_in, num_size, hipMemcpyHostToDevice));

  hipLaunchConfig_t config;
  config.gridDim = totalBlocks;
  config.blockDim = ntib;
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = nullptr;
  config.dynamicSmemBytes = 0;

  void* kernel_params[] = {&dptr_in, &dptr_out, reinterpret_cast<void*>(const_cast<size_t*>(&num_elems))};

  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<const void*>(&ClusterOccupancyKernel),
                               kernel_params));
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
