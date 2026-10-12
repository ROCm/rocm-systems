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

#include <vector>

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

// Fit the compiler's per-axis encoding and 16-workgroup cluster limit.
#define OVERSIZED_CLX 4
#define OVERSIZED_CLY 4
#define OVERSIZED_CLZ 1
constexpr int oversizedClusterSize = OVERSIZED_CLX * OVERSIZED_CLY * OVERSIZED_CLZ;

__global__ void CLUSTER_DIMS(CLX, CLY, CLZ) ClusterLaunchKernelBasicCD(int* in, int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

__global__ void CLUSTER_DIMS(OVERSIZED_CLX, OVERSIZED_CLY, OVERSIZED_CLZ)
    ClusterLaunchKernelOversizedCD(int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

// The compiler accepts a zero in some but not all axes and emits it into the code object.
__global__ void CLUSTER_DIMS(0, 1, 1) ClusterLaunchKernelZeroDimCD(int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

// All-zero dims mean "no cluster" and emit no cluster metadata.
__global__ void CLUSTER_DIMS(0, 0, 0) ClusterLaunchKernelNoClusterCD(int* out, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
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
 * Test Description
 * ------------------------
 *  - Launches a kernel whose __cluster_dims__ exceeds the device maximum and checks that the
 *    runtime rejects it with hipErrorInvalidClusterSize.
 *  - Without the runtime check this test hangs rather than fails.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterCompilerDirective.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

HIP_TEST_CASE(Unit_hipClusterLaunch_CompilerDirective_Negative_OversizedCluster) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  hipLaunchAttribute attribute[1];
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 1;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  hipLaunchConfig_t config = {};
  config.numAttrs = 1;
  config.attrs = attribute;
  config.blockDim = {ntib, 1, 1};
  config.gridDim = {OVERSIZED_CLX, OVERSIZED_CLY, OVERSIZED_CLZ};

  int maxClusterSize = 0;
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(
      &maxClusterSize, reinterpret_cast<const void*>(ClusterLaunchKernelOversizedCD), &config));
  INFO("Maximum cluster size reported by the device: " << maxClusterSize);
  if (maxClusterSize >= oversizedClusterSize) {
    HIP_SKIP_TEST("Device places clusters this large, so there is nothing to reject");
    return;
  }

  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, num_size));

  // Grid is a whole number of clusters, so only the size can cause a rejection.
  ClusterLaunchKernelOversizedCD<<<config.gridDim, ntib>>>(dptr, num_elems);
  const hipError_t launch_status = hipGetLastError();

  // Free before REQUIRE; drain first if the launch was wrongly accepted.
  if (launch_status == hipSuccess) {
    HIP_CHECK(hipDeviceSynchronize());
  }
  HIP_CHECK(hipFree(dptr));

  INFO("Launch returned: " << hipGetErrorString(launch_status));
  REQUIRE(launch_status == hipErrorInvalidClusterSize);
}

/**
 * Test Description
 * ------------------------
 *  - Launches a kernel whose __cluster_dims__ has a zero in one axis and checks that the runtime
 *    rejects it with hipErrorInvalidConfiguration, as it does for a zero cluster dimension passed
 *    through hipLaunchAttributeClusterDimension.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterCompilerDirective.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

HIP_TEST_CASE(Unit_hipClusterLaunch_CompilerDirective_Negative_ZeroClusterDim) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, num_size));

  ClusterLaunchKernelZeroDimCD<<<nbig, ntib>>>(dptr, num_elems);
  const hipError_t launch_status = hipGetLastError();

  // Free before REQUIRE; drain first if the launch was wrongly accepted.
  if (launch_status == hipSuccess) {
    HIP_CHECK(hipDeviceSynchronize());
  }
  HIP_CHECK(hipFree(dptr));

  INFO("Launch returned: " << hipGetErrorString(launch_status));
  REQUIRE(launch_status == hipErrorInvalidConfiguration);
}

/**
 * Test Description
 * ------------------------
 *  - Adds a kernel whose __cluster_dims__ has a zero in one axis to a graph and checks that the
 *    rejection reaches the caller through hipGraphInstantiate or hipGraphLaunch.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterCompilerDirective.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

HIP_TEST_CASE(Unit_hipClusterLaunch_CompilerDirective_Negative_ZeroClusterDim_Graph) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, num_size));
  int n = num_elems;
  void* kernel_args[] = {&dptr, &n};

  hipKernelNodeParams node_params = {};
  node_params.func = reinterpret_cast<void*>(ClusterLaunchKernelZeroDimCD);
  node_params.gridDim = dim3(nbig);
  node_params.blockDim = dim3(ntib);
  node_params.kernelParams = kernel_args;

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &node_params));

  // The kernel's command is built at instantiation or at first launch, depending on the
  // graph execution path, so accept the rejection at either point.
  hipGraphExec_t graph_exec = nullptr;
  const hipError_t instantiate_status =
      hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  hipError_t launch_status = hipSuccess;
  if (instantiate_status == hipSuccess) {
    launch_status = hipGraphLaunch(graph_exec, nullptr);
    // Drain before freeing if the launch was wrongly accepted.
    if (launch_status == hipSuccess) {
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(dptr));
  // Clear the last error so the rejection doesn't leak into later tests.
  (void)hipGetLastError();

  INFO("hipGraphInstantiate returned: " << hipGetErrorString(instantiate_status));
  INFO("hipGraphLaunch returned: " << hipGetErrorString(launch_status));
  const hipError_t status =
      instantiate_status != hipSuccess ? instantiate_status : launch_status;
  REQUIRE(status == hipErrorInvalidConfiguration);
}

/**
 * Test Description
 * ------------------------
 *  - Launches a kernel with __cluster_dims__(0, 0, 0), which means no cluster, and checks that it
 *    runs as an ordinary launch with every block computing its own index.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterCompilerDirective.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

HIP_TEST_CASE(Unit_hipClusterLaunch_CompilerDirective_NoCluster) {
  if (!CheckTargetSupport()) {
    INFO("Target Not Supported!");
    return;
  }

  // One element per thread, several blocks, so a wrong block index shows in the output.
  constexpr int threads_per_block = num_elems / nbig;

  std::vector<int> hout(num_elems, -1);
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, num_size));
  HIP_CHECK(hipMemset(dptr, 0xff, num_size));

  ClusterLaunchKernelNoClusterCD<<<nbig, threads_per_block>>>(dptr, num_elems);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hout.data(), dptr, num_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dptr));

  for (size_t idx = 0; idx < num_elems; ++idx) {
    INFO("Index " << idx);
    REQUIRE(hout[idx] == static_cast<int>(idx));
  }
}

/**
 * End doxygen group ClusterTest.
 * @}
 */
