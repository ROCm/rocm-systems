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
/*
Testcase Scenarios :

*/
#include "occupancy_common.hh"
#include "distributed_shared_memory.hh"
#include "resource_guards.hh"

static __global__ void f1(float* a) { *a = 1.0; }
static void host_f1(float* a) { *a = 1.0; }
static __global__ void reverse(int* d, int n) {
  __shared__ int shBuf[64];
  int t = threadIdx.x;
  int tr = n - t - 1;

  shBuf[t] = d[t];
  __syncthreads();
  d[t] = shBuf[tr];
};

TEST_CASE("Unit_hipOccupancyMaxPotentialClusterSize_Positive_RangeValidation") {
  int clusterSize;
  hipDeviceProp_t props;
  hipLaunchConfig_t config = {};
  hipError_t hip_error;

  config.blockDim = {1024};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  hip_error =
      hipOccupancyMaxPotentialClusterSize(&clusterSize, reinterpret_cast<const void*>(f1), &config);

  if (props.clusterLaunch) {
    INFO("Max potential cluster size is: " << clusterSize);
    // at the time of this writing a SPI could be drive up to 15 CUs. The number of CUs per SE
    // varies per silicon, but any AMD silicon should have at least 8
    REQUIRE((clusterSize > 8 && clusterSize <= 16));
  } else {
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }
}

// TODO gehernan enable this test (requires distributed shared memory)
TEST_CASE("Unit_hipOccupancyMaxPotentialClusterSize_Verify_Launch") {
  int numBytes;
  LinearAllocGuard<int> h_output;
  LinearAllocGuard<int> d_output;
  hipLaunchAttribute attribute[1];
  hipError_t hip_error;
  hipLaunchConfig_t config = {};
  int clusterSize;

  SECTION("launch with maximum size no shmem") {
    hipLaunchConfig_t f1config = {};
    int f1MaxClusterSize = -1;
    LinearAllocGuard<float> output(LinearAllocs::hipMalloc, sizeof(float));

    f1config.blockDim = {1024};
    HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&f1MaxClusterSize,
                                                  reinterpret_cast<const void*>(f1), &f1config));
    attribute[0].id = hipLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = f1MaxClusterSize;
    config.numAttrs = 1;
    config.attrs = attribute;
    // make sure we can launch a kernel with the maximum cluster size
    HIP_CHECK(hipLaunchKernelEx(&f1config, f1, output.ptr()));
    HIP_CHECK(hipDeviceSynchronize());
  }

  config.blockDim = {256};
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(
      &clusterSize, reinterpret_cast<const void*>(distributed_shmem_kernel), &config));
  REQUIRE(clusterSize >= 8);
  config.gridDim = {static_cast<unsigned int>(clusterSize)};
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = clusterSize;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.numAttrs = 1;
  config.attrs = attribute;
  config.dynamicSmemBytes = ThreadsPerBlock * ElementsPerThread * sizeof(int);
  numBytes = clusterSize * config.dynamicSmemBytes;

  SECTION("launch with maximum size shmem") {
    int expected = 0;

    d_output = LinearAllocGuard<int>(LinearAllocs::hipMalloc, numBytes);
    h_output = LinearAllocGuard<int>(LinearAllocs::hipHostMalloc, numBytes);

    // this launch should complete ok as it is not above the maximum cluster size
    HIP_CHECK(hipLaunchKernelEx(&config, distributed_shmem_kernel, d_output.ptr()));
    HIP_CHECK(hipMemcpy(h_output.ptr(), d_output.ptr(), numBytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < clusterSize; i++) {
      for (int j = 0; j < ThreadsPerBlock * ElementsPerThread; j++) {
        expected += i;
      }
    }

    REQUIRE(h_output.ptr()[0] == expected);
  }

  SECTION("cluster too big") {
    d_output = LinearAllocGuard<int>(LinearAllocs::hipMalloc, numBytes);
    attribute[0].val.clusterDim.x = clusterSize + 1;
    // this launch is above the maximum cluster size
    hip_error = hipLaunchKernelEx(&config, distributed_shmem_kernel, d_output.ptr());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }
}

TEST_CASE("Unit_hipOccupancyMaxPotentialClusterSize_Negative_Parameters") {
  hipLaunchConfig_t config = {};
  hipError_t hip_error;
  int clusterSize = -1;
  hipDeviceProp_t props;
  hipLaunchAttribute attribute[1];

  config.blockDim = {1024};
  attribute[0].id = hipLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 1;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.numAttrs = 1;
  config.attrs = attribute;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  if (!props.clusterLaunch) {
    SUCCEED("cluster launches are not supported on this device");
    return;
  }

  SECTION("block too big") {
    config.blockDim = std::numeric_limits<unsigned int>::max();
    hip_error = hipOccupancyMaxPotentialClusterSize(&clusterSize, reinterpret_cast<const void*>(f1),
                                                    &config);
    REQUIRE(hip_error == hipErrorInvalidValue);
  }

  SECTION("invalid function pointer") {
    hip_error = hipOccupancyMaxPotentialClusterSize(
        &clusterSize, reinterpret_cast<const void*>(host_f1), &config);
    REQUIRE(hip_error == hipErrorInvalidDeviceFunction);
  }

  SECTION("not enough shared memory") {
    config.dynamicSmemBytes = props.sharedMemPerBlock + 1;
    HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&clusterSize, reinterpret_cast<const void*>(f1),
                                                  &config));
    REQUIRE(clusterSize == 0);
  }

  SECTION("not enough shared memory 2") {
    config.dynamicSmemBytes = props.sharedMemPerBlock;
    HIP_CHECK(hipOccupancyMaxActiveClusters(&clusterSize, reinterpret_cast<const void*>(reverse),
                                            &config));
    // reverse() uses a bit of (non-dynamic) shared memory; enough to go over the limit
    REQUIRE(clusterSize == 0);
  }
}
