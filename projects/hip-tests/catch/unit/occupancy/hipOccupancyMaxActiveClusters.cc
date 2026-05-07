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

static __global__ void f1(float* a) { *a = 1.0; }
static __global__ __cluster_dims__(1, 1, 1) void f1_with_attr(float* a) { *a = 1.0; }
static void host_f1(float* a) { *a = 1.0; }
static __global__ void reverse(int* d, int n) {
  __shared__ int shBuf[64];
  int t = threadIdx.x;
  int tr = n - t - 1;

  shBuf[t] = d[t];
  __syncthreads();
  d[t] = shBuf[tr];
};

TEST_CASE("Unit_hipOccupancyMaxActiveClusters_Positive_RangeValidation") {
  hipLaunchConfig_t config = {};
  hipLaunchAttribute attribute[1];
  int clustersNoLDS = -1, clustersLDS = -1;
  int maxClusterSize = -1;
  // maximum number of blocks per CU using no shared memory
  int maxBlocks;
  hipDeviceProp_t props;
  int numSharedEngines;

  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  if (!props.clusterLaunch) {
    SUCCEED("cluster launches are not supported on this device");
    return;
  }

  auto& clusterDim = attribute[0].val.clusterDim;

  attribute[0].id = hipLaunchAttributeClusterDimension;
  clusterDim.x = 4;
  clusterDim.y = 2;
  clusterDim.z = 2;
  config.numAttrs = 1;
  config.attrs = attribute;
  config.blockDim = {128};
  config.gridDim = {4, 2, 2};

  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&maxClusterSize, reinterpret_cast<const void*>(f1),
                                                &config));
  INFO("sharedMemPerBlock: " << props.sharedMemPerBlock);
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, reinterpret_cast<const void*>(f1), 128, 0));
  REQUIRE(maxBlocks > 0);
  // if using a partial simulation; this number would need to be changed
  // (e.g. to 2 SEs if using 2 XCCs)
  numSharedEngines = props.multiProcessorCount / maxClusterSize;

  // these check that shared memory usage has an effect on occupancy calculations
  HIP_CHECK(
      hipOccupancyMaxActiveClusters(&clustersNoLDS, reinterpret_cast<const void*>(f1), &config));
  INFO("occupancy with no shared memory: " << clustersNoLDS);
  // limited by the number of in-flight clusters per SE
  REQUIRE(clustersNoLDS == 30);

  config.dynamicSmemBytes = props.sharedMemPerBlock / 8;
  HIP_CHECK(
      hipOccupancyMaxActiveClusters(&clustersLDS, reinterpret_cast<const void*>(f1), &config));
  INFO("occupancy limited by shared memory: " << clustersLDS);

  float ratio = maxBlocks / static_cast<float>(clusterDim.x * clusterDim.y * clusterDim.z);
  // limited due to shared memory
  REQUIRE(clustersLDS == (maxClusterSize * ratio * numSharedEngines) / 2);

  config.dynamicSmemBytes = props.sharedMemPerBlock / 4;
  HIP_CHECK(
      hipOccupancyMaxActiveClusters(&clustersLDS, reinterpret_cast<const void*>(f1), &config));
  INFO("occupancy non-max cluster size: " << clustersLDS);
  // further limited by shared memory; the number of clusters per SE is lower than the maximum
  // cluster size
  REQUIRE(clustersLDS == (maxBlocks * ratio * numSharedEngines) / 4);
}

TEST_CASE("Unit_hipOccupancyMaxActiveClusters_Negative_Zero_Cluster") {
  hipLaunchConfig_t config = {};
  hipLaunchAttribute attribute[1];
  hipError_t hip_error;
  int numClusters = -1;
  hipDeviceProp_t props;

  attribute[0].id = hipLaunchAttributeClusterDimension;
  config.numAttrs = 1;
  config.attrs = attribute;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  // on Nvidia this will not produce an error, even if the device does not support cluster
  // launches numClusters could be zero; on AMD it will be an error as HSA does not support such
  // size
  attribute[0].val.clusterDim.x = 0;
  attribute[0].val.clusterDim.y = 0;
  attribute[0].val.clusterDim.z = 0;
  hip_error =
      hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1), &config);
  REQUIRE(hip_error == hipErrorInvalidClusterSize);
}

TEST_CASE("Unit_hipOccupancyMaxActiveClusters_Negative_Parameters") {
  hipLaunchConfig_t config = {};
  hipLaunchAttribute attribute[1];
  hipError_t hip_error;
  int numClusters = -1;
  hipDeviceProp_t props;

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
    hip_error =
        hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1), &config);
    REQUIRE(hip_error == hipErrorInvalidValue);
  }

  SECTION("cluster too big") {
    attribute[0].val.clusterDim.x = std::numeric_limits<unsigned int>::max();
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    hip_error =
        hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1), &config);
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }

  SECTION("absent cluster dims") {
    hipLaunchConfig_t defaultConfig = {};

    hip_error = hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1),
                                              &defaultConfig);
    // the cluster dims must be specified either as a symbol attribute or in the launch config
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }

  SECTION("attribute cluster dims != launch config dims") {
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 2;
    attribute[0].val.clusterDim.z = 2;
    hip_error = hipOccupancyMaxActiveClusters(&numClusters,
                                              reinterpret_cast<const void*>(f1_with_attr), &config);
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }

  SECTION("divisibility") {
    config.gridDim.x = 3;
    attribute[0].val.clusterDim.x = 2;
    hip_error =
        hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1), &config);
    REQUIRE(hip_error == hipErrorInvalidClusterSize);
  }

  SECTION("invalid function pointer") {
    hip_error = hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(host_f1),
                                              &config);
    REQUIRE(hip_error == hipErrorInvalidDeviceFunction);
  }

  SECTION("not enough shared memory 1") {
    config.dynamicSmemBytes = props.sharedMemPerBlock + 1;
    // if dynamicSmemBytes is higher than the device could handle; it is not an error from the
    // point of view of this call; it simply means that the number of clusters that could be active
    // would be zero
    HIP_CHECK(
        hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(f1), &config));
    REQUIRE(numClusters == 0);
  }

  SECTION("not enough shared memory 2") {
    config.dynamicSmemBytes = props.sharedMemPerBlock;
    HIP_CHECK(hipOccupancyMaxActiveClusters(&numClusters, reinterpret_cast<const void*>(reverse),
                                            &config));
    // reverse() uses a bit of (non-dynamic) shared memory; enough to go over the limit
    REQUIRE(numClusters == 0);
  }
}
