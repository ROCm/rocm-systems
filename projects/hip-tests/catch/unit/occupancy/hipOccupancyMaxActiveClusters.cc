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

#include <algorithm>

static __global__ void f1(float* a) { *a = 1.0; }
static __global__ CLUSTER_DIMS(1, 1, 1) void f1_with_attr(float* a) { *a = 1.0; }
static void host_f1(float* a) { *a = 1.0; }
static __global__ void reverse(int* d, int n) {
  __shared__ int shBuf[64];
  int t = threadIdx.x;
  int tr = n - t - 1;

  shBuf[t] = d[t];
  __syncthreads();
  d[t] = shBuf[tr];
};

HIP_TEST_CASE(Unit_hipOccupancyMaxActiveClusters_Positive_RangeValidation) {
  hipLaunchConfig_t config = {};
  hipLaunchAttribute attribute[1];
  int clustersNoLDS = -1, clustersLDS = -1;
  int maxClusterSize = -1;
  hipDeviceProp_t props;
  // a maximum of 15 clusters can be in flight per shader engine
  constexpr int maxInFlightClustersPerSE = 15;

  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  if (!props.clusterLaunch) {
    HIP_SKIP_TEST("cluster launches are not supported on this device");
  }

  auto& clusterDim = attribute[0].val.clusterDim;

  attribute[0].id = hipLaunchAttributeClusterDimension;
  config.numAttrs = 1;
  config.attrs = attribute;
  config.blockDim = {128};

  // does not validate the cluster dimensions, so it can be queried before picking them
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&maxClusterSize, reinterpret_cast<const void*>(f1),
                                                &config));
  INFO("maxClusterSize: " << maxClusterSize);
  REQUIRE(maxClusterSize > 0);

  // the largest of these cluster shapes that the device supports
  static constexpr dim3 clusterShapes[] = {{4, 2, 2}, {2, 2, 2}, {2, 2, 1}, {2, 1, 1}, {1, 1, 1}};
  for (const dim3& shape : clusterShapes) {
    if (static_cast<int>(shape.x * shape.y * shape.z) <= maxClusterSize) {
      clusterDim.x = shape.x;
      clusterDim.y = shape.y;
      clusterDim.z = shape.z;
      break;
    }
  }
  config.gridDim = {clusterDim.x, clusterDim.y, clusterDim.z};
  const int clusterSize = clusterDim.x * clusterDim.y * clusterDim.z;

  // the maximum cluster size is the number of CUs in a shader engine
  const int cusPerSE = maxClusterSize;
  const int numSE = props.multiProcessorCount / cusPerSE;
  INFO("clusterSize: " << clusterSize << ", cusPerSE: " << cusPerSE << ", numSE: " << numSE);
  INFO("sharedMemPerBlock: " << props.sharedMemPerBlock);
  REQUIRE(numSE > 0);

  // clusters that fit per SE given the blocks per CU, limited by the in-flight cluster count
  auto expectedClusters = [&](size_t dynamicSmemBytes) {
    int blocksPerCU = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerCU, reinterpret_cast<const void*>(f1), 128, dynamicSmemBytes));
    const int clustersPerSE =
        std::min(blocksPerCU * cusPerSE / clusterSize, maxInFlightClustersPerSE);
    return clustersPerSE * numSE;
  };

  HIP_CHECK(
      hipOccupancyMaxActiveClusters(&clustersNoLDS, reinterpret_cast<const void*>(f1), &config));
  INFO("occupancy with no shared memory: " << clustersNoLDS);
  REQUIRE(clustersNoLDS > 0);
  REQUIRE(clustersNoLDS == expectedClusters(0));

  // these check that shared memory usage has an effect on occupancy calculations
  for (size_t divisor : {8, 4}) {
    config.dynamicSmemBytes = props.sharedMemPerBlock / divisor;
    HIP_CHECK(
        hipOccupancyMaxActiveClusters(&clustersLDS, reinterpret_cast<const void*>(f1), &config));
    INFO("dynamicSmemBytes: " << config.dynamicSmemBytes << ", occupancy: " << clustersLDS);
    REQUIRE(clustersLDS == expectedClusters(config.dynamicSmemBytes));
    REQUIRE(clustersLDS <= clustersNoLDS);
  }
}

HIP_TEST_CASE(Unit_hipOccupancyMaxActiveClusters_Negative_Zero_Cluster) {
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

HIP_TEST_CASE(Unit_hipOccupancyMaxActiveClusters_Negative_Parameters) {
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
