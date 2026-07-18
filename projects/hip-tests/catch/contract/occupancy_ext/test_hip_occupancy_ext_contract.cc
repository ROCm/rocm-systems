/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
__global__ void OccupancyExtKernel(int* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && output != nullptr) {
    *output = 1;
  }
}
}  // namespace

HIP_TEST_CASE(Contract_OccupancyExt_MaxActiveBlocksPerMultiprocessorWithFlags_DefaultMatchesPlain) {
  // The default-flags occupancy query must agree with the non-flags query for
  // the same kernel and block size. The runtime documents that the default
  // occupancy flag is the baseline behavior, so both entry points must be
  // consistent.
  constexpr int kBlockSize = 64;
  constexpr size_t kDynamicSmem = 0;

  int num_blocks_plain = -1;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_plain, OccupancyExtKernel,
                                                         kBlockSize, kDynamicSmem));
  REQUIRE(num_blocks_plain > 0);

  int num_blocks_with_flags = -1;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &num_blocks_with_flags, OccupancyExtKernel, kBlockSize, kDynamicSmem, hipOccupancyDefault));
  REQUIRE(num_blocks_with_flags > 0);

  REQUIRE(num_blocks_with_flags == num_blocks_plain);
}

HIP_TEST_CASE(
    Contract_OccupancyExt_MaxActiveBlocksPerMultiprocessorWithFlags_DisableCachingOverrideSucceeds) {
  // The disable-caching-override flag must be accepted by the with-flags query
  // and must return a non-negative occupancy. The value may legitimately differ
  // from the default-flags query, so no equality is asserted.
  constexpr int kBlockSize = 64;
  constexpr size_t kDynamicSmem = 0;

  int num_blocks = -1;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &num_blocks, OccupancyExtKernel, kBlockSize, kDynamicSmem,
      hipOccupancyDisableCachingOverride));

  REQUIRE(num_blocks >= 0);
}

HIP_TEST_CASE(Contract_OccupancyExt_MaxActiveBlocksPerMultiprocessorWithFlags_RejectsInvalidInputs) {
  // A null output pointer and an invalid flag value must both be rejected with
  // hipErrorInvalidValue.
  constexpr int kBlockSize = 64;
  constexpr size_t kDynamicSmem = 0;

  HIP_CHECK_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                      nullptr, OccupancyExtKernel, kBlockSize, kDynamicSmem, hipOccupancyDefault),
                  hipErrorInvalidValue);

  int num_blocks = -1;
  const unsigned int invalid_flags = 0xFFFFFFFFu;
  HIP_CHECK_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                      &num_blocks, OccupancyExtKernel, kBlockSize, kDynamicSmem, invalid_flags),
                  hipErrorInvalidValue);
}

// The cluster-occupancy queries (hipOccupancyMaxPotentialClusterSize,
// hipOccupancyMaxActiveClusters) and the hipLaunchAttributeClusterDimension
// attribute are AMD-only in this HIP release; the NVIDIA backend headers do not
// define them, so this contract builds only on AMD. The per-multiprocessor
// occupancy contracts above are portable.
#if HT_AMD
HIP_TEST_CASE(Contract_OccupancyExt_ClusterQueries_CapabilityGatedRange) {
  int current_device = 0;
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDevice(&current_device));
  HIP_CHECK(hipGetDeviceProperties(&props, current_device));

  if (!props.clusterLaunch) {
    HIP_SKIP_TEST("Cluster launch is not supported on this device.");
    return;
  }

  // Build a minimal launch configuration with a single cluster-dimension
  // attribute. The cluster queries must succeed and report a usable range: a
  // maximum cluster size of at least one block and a non-negative count of
  // concurrently active clusters.
  hipLaunchAttribute cluster_attr{};
  cluster_attr.id = hipLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = 1;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;

  hipLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(128, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  const void* kernel = reinterpret_cast<const void*>(&OccupancyExtKernel);

  int cluster_size = -1;
  HIP_CHECK(hipOccupancyMaxPotentialClusterSize(&cluster_size, kernel, &config));
  REQUIRE(cluster_size >= 1);

  int num_clusters = -1;
  HIP_CHECK(hipOccupancyMaxActiveClusters(&num_clusters, kernel, &config));
  REQUIRE(num_clusters >= 0);
}
#endif  // HT_AMD
