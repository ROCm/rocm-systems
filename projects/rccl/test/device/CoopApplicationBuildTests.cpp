/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device tests for nccl_device/coop.h as an application compiles it: RCCL's own
// -DROCM_VERSION is not visible there, so the warp-sync gate comes from HIP.
#undef ROCM_VERSION

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"

static_assert(NCCL_COOP_HAS_WARP_SYNC == (HIP_VERSION_MAJOR >= 7),
              "coop.h must take the warp-sync gate from HIP when ROCM_VERSION is undefined");

namespace RcclUnitTesting {
namespace {

constexpr int kWaves = 4;
// Wave 1 publishes this late, so a block barrier that releases early is seen by thread 0.
constexpr long long kPublishDelayCycles = 200000;

struct CoopThreadCase {
  __device__ static bool participates(int lane) { return lane == 0; }
  __device__ static void sync() { ncclCoopThread().sync(); }
};

struct CoopWarpCase {
  __device__ static bool participates(int) { return true; }
  __device__ static void sync() { ncclCoopWarp().sync(); }
};

struct CoopLanesCase {
  __device__ static bool participates(int lane) { return lane < 2; }
  __device__ static void sync() { ncclCoopLanes(0x3).sync(); }
};

// Only wave 1 runs the sub-warp coop sync. If that sync is a block-wide
// __syncthreads(), it pairs with the block barrier the other waves reach, and
// thread 0 sums the slots before wave 1 has published its own.
template <typename Case>
__global__ void kernelCoopSyncInOneWave(int* out) {
  __shared__ int slots[kWaves];
  const int wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  if (threadIdx.x < kWaves) slots[threadIdx.x] = 0;
  __syncthreads();
  if (wave == 1 && Case::participates(lane)) Case::sync();
  if (lane == 0) {
    if (wave == 1) {
      const long long start = clock64();
      while (clock64() - start < kPublishDelayCycles) {
      }
    }
    slots[wave] = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int w = 0; w < kWaves; ++w) sum += slots[w];
    *out = sum;
  }
}

class CoopApplicationBuildTest : public DeviceTestBase {
protected:
  template <typename Case>
  void runCase() {
    if (HIP_VERSION_MAJOR < 7) {
      GTEST_SKIP() << "HIP < 7 has no __syncwarp; coop sync is block-wide by design";
    }
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    DeviceBuffer<int> out(1);
    ASSERT_NO_FATAL_FAILURE(out.zero());
    kernelCoopSyncInOneWave<Case><<<1, kWaves * prop.warpSize>>>(out.ptr);
    ASSERT_NO_FATAL_FAILURE(syncAndCheck());
    EXPECT_EQ(out.download(), kWaves);
  }
};

TEST_F(CoopApplicationBuildTest, CoopThreadSyncIsNotBlockBarrier) {
  runCase<CoopThreadCase>();
}

TEST_F(CoopApplicationBuildTest, CoopWarpSyncIsNotBlockBarrier) {
  runCase<CoopWarpCase>();
}

TEST_F(CoopApplicationBuildTest, CoopLanesSyncIsNotBlockBarrier) {
  runCase<CoopLanesCase>();
}

} // namespace
} // namespace RcclUnitTesting
