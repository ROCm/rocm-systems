// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_gws_test.cpp
/// @brief HIP cooperative grid synchronization through real ROCr and the simulated GPU.

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

// ROCR keeps process-lifetime runtime state allocated after hipDeviceReset().
// Ignore only allocations whose stack includes the external HSA runtime while
// retaining LeakSanitizer coverage for rocjitsu and this test executable.
extern "C" const char *__lsan_default_suppressions() { return "leak:libhsa-runtime64.so\n"; }

// ROCR serializes its async-event pool with rocr::HybridMutex, an atomic CAS
// spinlock. The external HSA runtime is not instrumented, so ThreadSanitizer
// cannot observe that happens-before edge and reports the pool's heap reuse as
// a race. Ignore only interceptors called from the HSA runtime while retaining
// ThreadSanitizer coverage for rocjitsu and this test executable.
extern "C" const char *__tsan_default_suppressions() {
  return "called_from_lib:libhsa-runtime64.so\n";
}

// gfx90a uses CLR's GWS init kernel and compiler-generated GWS barriers.
__global__ void cooperative_exchange(int *values, unsigned iterations) {
  const auto grid = cooperative_groups::this_grid();
  unsigned acc = threadIdx.x;
  if (blockIdx.x == 0)
    for (unsigned i = 0; i < iterations; ++i) {
      acc = acc * 1664525u + 1013904223u;
      asm volatile("" : "+v"(acc));
    }
  if (threadIdx.x == 0)
    values[blockIdx.x] = blockIdx.x + 1;
  grid.sync();
  if (threadIdx.x == 0)
    values[gridDim.x + blockIdx.x] = values[(blockIdx.x + 1) % gridDim.x];
}

TEST(HipGwsTest, CooperativeGridPublishesBeforeNeighborReads) {
  constexpr int blocks = 8;
  int supported = 0;
  ASSERT_EQ(hipDeviceGetAttribute(&supported, hipDeviceAttributeCooperativeLaunch, 0), hipSuccess);
  ASSERT_EQ(supported, 1);
  for (unsigned threads : {64u, 256u}) {
    SCOPED_TRACE(threads);
    int *device = nullptr;
    ASSERT_EQ(hipMalloc(&device, 2 * blocks * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMemset(device, 0, 2 * blocks * sizeof(int)), hipSuccess);
    unsigned iterations = 300000;
    void *args[] = {&device, &iterations};
    ASSERT_EQ(hipLaunchCooperativeKernel(reinterpret_cast<const void *>(cooperative_exchange),
                                         dim3(blocks), dim3(threads), args, 0, nullptr),
              hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<int> values(2 * blocks);
    ASSERT_EQ(hipMemcpy(values.data(), device, values.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    for (int i = 0; i < blocks; ++i) {
      EXPECT_EQ(values[i], i + 1);
      EXPECT_EQ(values[blocks + i], (i + 1) % blocks + 1);
    }
    EXPECT_EQ(hipFree(device), hipSuccess);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return result;
}
