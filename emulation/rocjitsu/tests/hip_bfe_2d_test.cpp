// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression test for v_bfe_u32 (vector bit-field extract).
// A 2D HIP grid uses v_bfe_u32 to extract threadIdx.y from the packed
// thread-ID register. If v_bfe_u32 is broken, threadIdx.y is always 0
// and only half the output is written.

#include <hip/hip_runtime.h>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  _exit(rc);
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

__global__ void write_thread_y(int *out) { out[threadIdx.y] = threadIdx.y; }

TEST(BfeRegressionTest, ThreadIdxY) {
  constexpr int N = 4;

  int *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, N * sizeof(int)));
  HIP_ASSERT(hipMemset(d_out, -1, N * sizeof(int)));

  dim3 block(1, N);
  dim3 grid(1, 1);
  write_thread_y<<<grid, block>>>(d_out);
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<int> h(N);
  HIP_ASSERT(hipMemcpy(h.data(), d_out, N * sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(h[i], i);

  (void)hipFree(d_out);
}
