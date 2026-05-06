// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_memcpy_test.cpp
/// @brief Validates hipMemcpy H2D and D2H data correctness on simulated GPU.
///
/// Compiled with hipcc. Requires LD_PRELOAD=librocjitsu_kmd.so and
/// RJ_CONFIG env var.

#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#include <cstdlib>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

TEST(HipMemcpyTest, RoundTripFloat) {
  constexpr int N = 16;
  constexpr size_t bytes = N * sizeof(float);

  std::vector<float> src(N), dst(N, 0.0f);
  for (int i = 0; i < N; ++i)
    src[i] = static_cast<float>(i + 1) * 1.5f;

  float *d_buf = nullptr;
  HIP_ASSERT(hipMalloc(&d_buf, bytes));

  HIP_ASSERT(hipMemcpy(d_buf, src.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dst.data(), d_buf, bytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(dst[i], src[i]) << "mismatch at index " << i;

  (void)hipFree(d_buf);
}

TEST(HipMemcpyTest, RoundTripInt) {
  constexpr int N = 256;
  constexpr size_t bytes = N * sizeof(int);

  std::vector<int> src(N), dst(N, 0);
  for (int i = 0; i < N; ++i)
    src[i] = i * 42 + 7;

  int *d_buf = nullptr;
  HIP_ASSERT(hipMalloc(&d_buf, bytes));

  HIP_ASSERT(hipMemcpy(d_buf, src.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dst.data(), d_buf, bytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(dst[i], src[i]) << "mismatch at index " << i;

  (void)hipFree(d_buf);
}

TEST(HipMemcpyTest, DeviceToDevice) {
  constexpr int N = 64;
  constexpr size_t bytes = N * sizeof(float);

  std::vector<float> src(N), dst(N, 0.0f);
  for (int i = 0; i < N; ++i)
    src[i] = static_cast<float>(i) * 3.14f;

  float *d_src = nullptr, *d_dst = nullptr;
  HIP_ASSERT(hipMalloc(&d_src, bytes));
  HIP_ASSERT(hipMalloc(&d_dst, bytes));

  HIP_ASSERT(hipMemcpy(d_src, src.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice));
  HIP_ASSERT(hipMemcpy(dst.data(), d_dst, bytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(dst[i], src[i]) << "mismatch at index " << i;

  (void)hipFree(d_src);
  (void)hipFree(d_dst);
}
