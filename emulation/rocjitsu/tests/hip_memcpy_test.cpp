// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_memcpy_test.cpp
/// @brief Validates hipMemcpy H2D and D2H data correctness on simulated GPU.
///
/// Compiled with hipcc. Requires LD_PRELOAD=librocjitsu.so.

#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#include <cstdint>
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

/// @brief Round-trips a pageable buffer large enough to take HIP's pinned path.
/// @details A staged copy travels through a buffer HIP allocated from the
/// driver, which the daemon has mapped. A pinned one instead has SDMA address
/// the caller's own pageable memory, which the runtime registers through the
/// KFD SVM API -- a registration that creates no shared backing, so the daemon
/// can only service the transfer by reaching into the client. Only the second
/// kind exercises that path.
///
/// @note The transfer size is load-bearing. GPU_PINNED_MIN_XFER_SIZE (64 KiB)
/// is the documented pinning threshold, but a transfer that still fits one
/// staging buffer is staged regardless of it, and GPU_STAGING_BUFFER_SIZE
/// defaults to 1 MiB. Measured against this reproducer with the fix reverted:
/// 1 MiB still passes, 1.5 MiB already hangs. The 4 MiB used here keeps ~2.5x
/// margin over that boundary, so a shift in staging behaviour cannot quietly
/// turn this into a no-op. The margin is nearly free -- it costs ~0.3 s in
/// daemon mode -- whereas shrinking toward 64 KiB stops testing anything.
TEST(HipMemcpyTest, RoundTripPageableAbovePinThreshold) {
  constexpr size_t bytes = 4 * 1024 * 1024;
  constexpr size_t N = bytes / sizeof(int);

  std::vector<int> src(N), dst(N, 0);
  // Knuth's multiplicative hash. The index is narrowed first so the multiply
  // wraps in 32 bits; computing it in size_t would leave a 64-bit product whose
  // top half survives the shift and cannot be represented as an int.
  for (size_t i = 0; i < N; ++i)
    src[i] = static_cast<int>((static_cast<uint32_t>(i) * 2654435761u) >> 1);

  int *d_buf = nullptr;
  HIP_ASSERT(hipMalloc(&d_buf, bytes));

  HIP_ASSERT(hipMemcpy(d_buf, src.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dst.data(), d_buf, bytes, hipMemcpyDeviceToHost));

  size_t mismatches = 0;
  for (size_t i = 0; i < N; ++i)
    mismatches += static_cast<size_t>(dst[i] != src[i]);
  EXPECT_EQ(mismatches, 0u);

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
