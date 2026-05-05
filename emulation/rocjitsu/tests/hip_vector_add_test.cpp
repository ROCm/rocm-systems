// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_vector_add_test.cpp
/// @brief HIP vector_add kernel launch through real ROCR on simulated GPU.
///
/// Compiled with hipcc (HIP language support for __global__ kernels). Requires
/// LD_PRELOAD=librocjitsu_kmd.so and RJ_CONFIG/RJ_SCHEMA env vars.

#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

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

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

TEST(HipVectorAddTest, CorrectResult) {
  constexpr int N = 1024;
  constexpr size_t bytes = N * sizeof(float);

  std::vector<float> h_A(N), h_B(N), h_ref(N);
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i) * 0.1f;
    h_B[i] = static_cast<float>(i) * 0.2f;
    h_ref[i] = h_A[i] + h_B[i];
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  HIP_ASSERT(hipMalloc(&d_A, bytes));
  HIP_ASSERT(hipMalloc(&d_B, bytes));
  HIP_ASSERT(hipMalloc(&d_C, bytes));

  HIP_ASSERT(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  int blockSize = 64;
  int gridSize = (N + blockSize - 1) / blockSize;
  vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<float> h_C(N);
  HIP_ASSERT(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(h_C[i] - h_ref[i]) > 1e-5f)
      ++errors;
  }
  EXPECT_EQ(errors, 0) << errors << "/" << N << " elements differ";

  (void)hipFree(d_A);
  (void)hipFree(d_B);
  (void)hipFree(d_C);
}
