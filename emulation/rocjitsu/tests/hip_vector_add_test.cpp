// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_vector_add_test.cpp
/// @brief HIP vector_add kernel launch through real ROCR on simulated GPU.
///
/// Compiled with hipcc (HIP language support for __global__ kernels). Requires
/// LD_PRELOAD=librocjitsu.so.

#include <cmath>
#include <cstdlib>
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

template <typename T> __global__ void vector_add(const T *A, const T *B, T *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

template <typename T>
static void run_vector_add(const std::vector<T> &h_A, const std::vector<T> &h_B,
                           std::vector<T> &h_C) {
  const int N = static_cast<int>(h_A.size());
  const size_t bytes = N * sizeof(T);
  T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  HIP_ASSERT(hipMalloc(&d_A, bytes));
  HIP_ASSERT(hipMalloc(&d_B, bytes));
  HIP_ASSERT(hipMalloc(&d_C, bytes));

  HIP_ASSERT(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  int blockSize = 64;
  int gridSize = (N + blockSize - 1) / blockSize;
  vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  (void)hipFree(d_A);
  (void)hipFree(d_B);
  (void)hipFree(d_C);
}

TEST(HipVectorAddTest, CorrectResult) {
  constexpr int N = 1024;
  std::vector<float> h_A(N), h_B(N), h_C(N), h_ref(N);
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i) * 0.1f;
    h_B[i] = static_cast<float>(i) * 0.2f;
    h_ref[i] = h_A[i] + h_B[i];
  }
  ASSERT_NO_FATAL_FAILURE(run_vector_add(h_A, h_B, h_C));

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(h_C[i] - h_ref[i]) > 1e-5f) {
      ++errors;
    }
  }
  EXPECT_EQ(errors, 0) << errors << "/" << N << " elements differ";
}

TEST(HipVectorAddTest, PackedF64) {
  // Clang lowers a pair of doubles to v_pk_add_f64 on gfx1251.
  using Pair = double __attribute__((ext_vector_type(2)));
  constexpr int N = 251;
  std::vector<Pair> a(N), b(N), output(N);
  for (int i = 0; i < N; ++i) {
    a[i] = Pair{2.0 * i + 0.5, -3.0 * i};
    b[i] = Pair{-7.0 * i, 0.25 + 5.0 * i};
  }
  ASSERT_NO_FATAL_FAILURE(run_vector_add(a, b, output));
  for (int i = 0; i < N; ++i) {
    for (int lane = 0; lane < 2; ++lane) {
      ASSERT_EQ(output[i][lane], a[i][lane] + b[i][lane]) << "pair " << i << ", lane " << lane;
    }
  }
}
