// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gemm_test.cpp
/// @brief Debug a rocBLAS SGEMM under rocjitsu.
///
/// Demonstrates the most common rocBLAS GEMM bug: rocBLAS is COLUMN-major, but
/// C/C++ matrices are usually ROW-major. Calling rocblas_sgemm the "natural"
/// row-major way compiles, runs, returns rocblas_status_success, and produces
/// WRONG numbers. A host golden check catches it; the column-major-aware call
/// fixes it.
///
/// Under RJ_LOG=1, rocjitsu also prints "mfma detected in dispatch N" for the
/// rocBLAS kernel — confirming the GEMM actually uses the matrix cores (MFMA).
/// rocjitsu does not verify numerics for you; the host golden does that.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define ROCBLAS_CHECK(call)                                                    \
  do {                                                                         \
    rocblas_status st = (call);                                                \
    if (st != rocblas_status_success) {                                        \
      std::cerr << "rocBLAS error: " << rocblas_status_to_string(st)           \
                << std::endl;                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Row-major host reference: C(MxN) = A(MxK) * B(KxN).
static void host_gemm(const std::vector<float> &A, const std::vector<float> &B,
                      std::vector<float> &C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

static int count_mismatches(const std::vector<float> &got, const std::vector<float> &ref) {
  int bad = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::abs(got[i] - ref[i]) > 1e-3f * (1.0f + std::abs(ref[i])))
      ++bad;
  return bad;
}

/// Run rocblas_sgemm two ways. Both write row-major C into d_C.
///   correct=false : naive row-major call (BUG) -> computes the wrong product.
///   correct=true  : column-major-aware call    -> computes row-major A*B.
static void run_sgemm(rocblas_handle handle, const float *d_A, const float *d_B,
                      float *d_C, int M, int N, int K, bool correct) {
  const float alpha = 1.0f, beta = 0.0f;
  if (correct) {
    // rocBLAS is column-major. A row-major MxK matrix is, read column-major,
    // its KxM transpose. To get row-major C = A*B we compute C^T = B^T * A^T,
    // which in column-major is: (NxM) = (NxK) * (KxM) with the args swapped.
    ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
  } else {
    // BUG: call it as if rocBLAS were row-major (natural M,N,K, A then B).
    // Compiles and returns success, but computes the wrong product.
    ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N));
  }
}

int main() {
  const int M = 32, N = 32, K = 32;

  std::vector<float> h_A(M * K), h_B(K * N), h_ref(M * N), h_C(M * N);
  // Non-symmetric, non-commuting integer-valued data so layout mistakes show up.
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      h_A[i * K + k] = static_cast<float>((i % 5) - (k % 3));
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      h_B[k * N + j] = static_cast<float>((k % 4) - (j % 6) + 1);

  host_gemm(h_A, h_B, h_ref, M, N, K);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));

  std::cout << "rocBLAS SGEMM debugging: C(" << M << "x" << N << ") = A(" << M << "x" << K
            << ") * B(" << K << "x" << N << ")" << std::endl;
  std::cout << "Run under RJ_LOG=1 to see the GEMM dispatch and 'mfma detected'." << std::endl;
  std::cout << std::endl;

  // 1) Buggy call: treat rocBLAS as row-major.
  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  run_sgemm(handle, d_A, d_B, d_C, M, N, K, /*correct=*/false);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
  int bad_buggy = count_mismatches(h_C, h_ref);
  std::cout << "[buggy]  naive row-major call     -> "
            << (bad_buggy == 0 ? "PASSED" : "FAILED") << " (" << bad_buggy
            << " mismatched elements)" << std::endl;
  std::cout << "         C[0,0]=" << h_C[0] << "  expected=" << h_ref[0] << std::endl;

  // 2) Fixed call: column-major-aware (swap args to compute row-major A*B).
  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  run_sgemm(handle, d_A, d_B, d_C, M, N, K, /*correct=*/true);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
  int bad_fixed = count_mismatches(h_C, h_ref);
  std::cout << "[fixed]  column-major-aware call  -> "
            << (bad_fixed == 0 ? "PASSED" : "FAILED") << " (" << bad_fixed
            << " mismatched elements)" << std::endl;
  std::cout << "         C[0,0]=" << h_C[0] << "  expected=" << h_ref[0] << std::endl;

  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  std::cout << std::endl;
  if (bad_buggy != 0 && bad_fixed == 0) {
    std::cout << "Host golden check caught the column-major bug; the fixed call is correct."
              << std::endl;
    return EXIT_SUCCESS;
  }
  std::cerr << "Unexpected: buggy should FAIL and fixed should PASS." << std::endl;
  return EXIT_FAILURE;
}
