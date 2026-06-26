// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector_add.cpp
/// @brief Basic HIP vector addition example for rocjitsu debugging
///
/// This example demonstrates:
/// - Simple HIP kernel launch
/// - Host-device memory transfers
/// - Result verification
/// - Basic error checking

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Macro for checking HIP errors
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << std::endl;\
      std::cerr << "  " << hipGetErrorString(err) << std::endl;                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/// GPU kernel: element-wise vector addition
/// Each thread computes one element: C[i] = A[i] + B[i]
__global__ void vector_add_kernel(const float *A, const float *B, float *C, int N) {
  // Calculate global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check to avoid out-of-bounds access
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char **argv) {
  // Vector size (can be overridden via command line)
  int N = 1024;
  if (argc > 1) {
    N = std::atoi(argv[1]);
    if (N <= 0) {
      std::cerr << "Error: vector size must be positive" << std::endl;
      return EXIT_FAILURE;
    }
  }

  const size_t bytes = N * sizeof(float);

  std::cout << "Vector addition: C = A + B" << std::endl;
  std::cout << "  Vector size: " << N << " elements (" << bytes << " bytes)" << std::endl;
  std::cout << std::endl;

  // Allocate host memory
  std::vector<float> h_A(N);
  std::vector<float> h_B(N);
  std::vector<float> h_C(N);
  std::vector<float> h_ref(N);

  // Initialize input vectors
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i) * 0.1f;
    h_B[i] = static_cast<float>(i) * 0.2f;
    h_ref[i] = h_A[i] + h_B[i];  // CPU reference
  }

  // Allocate device memory
  std::cout << "Allocating device memory..." << std::endl;
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));

  // Copy input data to device
  std::cout << "Copying input data to device..." << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  // Launch kernel
  // Grid configuration: enough blocks to cover all elements
  const int blockSize = 64;  // Threads per block
  const int gridSize = (N + blockSize - 1) / blockSize;  // Blocks in grid

  std::cout << "Launching kernel: grid(" << gridSize << ", 1, 1), "
            << "block(" << blockSize << ", 1, 1)" << std::endl;

  vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

  // Check for kernel launch errors
  HIP_CHECK(hipGetLastError());

  // Wait for kernel to complete
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back to host
  std::cout << "Copying results back to host..." << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
  std::cout << std::endl;

  // Verify results
  int errors = 0;
  const float tolerance = 1e-5f;

  for (int i = 0; i < N; ++i) {
    if (std::fabs(h_C[i] - h_ref[i]) > tolerance) {
      if (errors < 10) {  // Print first 10 errors only
        std::cerr << "Error at index " << i << ": "
                  << "GPU=" << h_C[i] << ", "
                  << "CPU=" << h_ref[i] << ", "
                  << "diff=" << std::fabs(h_C[i] - h_ref[i]) << std::endl;
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << "Verification: PASSED" << std::endl;
    std::cout << "  All " << N << " elements correct!" << std::endl;
    std::cout << "  Sample: C[0] = " << h_C[0]
              << " (A[0]=" << h_A[0] << " + B[0]=" << h_B[0] << ")" << std::endl;
    if (N > 512) {
      std::cout << "  Sample: C[512] = " << h_C[512]
                << " (A[512]=" << h_A[512] << " + B[512]=" << h_B[512] << ")" << std::endl;
    }
  } else {
    std::cerr << "Verification: FAILED" << std::endl;
    std::cerr << "  " << errors << "/" << N << " elements differ" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << std::endl;

  // Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  std::cout << "Cleanup complete." << std::endl;

  return EXIT_SUCCESS;
}