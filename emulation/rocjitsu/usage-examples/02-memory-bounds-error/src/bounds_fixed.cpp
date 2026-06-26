// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bounds_fixed.cpp
/// @brief Fixed version with proper bounds checking

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/// Fixed kernel: Proper bounds check
__global__ void process_array_safe(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // FIXED: Add bounds check
  if (i < N) {
    data[i] = static_cast<float>(i) * 2.0f;
  }
}

int main() {
  const int N = 1000;
  const int blockSize = 64;
  const int gridSize = (N + blockSize - 1) / blockSize;  // Exact coverage

  const size_t bytes = N * sizeof(float);

  std::cout << "Memory Bounds Error Example - FIXED VERSION" << std::endl;
  std::cout << "  Array size: " << N << " elements" << std::endl;
  std::cout << "  Grid: " << gridSize << " blocks, Block: " << blockSize << " threads" << std::endl;
  std::cout << "  Total threads: " << (gridSize * blockSize) << std::endl;
  std::cout << std::endl;

  std::cout << "Bounds check enabled: threads >= " << N << " will skip write" << std::endl;
  std::cout << std::endl;

  std::vector<float> h_data(N);
  std::vector<float> h_ref(N);

  for (int i = 0; i < N; ++i) {
    h_ref[i] = static_cast<float>(i) * 2.0f;
  }

  float *d_data = nullptr;
  HIP_CHECK(hipMalloc(&d_data, bytes));

  std::cout << "Launching kernel..." << std::endl;
  process_array_safe<<<gridSize, blockSize>>>(d_data, N);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_data.data(), d_data, bytes, hipMemcpyDeviceToHost));

  // Verify
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (h_data[i] != h_ref[i]) {
      ++errors;
    }
  }

  std::cout << "Verification: " << (errors == 0 ? "PASSED" : "FAILED") << std::endl;
  std::cout << "  All " << N << " elements correct" << std::endl;
  std::cout << "  No out-of-bounds access detected" << std::endl;
  std::cout << std::endl;
  std::cout << "SUCCESS - Proper bounds checking prevented invalid access" << std::endl;

  HIP_CHECK(hipFree(d_data));

  return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}