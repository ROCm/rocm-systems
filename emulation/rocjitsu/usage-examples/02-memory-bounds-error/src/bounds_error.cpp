// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bounds_error.cpp
/// @brief Demonstrates out-of-bounds memory access bug

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/// Buggy kernel: No bounds check!
__global__ void process_array_buggy(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // BUG: No bounds check - may access beyond array!
  data[i] = static_cast<float>(i) * 2.0f;
}

int main() {
  const int N = 1000;  // Not a multiple of block size!
  const int blockSize = 64;
  const int gridSize = 16;  // 16 * 64 = 1024 > 1000

  const size_t bytes = N * sizeof(float);
  const size_t extra_bytes = (gridSize * blockSize) * sizeof(float);

  std::cout << "Memory Bounds Error Example - BUGGY VERSION" << std::endl;
  std::cout << "  Array size: " << N << " elements" << std::endl;
  std::cout << "  Grid: " << gridSize << " blocks, Block: " << blockSize << " threads" << std::endl;
  std::cout << "  Total threads: " << (gridSize * blockSize) << std::endl;
  std::cout << std::endl;

  if (gridSize * blockSize > N) {
    std::cout << "WARNING: Grid size (" << (gridSize * blockSize)
              << ") > array size (" << N << ")" << std::endl;
    std::cout << "         " << (gridSize * blockSize - N)
              << " threads will access invalid memory!" << std::endl;
    std::cout << std::endl;
  }

  // Allocate extra space to catch out-of-bounds writes
  std::vector<float> h_data(gridSize * blockSize, -1.0f);

  float *d_data = nullptr;
  HIP_CHECK(hipMalloc(&d_data, extra_bytes));
  HIP_CHECK(hipMemset(d_data, 0, extra_bytes));

  std::cout << "Launching kernel..." << std::endl;
  process_array_buggy<<<gridSize, blockSize>>>(d_data, N);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_data.data(), d_data, extra_bytes, hipMemcpyDeviceToHost));

  // Check for out-of-bounds writes
  int errors = 0;
  for (int i = N; i < gridSize * blockSize; ++i) {
    if (h_data[i] != 0.0f) {
      if (errors < 5) {
        std::cout << "ERROR: Out-of-bounds write at index " << i
                  << " = " << h_data[i] << std::endl;
      }
      ++errors;
    }
  }

  std::cout << std::endl;
  if (errors > 0) {
    std::cout << "MEMORY BOUNDS ERROR - " << errors
              << " out-of-bounds writes detected!" << std::endl;
    std::cout << "This is expected - the kernel has no bounds checking." << std::endl;
  } else {
    std::cout << "No out-of-bounds access detected (unexpected)." << std::endl;
  }

  HIP_CHECK(hipFree(d_data));

  return (errors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}