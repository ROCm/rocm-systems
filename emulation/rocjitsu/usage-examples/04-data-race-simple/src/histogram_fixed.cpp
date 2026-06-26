// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file histogram_fixed.cpp
/// @brief Histogram example FIXED with atomic operations
///
/// This example demonstrates the correct way to implement histogram
/// using atomic operations to avoid race conditions.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/// GPU kernel: Compute histogram (FIXED with atomics)
/// Uses atomicAdd to safely increment histogram bins from multiple threads
__global__ void histogram_atomic(const int *data, int *bins, int N, int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    int bin = data[i] % num_bins;

    // FIXED: Use atomic operation for thread-safe update
    // atomicAdd performs read-modify-write as a single atomic operation
    atomicAdd(&bins[bin], 1);
  }
}

int main(int argc, char **argv) {
  // Parameters
  const int N = 10000;          // Number of elements
  const int num_bins = 256;     // Number of histogram bins
  const size_t data_bytes = N * sizeof(int);
  const size_t bins_bytes = num_bins * sizeof(int);

  std::cout << "Histogram Example - FIXED WITH ATOMICS" << std::endl;
  std::cout << "  Input size: " << N << " elements" << std::endl;
  std::cout << "  Number of bins: " << num_bins << std::endl;
  std::cout << std::endl;

  // Allocate host memory
  std::vector<int> h_data(N);
  std::vector<int> h_bins(num_bins, 0);
  std::vector<int> h_ref_bins(num_bins, 0);

  // Initialize data with random values (same seed as buggy version)
  std::srand(42);
  for (int i = 0; i < N; ++i) {
    h_data[i] = std::rand() % num_bins;
    h_ref_bins[h_data[i]]++;  // CPU reference histogram
  }

  // Allocate device memory
  std::cout << "Allocating memory..." << std::endl;
  int *d_data = nullptr;
  int *d_bins = nullptr;

  HIP_CHECK(hipMalloc(&d_data, data_bytes));
  HIP_CHECK(hipMalloc(&d_bins, bins_bytes));

  // Copy data to device
  std::cout << "Copying data to device..." << std::endl;
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), data_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_bins, 0, bins_bytes));  // Zero histogram

  // Launch kernel
  const int blockSize = 64;
  const int gridSize = (N + blockSize - 1) / blockSize;

  std::cout << "Launching kernel: grid(" << gridSize << ", 1, 1), "
            << "block(" << blockSize << ", 1, 1)" << std::endl;
  std::cout << std::endl;

  histogram_atomic<<<gridSize, blockSize>>>(d_data, d_bins, N, num_bins);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back
  std::cout << "Copying results back..." << std::endl;
  HIP_CHECK(hipMemcpy(h_bins.data(), d_bins, bins_bytes, hipMemcpyDeviceToHost));
  std::cout << std::endl;

  // Verify results
  int expected_sum = 0;
  int actual_sum = 0;
  int bin_errors = 0;

  for (int i = 0; i < num_bins; ++i) {
    expected_sum += h_ref_bins[i];
    actual_sum += h_bins[i];

    if (h_bins[i] != h_ref_bins[i]) {
      ++bin_errors;
      if (bin_errors <= 5) {
        std::cout << "  Bin " << i << ": expected=" << h_ref_bins[i]
                  << ", actual=" << h_bins[i] << std::endl;
      }
    }
  }

  std::cout << "Verification: " << (bin_errors == 0 ? "PASSED" : "FAILED") << std::endl;
  std::cout << "  Expected sum: " << expected_sum << std::endl;
  std::cout << "  Actual sum: " << actual_sum << std::endl;

  if (bin_errors == 0) {
    std::cout << "  All histogram bins correct!" << std::endl;
    std::cout << std::endl;
    std::cout << "NO RACES DETECTED - Results are correct!" << std::endl;
    std::cout << "Atomic operations successfully prevented race conditions." << std::endl;
  } else {
    std::cout << "  Mismatched bins: " << bin_errors << std::endl;
  }

  // Cleanup
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_bins));

  return (bin_errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}