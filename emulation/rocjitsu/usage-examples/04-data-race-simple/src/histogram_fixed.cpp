// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file histogram_fixed.cpp
/// @brief Fixed block-local histogram: __syncthreads() before peer LDS read.
///
/// Same structure as histogram_race.cpp but adds the missing barrier before
/// reading another wave's shared slot. RJ_RACE=1 should report no LDS races.
/// For global per-element atomics (inter-workgroup), see Example 05.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void histogram_lds_fixed(const int *data, int *bins, int N, int num_bins) {
  __shared__ int thread_bin[128];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  if (gid < N)
    thread_bin[tid] = data[gid] % num_bins;
  else
    thread_bin[tid] = 0;

  __syncthreads();  // FIXED: all thread_bin[] writes visible before read

  int bin = thread_bin[tid];  // read own slot (correct bin after barrier)
  if (gid < N)
    atomicAdd(&bins[bin], 1);
}

int main(int argc, char **argv) {
  const int N = 10000;
  const int num_bins = 256;
  const int blockSize = 128;
  const int gridSize = (N + blockSize - 1) / blockSize;
  const size_t data_bytes = N * sizeof(int);
  const size_t bins_bytes = num_bins * sizeof(int);

  std::cout << "Histogram Example - FIXED (LDS barrier before peer read)" << std::endl;
  std::cout << "  Input size: " << N << " elements" << std::endl;
  std::cout << "  Number of bins: " << num_bins << std::endl;
  std::cout << "  Block size: " << blockSize << std::endl;
  std::cout << std::endl;

  std::vector<int> h_data(N);
  std::vector<int> h_bins(num_bins, 0);
  std::vector<int> h_ref_bins(num_bins, 0);

  std::srand(42);
  for (int i = 0; i < N; ++i) {
    h_data[i] = std::rand() % num_bins;
    h_ref_bins[h_data[i]]++;
  }

  int *d_data = nullptr;
  int *d_bins = nullptr;

  HIP_CHECK(hipMalloc(&d_data, data_bytes));
  HIP_CHECK(hipMalloc(&d_bins, bins_bytes));

  HIP_CHECK(hipMemcpy(d_data, h_data.data(), data_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_bins, 0, bins_bytes));

  std::cout << "Launching kernel: grid(" << gridSize << ", 1, 1), "
            << "block(" << blockSize << ", 1, 1)" << std::endl;
  std::cout << std::endl;

  histogram_lds_fixed<<<gridSize, blockSize>>>(d_data, d_bins, N, num_bins);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_bins.data(), d_bins, bins_bytes, hipMemcpyDeviceToHost));
  std::cout << std::endl;

  int expected_sum = N;
  int actual_sum = 0;
  int bin_errors = 0;

  for (int i = 0; i < num_bins; ++i) {
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
    std::cout << "NO LDS RACES expected under RJ_RACE=1." << std::endl;
  }

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_bins));

  return (bin_errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
