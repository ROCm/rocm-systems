// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file global_fixed.cpp
/// @brief Fixed global counter: atomicAdd serializes the read-modify-write.
///
/// Same multi-block reduction as global_race.cpp, but each update uses
/// atomicAdd so no increments are lost. The host golden check passes. Note
/// that RJ_RACE=1 is silent for BOTH versions — it does not track global
/// contention — so the host check is what proves the fix (see docs/race-detector.md).

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

/// FIXED: atomicAdd makes each contribution an indivisible read-modify-write.
__global__ void sum_atomic(int *result, const int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(result, data[idx]);  // SAFE: hardware-serialized RMW
  }
}

int main() {
  const int N = 1000;
  const int blockSize = 100;
  const int gridSize = 10;

  std::cout << "Global Counter Example - FIXED (atomicAdd)" << std::endl;
  std::cout << "  Elements: " << N << " (each = 1)" << std::endl;
  std::cout << "  Launch: grid(" << gridSize << ", 1, 1), block(" << blockSize << ", 1, 1)"
            << std::endl;
  std::cout << std::endl;

  std::vector<int> h_data(N, 1);
  const int expected = N;

  int *d_data = nullptr;
  int *d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

  HIP_CHECK(hipMemcpy(d_data, h_data.data(), N * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

  sum_atomic<<<gridSize, blockSize>>>(d_result, d_data, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost));

  std::cout << "Verification: " << (result == expected ? "PASSED" : "FAILED") << std::endl;
  std::cout << "  Expected sum: " << expected << std::endl;
  std::cout << "  Actual sum: " << result << std::endl;

  if (result == expected)
    std::cout << "  All updates accounted for (atomicAdd)." << std::endl;

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_result));

  return (result == expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
