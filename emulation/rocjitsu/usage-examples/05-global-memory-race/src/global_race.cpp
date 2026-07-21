// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file global_race.cpp
/// @brief Multi-block global counter race — a bug RJ_RACE=1 does NOT catch.
///
/// Many threads across multiple blocks perform a non-atomic read-modify-write
/// on one global counter (`*result += data[idx]`). This is a real data race
/// that corrupts the result, but rocjitsu's race detector does NOT report it:
/// the detector is intra-workgroup only and tracks LDS/VGPR/SGPR sync hazards,
/// not inter-workgroup contention on a global address (see docs/race-detector.md).
///
/// The bug is therefore caught here by a HOST golden check, not by RJ_RACE=1.
/// Contrast with Example 04, whose LDS race IS reported by RJ_RACE=1.

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

/// BUG: non-atomic RMW on a global counter shared by every thread/block.
__global__ void sum_with_race(int *result, const int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    *result += data[idx];  // RACE: read-modify-write with no atomic/serialization
  }
}

int main() {
  const int N = 1000;
  const int blockSize = 100;
  const int gridSize = 10;  // 10 blocks all updating the same global counter

  std::cout << "Global Counter Example - MULTI-BLOCK RACE (host-detected)" << std::endl;
  std::cout << "  Elements: " << N << " (each = 1)" << std::endl;
  std::cout << "  Launch: grid(" << gridSize << ", 1, 1), block(" << blockSize << ", 1, 1)"
            << std::endl;
  std::cout << "  Note: RJ_RACE=1 does NOT flag inter-block global RMW —" << std::endl;
  std::cout << "        this bug is caught by the host golden check below." << std::endl;
  std::cout << std::endl;

  std::vector<int> h_data(N, 1);
  const int expected = N;  // sum of N ones

  int *d_data = nullptr;
  int *d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

  HIP_CHECK(hipMemcpy(d_data, h_data.data(), N * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

  sum_with_race<<<gridSize, blockSize>>>(d_result, d_data, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost));

  std::cout << "Verification: " << (result == expected ? "PASSED" : "FAILED") << std::endl;
  std::cout << "  Expected sum: " << expected << std::endl;
  std::cout << "  Actual sum: " << result << std::endl;
  std::cout << "  Lost updates: " << (expected - result) << std::endl;

  if (result != expected) {
    std::cout << std::endl;
    std::cout << "Lost updates from a non-atomic global RMW. RJ_RACE=1 stays silent"
              << std::endl;
    std::cout << "(inter-workgroup global races are out of scope); fix with atomicAdd"
              << std::endl;
    std::cout << "(see global_fixed.cpp) and re-check the host sum." << std::endl;
  }

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_result));

  return (result == expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
