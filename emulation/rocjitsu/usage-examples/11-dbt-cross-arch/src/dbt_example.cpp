// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_example.cpp
/// @brief A simple HIP kernel compiled for gfx950, used as DBT input.
///
/// This program is built with --offload-arch=gfx950 so its host binary bundles a
/// gfx950 (CDNA4) code object. Example 11 then uses `rj_dbt_translate` to
/// translate that code object to gfx942 (CDNA3) and inspect the result.
///
/// Note: this is an ordinary HIP program. Running it directly on a non-gfx950
/// device does NOT invoke DBT — HIP loads its own fat-binary code object and
/// would fail on an ISA mismatch. DBT translate-and-run is an HSA-level feature
/// (see the README "Running translated code" section); the translation itself is
/// demonstrated by `rj_dbt_translate` here, which needs no GPU.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void scale_kernel(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    data[i] = i * 2.0f;
}

int main() {
  const int N = 256;
  std::vector<float> h_data(N);

  float *d_data = nullptr;
  if (hipMalloc(&d_data, N * sizeof(float)) != hipSuccess) {
    std::cerr << "hipMalloc failed" << std::endl;
    return 1;
  }

  scale_kernel<<<(N + 63) / 64, 64>>>(d_data, N);
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "kernel launch/sync failed" << std::endl;
    return 1;
  }

  (void)hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < N; ++i)
    if (h_data[i] != i * 2.0f) {
      correct = false;
      break;
    }

  std::cout << "scale_kernel (gfx950): " << (correct ? "PASSED" : "FAILED") << std::endl;

  (void)hipFree(d_data);
  return correct ? 0 : 1;
}
