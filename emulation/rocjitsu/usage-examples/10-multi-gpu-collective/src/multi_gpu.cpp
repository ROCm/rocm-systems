// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file multi_gpu.cpp
/// @brief Hand-written multi-GPU all-reduce under rocjitsu (no RCCL).
///
/// rocjitsu emulates a multi-GPU system (device enumeration, per-device memory
/// and compute), but RCCL collectives do NOT run under it: RCCL's inter-GPU
/// transport fails during communicator init (`amdgpu_bo_import` error -> abort),
/// even with HSA_NO_SCRATCH_RECLAIM=1. So this implements all-reduce manually:
///
///   1. per-GPU compute  - each GPU fills a buffer with its rank value (r+1)
///   2. host reduce       - copy each GPU's buffer to the host and sum
///   3. broadcast         - copy the reduced result back to every GPU
///   4. verify            - every GPU now holds the sum (an all-reduce)
///
/// This uses only basic HIP (hipSetDevice / hipMalloc / hipMemcpy + a kernel per
/// GPU), all of which the emulator supports. Run with the 2-GPU config.

#include <hip/hip_runtime.h>
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

// Per-GPU compute: fill the buffer with this rank's contribution.
__global__ void fill_kernel(float *buf, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    buf[i] = value;
}

int main() {
  const int count = 8;

  int ngpu = 0;
  HIP_CHECK(hipGetDeviceCount(&ngpu));

  std::cout << "Multi-GPU manual all-reduce (no RCCL)" << std::endl;
  std::cout << "  GPUs detected: " << ngpu << "   elements: " << count << std::endl;
  std::cout << std::endl;

  if (ngpu < 2) {
    std::cout << "Need >= 2 GPUs. Run with configs/amdgpu_cdna4_kmd_2gpu.json." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<float *> d_buf(ngpu, nullptr);

  // 1) Per-GPU compute: GPU r fills its buffer with (r + 1).
  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&d_buf[r], count * sizeof(float)));
    fill_kernel<<<(count + 63) / 64, 64>>>(d_buf[r], count, static_cast<float>(r + 1));
    HIP_CHECK(hipGetLastError());
  }
  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  GPU " << r << ": filled with " << (r + 1) << std::endl;
  }

  // 2) Host reduce: sum every GPU's buffer element-wise.
  std::vector<float> acc(count, 0.0f), tmp(count);
  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMemcpy(tmp.data(), d_buf[r], count * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i)
      acc[i] += tmp[i];
  }
  const float expected = static_cast<float>(ngpu * (ngpu + 1) / 2);  // sum of 1..ngpu
  std::cout << std::endl;
  std::cout << "  host-reduced value: " << acc[0] << " (expect " << expected << ")" << std::endl;

  // 3) Broadcast the reduced result back to every GPU.
  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMemcpy(d_buf[r], acc.data(), count * sizeof(float), hipMemcpyHostToDevice));
  }

  // 4) Verify: every GPU now holds the all-reduced sum.
  bool ok = true;
  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMemcpy(tmp.data(), d_buf[r], count * sizeof(float), hipMemcpyDeviceToHost));
    bool gpu_ok = true;
    for (int i = 0; i < count; ++i)
      if (tmp[i] != expected) {
        gpu_ok = false;
        break;
      }
    std::cout << "  GPU " << r << " after all-reduce: " << tmp[0]
              << (gpu_ok ? "  OK" : "  MISMATCH") << std::endl;
    ok = ok && gpu_ok;
  }

  for (int r = 0; r < ngpu; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipFree(d_buf[r]));
  }

  std::cout << std::endl;
  std::cout << "All-reduce " << (ok ? "PASSED" : "FAILED")
            << " (every GPU holds the sum across ranks)" << std::endl;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
