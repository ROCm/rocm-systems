/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/**
 * Test Description
 * ------------------------
 *  - gfx1201: tall/unaligned H2D hipMemcpy2DAsync must complete. Unpatched HIP
 *    hangs in KernelBlitManager::writeBufferRect pin+copyBufferRect.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync_gfx1201_test.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 *  - gfx1201 (skipped on other ASICs)
 */
HIP_TEST_CASE(Unit_hipMemcpy2DAsync_gfx1201_TallUnalignedH2D) {
  hipDeviceProp_t prop{};
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  const std::string gfxName(prop.gcnArchName);
  if (gfxName.find("gfx1201") == std::string::npos) {
    HIP_SKIP_TEST("gfx1201-only KernelBlit BufferRect hang regression");
  }

  // Geometry from the llama.cpp ggml set_tensor_2d hang class (reduced height).
  constexpr size_t width = 272;
  constexpr size_t height = 4096;
  constexpr size_t pitch = 748;

  void* d = nullptr;
  HIP_CHECK(hipMalloc(&d, pitch * height));
  std::vector<uint8_t> h(pitch * height, 0x5A);
  for (size_t y = 0; y < height; ++y) {
    std::memset(h.data() + y * pitch, static_cast<int>(y & 0xFF), width);
  }

  HIP_CHECK(hipMemcpy2DAsync(d, pitch, h.data(), pitch, width, height, hipMemcpyHostToDevice,
                             hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  std::vector<uint8_t> back(pitch * height, 0);
  HIP_CHECK(hipMemcpy2DAsync(back.data(), pitch, d, pitch, width, height, hipMemcpyDeviceToHost,
                             hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  for (size_t y = 0; y < height; ++y) {
    REQUIRE(std::memcmp(back.data() + y * pitch, h.data() + y * pitch, width) == 0);
  }

  HIP_CHECK(hipFree(d));
}
