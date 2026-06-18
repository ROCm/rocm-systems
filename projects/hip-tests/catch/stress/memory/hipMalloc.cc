/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

 #include <hip_test_common.hh>

 #include <algorithm>
 #include <cstdlib>
 #include <ctime>
 #include <execution>
 #include <iostream>
 #include <memory>

// Stress allocation tests
// Try to allocate as much memory as possible, backing off gradually on failure.
// Level 0 (quick level): cap allocation and host staging; level 2 uses ~95% free VRAM.
HIP_TEST_CASE(Stress_hipMalloc_HighSizeAlloc) {
  size_t devMemTotal{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemTotal));
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemTotal > 0);

  constexpr size_t kQuickAllocCap = 32u * 1024u * 1024u;

  size_t alloc_size = static_cast<size_t>(devMemFree * 0.95);
  if (isQuickLevel()) {
    alloc_size = std::min(alloc_size, kQuickAllocCap);
  }
  alloc_size = std::max(alloc_size, size_t{1});

  char* d_ptr{nullptr};
  constexpr size_t kMaxRetries = 10;
  size_t counter{0};
  std::cout << "[Stress_hipMalloc_HighSizeAlloc] Free Mem Available: " << devMemFree
            << " bytes out of " << devMemTotal << " bytes" << std::endl;
  std::cout << "[Stress_hipMalloc_HighSizeAlloc] Target allocation: " << alloc_size << " bytes"
            << (isQuickLevel() ? " (level_0 cap)" : "") << std::endl;
  while (hipMalloc(&d_ptr, alloc_size) != hipSuccess && alloc_size > 1) {
    counter++;
    alloc_size = static_cast<size_t>(alloc_size * 0.95);
    std::cout << "[Stress_hipMalloc_HighSizeAlloc] Attempt to allocate " << alloc_size
              << " bytes out of " << devMemTotal << " bytes failed!" << std::endl;
    REQUIRE(counter <= kMaxRetries);
  }
  REQUIRE(d_ptr != nullptr);

  // Use a random fill value so repeated runs are cache-unfriendly; this helps
  // surface issues that only appear when memory contents differ between runs.
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  unsigned char fill_val = static_cast<unsigned char>(std::rand() % 255 + 1);
  std::cout << "[Stress_hipMalloc_HighSizeAlloc] Fill value for this run: " << static_cast<int>(fill_val)
            << std::endl;

  HIP_CHECK(hipMemset(d_ptr, fill_val, alloc_size));
  auto ptr = std::unique_ptr<unsigned char[]>{new unsigned char[alloc_size]};
  HIP_CHECK(hipMemcpy(ptr.get(), d_ptr, alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_ptr));
  REQUIRE(std::all_of(std::execution::par_unseq, ptr.get(), ptr.get() + alloc_size,
                       [fill_val](unsigned char n) { return n == fill_val; }));
}
