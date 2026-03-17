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
 #include <memory>
 
// Stress allocation tests
// Try to allocate as much memory as possible, backing off gradually on failure.
TEST_CASE(Stress_hipMalloc_HighSizeAlloc) {
  size_t devMemTotal{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemTotal));
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemTotal > 0);

  char* d_ptr{nullptr};
  constexpr size_t kMaxRetries = 10;
  size_t counter{0};
  // Reserve a small buffer so the runtime can still create queues / contexts
  devMemFree *= 0.95;
  INFO("Free Mem Available: " << devMemFree << " bytes out of " << devMemTotal << " bytes!");
  while (hipMalloc(&d_ptr, devMemFree) != hipSuccess && devMemFree > 1) {
    counter++;
    devMemFree = static_cast<size_t>(devMemFree * 0.95);  // back off by ~5% each attempt
    INFO("Attempt to allocate " << devMemFree << " bytes out of " << devMemTotal
                                << " bytes failed!");
    REQUIRE(counter <= kMaxRetries);
  }

  // Use a random fill value so repeated runs are cache-unfriendly; this helps
  // surface issues that only appear when memory contents differ between runs.
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  unsigned char fill_val = static_cast<unsigned char>(std::rand() % 255 + 1);
  INFO("Fill value for this run: " << static_cast<int>(fill_val));
 
   HIP_CHECK(hipMemset(d_ptr, fill_val, devMemFree));
   auto ptr = std::unique_ptr<unsigned char[]>{new unsigned char[devMemFree]};
   HIP_CHECK(hipMemcpy(ptr.get(), d_ptr, devMemFree, hipMemcpyDeviceToHost));
   HIP_CHECK(hipFree(d_ptr));
   REQUIRE(std::all_of(std::execution::par_unseq, ptr.get(), ptr.get() + devMemFree,
                        [fill_val](unsigned char n) { return n == fill_val; }));
 }