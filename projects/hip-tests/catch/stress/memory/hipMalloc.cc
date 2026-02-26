/*
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

 #include <hip_test_common.hh>

 #include <algorithm>
 #include <cstdlib>
 #include <ctime>
 #include <execution>
 #include <memory>
 
// Stress allocation tests
// Try to allocate as much memory as possible, backing off gradually on failure.
TEST_CASE("Stress_hipMalloc_HighSizeAlloc") {
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