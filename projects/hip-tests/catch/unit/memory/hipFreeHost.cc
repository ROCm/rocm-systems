/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic test that checks behaviour for invalid memory as well as host registered memory.
 * Test source
 * ------------------------
 *  - memory/hipFreeHost.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipFreeHost_InvalidMemory") {
  SECTION("Nullptr") { HIP_CHECK(hipFreeHost(nullptr)); }

  SECTION("Invalid ptr") {
    void* invalid_ptr;
    HIP_CHECK_ERROR(hipFreeHost(&invalid_ptr), hipErrorInvalidValue);
  }

  SECTION("Host registered memory") {
    char* ptr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);

    HIP_CHECK(hipHostRegister(ptr, sizeof(char), flag));
    HIP_CHECK_ERROR(hipFreeHost(ptr), hipErrorInvalidValue);
  }

#if (HT_AMD == 1) && (HT_LINUX == 1)
  SECTION("Host registered memory AMD Linux") {
    char* ptr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped, 
                         hipHostRegisterIoMemory);
    HIP_CHECK(hipHostRegister(ptr, sizeof(char), flag));
    HIP_CHECK_ERROR(hipFreeHost(ptr), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test verifies that double free returns an error.
 * Test source
 * ------------------------
 *  - memory/hipFreeHost.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipFreeHost_DoubleFree") {
  void* ptr = NULL;
  size_t ptr_size = 1024;

  HIP_CHECK(hipHostMalloc(&ptr, ptr_size));
  HIP_CHECK(hipFreeHost(ptr));
  HIP_CHECK_ERROR(hipFreeHost(ptr), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Caling hipFreeHost from different thread on each pointer
 * Test source
 * ------------------------
 *  - memory/hipFreeHost.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipFreeHost_Multithreading") {
  std::vector<unsigned long*> ptrs(10);
  size_t ptr_size = 1024;

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipHostMalloc(&ptr, ptr_size));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.emplace_back(([ptr] {
      HIP_CHECK_THREAD(hipFreeHost(ptr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

TEST_CASE("Unit_hipFreeHost_Capture") {
  void* ptr = nullptr;
  constexpr size_t kPtrSize = 1024;

  HIP_CHECK(hipHostMalloc(&ptr, kPtrSize));

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipFreeHost(ptr), capture_error);
  END_CAPTURE_SYNC(capture_error);
}
