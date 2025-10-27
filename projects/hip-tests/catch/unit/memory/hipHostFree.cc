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
 *  - memory/hipHostFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipHostFree_InvalidMemory") {
  SECTION("Nullptr") { HIP_CHECK(hipHostFree(nullptr)); }

  SECTION("Invalid ptr") {
    void* invalid_ptr;
    HIP_CHECK_ERROR(hipHostFree(&invalid_ptr), hipErrorInvalidValue);
  }

  SECTION("Host registered memory") {
    constexpr size_t kPtrSize = 1024;
    auto ptr = std::make_unique<char[]>(kPtrSize);
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);

    HIP_CHECK(hipHostRegister(ptr.get(), kPtrSize, flag));
    HIP_CHECK_ERROR(hipHostFree(ptr.get()), hipErrorInvalidValue);
    HIP_CHECK(hipHostUnregister(ptr.get()));
  }

#if (HT_AMD == 1) && (HT_LINUX == 1)
  SECTION("Host registered memory AMD Linux") {
    const size_t ptr_size = 1024;
    char* ptr = new char[ptr_size];
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped, 
                         hipHostRegisterIoMemory);
    HIP_CHECK(hipHostRegister(ptr, ptr_size, flag));
    HIP_CHECK_ERROR(hipHostFree(ptr), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Test verifies that double free returns an error.
 * Test source
 * ------------------------
 *  - memory/hipHostFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipHostFree_DoubleFree") {
  void* ptr = NULL;
  size_t ptr_size = 1024;

  HIP_CHECK(hipHostMalloc(&ptr, ptr_size));
  HIP_CHECK(hipHostFree(ptr));
  HIP_CHECK_ERROR(hipHostFree(ptr), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Caling hipHostFree from different thread on each pointer
 * Test source
 * ------------------------
 *  - memory/hipHostFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipHostFree_Multithreading") {
  std::vector<unsigned long*> ptrs(10);
  size_t ptr_size = 1024;

  for (auto ptr : ptrs) {
    HIP_CHECK(hipHostMalloc(&ptr, ptr_size));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.emplace_back(([ptr] {
      HIP_CHECK_THREAD(hipHostFree(ptr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

TEST_CASE("Unit_hipHostFree_Capture") {
  void* host_ptr = nullptr;
  constexpr size_t kAllocSize = 1024;
  HIP_CHECK(hipHostMalloc(&host_ptr, kAllocSize));

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipHostFree(host_ptr), capture_error);
  END_CAPTURE_SYNC(capture_error);
}
