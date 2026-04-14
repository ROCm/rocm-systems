/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
TEST_CASE(Unit_hipFreeHost_InvalidMemory) {
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
    HIP_CHECK(hipHostUnregister(ptr));
    delete ptr;
  }

#if (HT_AMD == 1) && (HT_LINUX == 1)
  SECTION("Host registered memory AMD Linux") {
    char* ptr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped,
                         hipHostRegisterIoMemory);
    HIP_CHECK(hipHostRegister(ptr, sizeof(char), flag));
    HIP_CHECK_ERROR(hipFreeHost(ptr), hipErrorInvalidValue);
    HIP_CHECK(hipHostUnregister(ptr));
    delete ptr;
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
TEST_CASE(Unit_hipFreeHost_DoubleFree) {
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
TEST_CASE(Unit_hipFreeHost_Multithreading) {
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

TEST_CASE(Unit_hipFreeHost_Capture_negative) {
  void* ptr = nullptr;
  constexpr size_t kPtrSize = 1024;

  HIP_CHECK(hipHostMalloc(&ptr, kPtrSize));

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipFreeHost(ptr), capture_error);
  END_CAPTURE_SYNC(capture_error);
}
