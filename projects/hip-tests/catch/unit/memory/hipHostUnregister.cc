/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

namespace hipHostUnregisterTests {
constexpr unsigned int allFlags = hipHostRegisterDefault |   // 0
                                  hipHostRegisterPortable |  // 1
                                  hipHostRegisterMapped |    // 2
                                  hipHostRegisterIoMemory    // 4
#if HT_NVIDIA
                                  | cudaHostRegisterReadOnly;  // 8
#else
    ;
#endif

inline bool hipHostRegisterSupported() {
#if HT_NVIDIA
  // unable to query for cudaDevAttrHostRegisterSupported equivalent
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-40");
  HipTest::HIP_SKIP_TEST("hipHostRegister is not supported on this device");
  return false;
#else
  return true;
#endif
}


TEST_CASE("Unit_hipHostUnregister_MemoryNotAccessibleAfterUnregister") {
  if (!hipHostRegisterSupported()) {
    return;
  }
  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
#if defined(_WIN32)
    // hipHostRegisterIoMemory not supported on Windows; skip flags 4-7
    if (flag & hipHostRegisterIoMemory) continue;
#endif
      DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));

      void* device_memory;
      HIP_CHECK(hipHostGetDevicePointer(&device_memory, x.get(), 0));

      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostGetDevicePointer(&device_memory, x.get(), 0), hipErrorInvalidValue);
    }
  }
}

HIP_TEST_CASE(Unit_hipHostUnregister_NullPtr) {
  HIP_CHECK_ERROR(hipHostUnregister(nullptr), hipErrorInvalidValue);
}

HIP_TEST_CASE(Unit_hipHostUnregister_Ptr_Different_Than_Specified_To_Register) {
  std::vector<int> alloc(2);
  HIP_CHECK(hipHostRegister(alloc.data(), alloc.size(), 0));
  HIP_CHECK_ERROR(hipHostUnregister(&alloc.data()[1]), hipErrorHostMemoryNotRegistered);
  HIP_CHECK(hipHostUnregister(alloc.data()));
}

HIP_TEST_CASE(Unit_hipHostUnregister_NotRegisteredPointer) {
  auto x = std::unique_ptr<int>(new int);
  HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
}

HIP_TEST_CASE(Unit_hipHostUnregister_AlreadyUnregisteredPointer) {
  if (!hipHostRegisterSupported()) {
    return;
  }
  // try all combinations of flags
  for (unsigned int flag = 0; flag <= allFlags; ++flag) {
#if defined(_WIN32)
    // hipHostRegisterIoMemory not supported on Windows; skip flags 4-7
    if (flag & hipHostRegisterIoMemory) continue;
#endif
    DYNAMIC_SECTION("Using flag: " << flag) {
      auto x = std::unique_ptr<int>(new int);
      HIP_CHECK(hipHostRegister(x.get(), sizeof(int), flag));
      HIP_CHECK(hipHostUnregister(x.get()));
      HIP_CHECK_ERROR(hipHostUnregister(x.get()), hipErrorHostMemoryNotRegistered);
    }
  }
}

HIP_TEST_CASE(Unit_hipHostUnregister_Capture) {
  constexpr size_t kBufferSize = 1024;
  auto buffer = std::make_unique<int[]>(kBufferSize);
  hipError_t capture_error = hipSuccess;

  HIP_CHECK_ERROR(hipHostRegister(buffer.get(), kBufferSize * sizeof(int), 0), capture_error);

  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipHostUnregister(buffer.get()), capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error != hipSuccess) {
    HIP_CHECK(hipHostUnregister(buffer.get()));
  }
}

}  // namespace hipHostUnregisterTests
