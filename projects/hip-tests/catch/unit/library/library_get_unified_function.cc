/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Tests for hipLibraryGetUnifiedFunction against the ahead-of-time-compiled code
// object (library_code_load.code, built from library_code_load.cc by the CMake
// custom target). That code object defines ordinary kernels and device variables
// only, so none of its symbols is a unified function on any backend.
//
// On AMD no device supports unified function pointers, so the exact error codes
// are pinned there. On NVIDIA the call forwards to cuLibraryGetUnifiedFunction,
// whose codes for these cases are not documented, so only failure is required.

#include <hip_test_common.hh>

#include <string>

namespace {

const std::string kCodeFile = "library_code_load.code";

}  // namespace

HIP_TEST_CASE(Unit_hipLibraryGetUnifiedFunction_Negative_Parameters) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(
      hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* fptr = nullptr;

  SECTION("null function out-param") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(nullptr, lib, "add_kernel"), hipErrorInvalidValue);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(nullptr, lib, "add_kernel") != hipSuccess);
#endif
  }
  SECTION("null symbol") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(&fptr, lib, nullptr), hipErrorInvalidValue);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(&fptr, lib, nullptr) != hipSuccess);
#endif
  }
  SECTION("empty symbol") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(&fptr, lib, ""), hipErrorInvalidValue);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(&fptr, lib, "") != hipSuccess);
#endif
  }
  SECTION("null library") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(&fptr, nullptr, "add_kernel"),
                    hipErrorInvalidResourceHandle);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(&fptr, nullptr, "add_kernel") != hipSuccess);
#endif
  }

  REQUIRE(fptr == nullptr);
  HIP_CHECK(hipLibraryUnload(lib));
}

// A kernel is not a unified function, and neither is a symbol the library does
// not define, so both lookups must fail without producing a pointer.
HIP_TEST_CASE(Unit_hipLibraryGetUnifiedFunction_Negative_NotFound) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(
      hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  void* fptr = nullptr;

  SECTION("kernel symbol") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(&fptr, lib, "add_kernel"), hipErrorNotFound);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(&fptr, lib, "add_kernel") != hipSuccess);
#endif
  }
  SECTION("unknown symbol") {
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK_ERROR(hipLibraryGetUnifiedFunction(&fptr, lib, "no_such_function"), hipErrorNotFound);
#else
    REQUIRE(hipLibraryGetUnifiedFunction(&fptr, lib, "no_such_function") != hipSuccess);
#endif
  }

  REQUIRE(fptr == nullptr);
  HIP_CHECK(hipLibraryUnload(lib));
}

// hipLibraryGetUnifiedFunction reports not-found on AMD because no AMD device
// supports unified function pointers. Pin the device property it relies on.
HIP_TEST_CASE(Unit_hipLibraryGetUnifiedFunction_NoDeviceReportsSupport) {
#ifdef __HIP_PLATFORM_AMD__
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  for (int device = 0; device < device_count; ++device) {
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    INFO("device " << device);
    REQUIRE(props.unifiedFunctionPointers == 0);
  }
#else
  HIP_SKIP_TEST("Support for unified function pointers varies by NVIDIA device.");
#endif
}
