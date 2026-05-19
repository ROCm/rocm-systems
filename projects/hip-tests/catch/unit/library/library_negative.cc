/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_library_negative) {
  SECTION("load negative") {
    HIP_CHECK_ERROR(hipLibraryLoadData(nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipLibraryLoadFromFile(nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipLibraryUnload(nullptr), hipErrorInvalidResourceHandle);
    HIP_CHECK_ERROR(hipLibraryGetKernel(nullptr, nullptr, nullptr), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipLibraryGetKernelCount(nullptr, nullptr), hipErrorInvalidValue);
  }

  SECTION("Load random code") {
    const char* code =  // definitely not compile-able
        "Call me Ishmael. Some years ago—never mind how long precisely";
    hipLibrary_t lib;
    hipKernel_t kernel;
    // Default behavior is lazy load, so if we pass anything to it, it should pass
    HIP_CHECK(hipLibraryLoadData(&lib, code, nullptr, nullptr, 0, nullptr, nullptr, 0));
    // But this check will fail
    HIP_CHECK_ERROR(hipLibraryGetKernel(&kernel, lib, "moby"), hipErrorInvalidImage);
    HIP_CHECK(hipLibraryUnload(lib));
  }
}
