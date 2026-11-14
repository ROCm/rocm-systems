/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_temp_path.hh>

#include <string>

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

#if HT_AMD
HIP_TEST_CASE(Unit_hipLibraryLoadFromFile_Negative_FileAccess) {
  hipLibrary_t lib = nullptr;

  SECTION("null file name") {
    HIP_CHECK_ERROR(hipLibraryLoadFromFile(&lib, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
                    hipErrorInvalidValue);
  }

  SECTION("missing file") {
    const std::string path = hip_test::TempPath("hip-library-missing", ".code");
    HIP_CHECK_ERROR(
        hipLibraryLoadFromFile(&lib, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0),
        hipErrorInvalidValue);
  }

  // A file name longer than NAME_MAX makes the existence check itself fail, which must be reported
  // as hipErrorInvalidValue rather than escape as an exception.
  SECTION("existence check error") {
    const std::string path = (fs::temp_directory_path() / std::string(300, 'a')).string();
    HIP_CHECK_ERROR(
        hipLibraryLoadFromFile(&lib, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0),
        hipErrorInvalidValue);
  }

  REQUIRE(lib == nullptr);
}
#endif
