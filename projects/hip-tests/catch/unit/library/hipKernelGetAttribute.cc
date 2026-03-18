/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <utils.hh>

TEST_CASE("Unit_hipKernelGetAttribute_Positive_Basic") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "reverse_kernel.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  int pi;
  int device_id = 0;

  SECTION("binaryVersion") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, device_id));
    const auto major = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMajor, 0);
    const auto minor = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMinor, 0);
    REQUIRE(pi == major * 10 + minor);
  }

  SECTION("cacheModeCA") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel, device_id));
    REQUIRE((pi == 0 || pi == 1));
  }

  SECTION("maxThreadsPerBlock") {
    HIP_CHECK(
        hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel, device_id));
    REQUIRE(pi == GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  }

  SECTION("numRegs") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel, device_id));
    REQUIRE(pi >= 0);
  }

  SECTION("ptxVersion") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel, device_id));
    REQUIRE(pi > 0);
  }

#if !defined(ENABLE_ADDRESS_SANITIZER)
  SECTION("sharedSizeBytes") {
    hipKernel_t kernel;
    HIP_CHECK(hipLibraryGetKernel(&kernel, library, "reverse"));
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, device_id));
    REQUIRE(pi == 256);
  }
#endif

  SECTION("dynamicSharedSizeBytes") {
    hipKernel_t addKernel;
    HIP_CHECK(hipLibraryGetKernel(&addKernel, library, "add_kernel"));
    int addKernelMaxDynamicSharedMemorySize;
    HIP_CHECK(hipKernelGetAttribute(&addKernelMaxDynamicSharedMemorySize, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, addKernel, device_id));

    hipKernel_t reverseKernel;
    HIP_CHECK(hipLibraryGetKernel(&reverseKernel, library, "reverse"));
    int reverseKernelMaxDynamicSharedMemorySize;
    HIP_CHECK(hipKernelGetAttribute(&reverseKernelMaxDynamicSharedMemorySize, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, reverseKernel, device_id));

    REQUIRE(addKernelMaxDynamicSharedMemorySize > reverseKernelMaxDynamicSharedMemorySize);
  }

  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipKernelGetAttribute_Negative_Parameters") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  int pi;
  int device_id = 0;

  SECTION("pi == nullptr") {
    HIP_CHECK_ERROR(
        hipKernelGetAttribute(nullptr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(
        hipKernelGetAttribute(&pi, static_cast<hipFunction_attribute>(-1), kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("kernel == nullptr") {
    HIP_CHECK_ERROR(
        hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, nullptr, device_id),
        hipErrorInvalidHandle);
  }

  HIP_CHECK(hipLibraryUnload(library));
  HIP_CHECK(hipStreamDestroy(stream));
}
