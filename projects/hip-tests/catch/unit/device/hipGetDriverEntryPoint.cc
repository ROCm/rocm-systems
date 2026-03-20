/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * hipError_t hipGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                   hipDriverEntryPointQueryResult* driverStatus);
 * Gets function pointer of a request HIP API
 *
 * @param [in]  symbol  The API base name
 * @param [out] funcPtr Pinter to the requested function
 * @param [in]  flags Flags for the search
 * @param [out] driverStatus Optional returned status of the search
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */

/**
 * Test Description
 * ------------------------
 *  - This will perfrom the funtionality testing of hipGetDriverEntryPoint api
 * Test source
 * ------------------------
 *  - unit/device/hipGetDriverEntryPoint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipGetDriverEntryPoint_Positive) {
  void* funcPtr = nullptr;
  hipDriverEntryPointQueryResult status;

  SECTION("hipEnableDefault search flag") {
    HIP_CHECK(hipGetDriverEntryPoint("hipGetDeviceCount", &funcPtr, hipEnableDefault, &status));
  }

  SECTION("hipEnableLegacyStream search flag") {
    HIP_CHECK(
        hipGetDriverEntryPoint("hipGetDeviceCount", &funcPtr, hipEnableLegacyStream, &status));
  }

  SECTION("hipEnablePerThreadDefaultStream search flag") {
    HIP_CHECK(hipGetDriverEntryPoint("hipGetDeviceCount", &funcPtr, hipEnablePerThreadDefaultStream,
                                     &status));
  }

  REQUIRE(status == hipDriverEntryPointSuccess);

  hipError_t (*hipGetDeviceCount_ptr)(int*) = (hipError_t(*)(int*))funcPtr;
  int countFuncPtr;
  HIP_CHECK(hipGetDeviceCount_ptr(&countFuncPtr));

  int count;
  HIP_CHECK(hipGetDeviceCount(&count));

  REQUIRE(count > 0);
  REQUIRE(countFuncPtr == count);
}

/**
 * Test Description
 * ------------------------
 *  - This tests checks hipGetDriverEntryPoint api with negative parameters
 *  # symbol is empty
 *  # funcPtr pointer is null
 *  # Invalid flag
 * Test source
 * ------------------------
 *  - unit/device/hipGetDriverEntryPoint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */

HIP_TEST_CASE(Unit_hipGetDriverEntryPoint_Negative) {
  void* funcPtr = nullptr;
  hipDriverEntryPointQueryResult status;

  SECTION("Empty string as symbol") {
    HIP_CHECK_ERROR(hipGetDriverEntryPoint("", &funcPtr, hipEnableDefault, &status),
                    hipErrorInvalidValue);
  }

  SECTION("funtion pointer is nullptr") {
    HIP_CHECK_ERROR(hipGetDriverEntryPoint("hipGetDeviceCount", nullptr, hipEnableDefault, &status),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipGetDriverEntryPoint("hipGetDeviceCount", &funcPtr, -1, &status),
                    hipErrorInvalidValue);
  }
}

/**
 * hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** funcPtr, unsigned long long
 flags, hipDriverEntryPointQueryResult* driverStatus);
 * Gets function pointer of a request HIP API
 *
 * @param [in]  symbol  The API base name
 * @param [out] funcPtr Pinter to the requested function
 * @param [in]  flags Flags for the search
 * @param [out] driverStatus Optional returned status of the search
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 */

/**
 * Test Description
 * ------------------------
 *  - This will perfrom the funtionality testing og hipGetDriverEntryPoint_spt api
 * Test source
 * ------------------------
 *  - unit/device/hipGetDriverEntryPoint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */

HIP_TEST_CASE(Unit_hipGetDriverEntryPoint_spt_Positive) {
  void* funcPtr = nullptr;
  hipDriverEntryPointQueryResult status;

  SECTION("hipEnableDefault search flag") {
    HIP_CHECK(hipGetDriverEntryPoint_spt("hipGetDeviceCount", &funcPtr, hipEnableDefault, &status));
  }

  SECTION("hipEnableLegacyStream search flag") {
    HIP_CHECK(
        hipGetDriverEntryPoint_spt("hipGetDeviceCount", &funcPtr, hipEnableLegacyStream, &status));
  }

  SECTION("hipEnablePerThreadDefaultStream search flag") {
    HIP_CHECK(hipGetDriverEntryPoint_spt("hipGetDeviceCount", &funcPtr,
                                         hipEnablePerThreadDefaultStream, &status));
  }

  REQUIRE(status == hipDriverEntryPointSuccess);

  hipError_t (*hipGetDeviceCount_ptr)(int*) = (hipError_t(*)(int*))funcPtr;
  int countFuncPtr;
  HIP_CHECK(hipGetDeviceCount_ptr(&countFuncPtr));

  int count;
  HIP_CHECK(hipGetDeviceCount(&count));

  REQUIRE(count > 0);
  REQUIRE(countFuncPtr == count);
}

/**
 * Test Description
 * ------------------------
 *  - This tests checks hipGetDriverEntryPoint_spt api with negative parameters
 *  # symbol is empty
 *  # funcPtr pointer is null
 *  # Invalid flag
 * Test source
 * ------------------------
 *  - unit/device/hipGetDriverEntryPoint.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */

HIP_TEST_CASE(Unit_hipGetDriverEntryPoint_spt_Negative) {
  void* funcPtr = nullptr;
  hipDriverEntryPointQueryResult status;

  SECTION("Empty string as symbol") {
    HIP_CHECK_ERROR(hipGetDriverEntryPoint_spt("", &funcPtr, hipEnableDefault, &status),
                    hipErrorInvalidValue);
  }

  SECTION("funtion pointer is nullptr") {
    HIP_CHECK_ERROR(
        hipGetDriverEntryPoint_spt("hipGetDeviceCount", nullptr, hipEnableDefault, &status),
        hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipGetDriverEntryPoint_spt("hipGetDeviceCount", &funcPtr, -1, &status),
                    hipErrorInvalidValue);
  }
}
/**
 * End doxygen group DeviceTest.
 * @}
 */
