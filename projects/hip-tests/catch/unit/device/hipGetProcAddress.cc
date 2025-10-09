/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

/**
 * hipError_t hipGetProcAddress	(const char * 	symbol, void ** pfn, int
 * hipVersion, uint64_t flags, hipDriverProcAddressQueryResult * symbolStatus)
 * Gets the pointer of requested HIP driver function.
 * Parameters
 * [in]	 symbol	The Symbol name of the driver function to request.
 * [out] pfn	Output pointer to the requested driver function.
 * [in]	 hipVersion	The HIP version for the requested driver function
 * symbol. [in]	 flags	Currently only default flag is suppported. [out]
 * symbolStatus	Optional enumeration for returned status of searching for symbol
 * driver function based on the input hipVersion. Returns hipSuccess if the
 * returned pfn is addressed to the pointer of found driver function.
 */

/**
 * Test Description
 * ------------------------
 *  - This will perfrom the funtionality testing of hipGetProcAddress api
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGetProcAddress_Positive") {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("hipEnableDefault search flag") {
    HIP_CHECK(hipGetProcAddress("hipGetDeviceCount", &funcPtr,
                                currentHipVersion, HIP_GET_PROC_ADDRESS_DEFAULT,
                                &status));
  }

  SECTION("hipEnableLegacyStream search flag") {
    HIP_CHECK(hipGetProcAddress("hipGetDeviceCount", &funcPtr,
                                currentHipVersion,
                                HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
  }

  SECTION("hipEnablePerThreadDefaultStream search flag") {
    HIP_CHECK(hipGetProcAddress(
        "hipGetDeviceCount", &funcPtr, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  }

  REQUIRE(status == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*hipGetDeviceCount_ptr)(int *) = (hipError_t(*)(int *))funcPtr;
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
 *  - This tests checks hipGetProcAddress api with negative parameters
 *  # symbol is empty
 *  # funcPtr pointer is null
 *  # Invalid flag
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */

TEST_CASE("Unit_hipGetProcAddress_Negative") {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Empty string as symbol") {
    HIP_CHECK_ERROR(hipGetProcAddress("", &funcPtr, currentHipVersion,
                                      HIP_GET_PROC_ADDRESS_DEFAULT, &status),
                    hipErrorInvalidValue);
  }

  SECTION("funtion pointer is nullptr") {
    HIP_CHECK_ERROR(hipGetProcAddress("hipGetDeviceCount", nullptr,
                                      currentHipVersion,
                                      HIP_GET_PROC_ADDRESS_DEFAULT, &status),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipGetProcAddress("hipGetDeviceCount", &funcPtr,
                                      currentHipVersion, -1, &status),
                    hipErrorInvalidValue);
  }
}

/**
 * hipError_t hipGetProcAddress_spt (const char * symbol, void ** pfn, int
 * hipVersion, uint64_t flags, hipDriverProcAddressQueryResult * symbolStatus)
 * Gets the pointer of requested HIP driver function.
 * Parameters
 * [in]	 symbol	The Symbol name of the driver function to request.
 * [out] pfn	Output pointer to the requested driver function.
 * [in]	 hipVersion	The HIP version for the requested driver function
 * symbol. [in]	 flags	Currently only default flag is suppported. [out]
 * symbolStatus	Optional enumeration for returned status of searching for symbol
 * driver function based on the input hipVersion. Returns hipSuccess if the
 * returned pfn is addressed to the pointer of found driver function.
 */

/**
 * Test Description
 * ------------------------
 *  - This will perfrom the funtionality testing of hipGetProcAddress_spt api
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */

TEST_CASE("Unit_hipGetProcAddress_spt_Positive") {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("hipEnableDefault search flag") {
    HIP_CHECK(hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr,
                                    currentHipVersion,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, &status));
  }

  SECTION("hipEnableLegacyStream search flag") {
    HIP_CHECK(
        hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
  }

  SECTION("hipEnablePerThreadDefaultStream search flag") {
    HIP_CHECK(hipGetProcAddress_spt(
        "hipGetDeviceCount", &funcPtr, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  }

  REQUIRE(status == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*hipGetDeviceCount_ptr)(int *) = (hipError_t(*)(int *))funcPtr;
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
 *  - This tests checks hipGetProcAddress_spt api with negative parameters
 *  # symbol is empty
 *  # funcPtr pointer is null
 *  # Invalid flag
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */

TEST_CASE("Unit_hipGetProcAddress_spt_Negative") {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Empty string as symbol") {
    HIP_CHECK_ERROR(hipGetProcAddress_spt("", &funcPtr, currentHipVersion,
                                          HIP_GET_PROC_ADDRESS_DEFAULT,
                                          &status),
                    hipErrorInvalidValue);
  }

  SECTION("funtion pointer is nullptr") {
    HIP_CHECK_ERROR(
        hipGetProcAddress_spt("hipGetDeviceCount", nullptr, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_DEFAULT, &status),
        hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr,
                                          currentHipVersion, -1, &status),
                    hipErrorInvalidValue);
  }
}
/**
 * End doxygen group DeviceTest.
 * @}
 */
