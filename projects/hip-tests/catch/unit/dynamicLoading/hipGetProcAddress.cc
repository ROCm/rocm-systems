/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <dlfcn.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipGetProcAddress hipGetProcAddress_spt
 * @{
 * @ingroup DynamicLoading
 * `hipError_t hipGetProcAddress (const char * symbol, void ** pfn, int
                                  hipVersion, uint64_t flags,
                                  hipDriverProcAddressQueryResult *
 symbolStatus)` -
 * Gets the pointer of requested HIP driver function.
 * `hipError_t hipGetProcAddress_spt (const char * 	symbol, void ** pfn, int
                                  hipVersion, uint64_t flags,
                                  hipDriverProcAddressQueryResult *
 symbolStatus)` -
 * Gets the pointer of requested HIP driver function.
*/

/**
 * Test Description
 * ------------------------
 *  - This will perform the basic operation on hipGetProcAddress API.
 *  - Get the driver symbol with all three flags and validates the basic
 *  - functionlity of hipGetDeviceCount using the function pointer.
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
#if HT_AMD
HIP_TEST_CASE(Unit_hipGetProcAddress_Positive) {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Get driver symbol for default flag") {
    HIP_CHECK(hipGetProcAddress("hipGetDeviceCount", &funcPtr,
                                currentHipVersion, HIP_GET_PROC_ADDRESS_DEFAULT,
                                &status));
  }

  SECTION("Get driver symbol for legacy stream flag") {
    HIP_CHECK(hipGetProcAddress("hipGetDeviceCount", &funcPtr,
                                currentHipVersion,
                                HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
  }

  SECTION("Get driver symbol for Per thread stream flag") {
    HIP_CHECK(hipGetProcAddress(
        "hipGetDeviceCount", &funcPtr, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  }

  REQUIRE(status == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*hipGetDeviceCount_ptr)(int *) = (hipError_t(*)(int *))funcPtr;
  int countFuncPtr = 0;
  HIP_CHECK(hipGetDeviceCount_ptr(&countFuncPtr));

  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));

  REQUIRE(count > 0);
  REQUIRE(countFuncPtr > 0);
  REQUIRE(countFuncPtr == count);
}

/**
 * Test Description
 * ------------------------
 *  - This will verify behavior of hipGetProcAddress API with negative
 * parameters
 *  - 1) Empty symbol
 *  - 2) Invalid flag
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_Negative) {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Empty symbol") {
    HIP_CHECK_ERROR(hipGetProcAddress("", &funcPtr, currentHipVersion,
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
 * Test Description
 * ------------------------
 *  - This will perform the basic operation on hipGetProcAddress_spt API.
 *  - Get the driver symbol with all three flags and validates the basic
 *  - functionlity of hipGetDeviceCount using the function pointer.
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_spt_Positive) {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Get driver symbol for default flag") {
    HIP_CHECK(hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr,
                                    currentHipVersion,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, &status));
  }

  SECTION("Get driver symbol for legacy stream flag") {
    HIP_CHECK(
        hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
  }

  SECTION("Get driver symbol for per thread stream flag") {
    HIP_CHECK(hipGetProcAddress_spt(
        "hipGetDeviceCount", &funcPtr, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  }

  REQUIRE(status == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*hipGetDeviceCount_ptr)(int *) = (hipError_t(*)(int *))funcPtr;
  int countFuncPtr = 0;
  HIP_CHECK(hipGetDeviceCount_ptr(&countFuncPtr));

  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));

  REQUIRE(count > 0);
  REQUIRE(countFuncPtr > 0);
  REQUIRE(countFuncPtr == count);
}

/**
 * Test Description
 * ------------------------
 *  - This will verify behavior of hipGetProcAddress_spt API with negative
 * parameters
 *  - 1) Empty symbol
 *  - 2) Invalid flag
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_spt_Negative) {
  void *funcPtr = nullptr;
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  SECTION("Empty symbol") {
    HIP_CHECK_ERROR(hipGetProcAddress_spt("", &funcPtr, currentHipVersion,
                                          HIP_GET_PROC_ADDRESS_DEFAULT,
                                          &status),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipGetProcAddress_spt("hipGetDeviceCount", &funcPtr,
                                          currentHipVersion, -1, &status),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 * - This will verify behavior of hipGetProcAddress and hipGetProcAddress_spt
 * - and check with original API address
 * - 1) With hipMemcpy (it has spt version hipMemcpy_spt)
 * - 2) With hipMalloc (it doesn't has spt version)
 * - 3) with spt API hipMemcpy_spt
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_hipGetProcAddress_spt_CheckAddress) {
  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  void *funcPtr_default = nullptr;
  void *funcPtr_legacy = nullptr;
  void *funcPtr_spt = nullptr;

  void *handle = nullptr;
  handle = dlopen("libamdhip64.so", RTLD_LAZY);

  if (handle == nullptr) {
    // Try to find in the user defined rocm path
    char *rocmPath = nullptr;
    rocmPath = std::getenv("ROCM_PATH");

    if (rocmPath) {
      std::string libPathFile(rocmPath);
      libPathFile += "/lib/libamdhip64.so";
      handle = dlopen(libPathFile.c_str(), RTLD_LAZY);
    }
    // Try to find in the /opt/rocm/lib path
    else {
      handle = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LAZY);
    }
  }
  REQUIRE(handle != nullptr);

  void *sym_hipMemcpy = dlsym(handle, "hipMemcpy");
  void *sym_hipMemcpy_spt = dlsym(handle, "hipMemcpy_spt");
  void *sym_hipMalloc = dlsym(handle, "hipMalloc");

  SECTION("hipGetProcAddress - For hipMemcpy") {
    HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_default,
                                currentHipVersion, HIP_GET_PROC_ADDRESS_DEFAULT,
                                &status));

    HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_legacy, currentHipVersion,
                                HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_spt, currentHipVersion,
                                HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                                &status));

    REQUIRE(funcPtr_default == sym_hipMemcpy);
    REQUIRE(funcPtr_legacy == sym_hipMemcpy);
    REQUIRE(funcPtr_spt == sym_hipMemcpy_spt);
  }

  SECTION("hipGetProcAddress - For hipMalloc") {
    HIP_CHECK(hipGetProcAddress("hipMalloc", &funcPtr_default,
                                currentHipVersion, HIP_GET_PROC_ADDRESS_DEFAULT,
                                &status));

    HIP_CHECK(hipGetProcAddress("hipMalloc", &funcPtr_legacy, currentHipVersion,
                                HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress("hipMalloc", &funcPtr_spt, currentHipVersion,
                                HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                                &status));

    REQUIRE(funcPtr_default == sym_hipMalloc);
    REQUIRE(funcPtr_legacy == sym_hipMalloc);
    REQUIRE(funcPtr_spt == sym_hipMalloc);
  }

  SECTION("hipGetProcAddress - For hipMemcpy_spt") {
    HIP_CHECK(hipGetProcAddress("hipMemcpy_spt", &funcPtr_default,
                                currentHipVersion, HIP_GET_PROC_ADDRESS_DEFAULT,
                                &status));

    HIP_CHECK(hipGetProcAddress("hipMemcpy_spt", &funcPtr_legacy,
                                currentHipVersion,
                                HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress(
        "hipMemcpy_spt", &funcPtr_spt, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));

    REQUIRE(funcPtr_default == sym_hipMemcpy_spt);
    REQUIRE(funcPtr_legacy == sym_hipMemcpy_spt);
    REQUIRE(funcPtr_spt == sym_hipMemcpy_spt);
  }

  SECTION("hipGetProcAddress_spt - For hipMemcpy") {
    HIP_CHECK(hipGetProcAddress_spt("hipMemcpy", &funcPtr_default,
                                    currentHipVersion,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, &status));

    HIP_CHECK(
        hipGetProcAddress_spt("hipMemcpy", &funcPtr_legacy, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress_spt(
        "hipMemcpy", &funcPtr_spt, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));

    REQUIRE(funcPtr_default == sym_hipMemcpy_spt);
    REQUIRE(funcPtr_legacy == sym_hipMemcpy);
    REQUIRE(funcPtr_spt == sym_hipMemcpy_spt);
  }

  SECTION("hipGetProcAddress_spt - For hipMalloc") {
    HIP_CHECK(hipGetProcAddress_spt("hipMalloc", &funcPtr_default,
                                    currentHipVersion,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, &status));

    HIP_CHECK(
        hipGetProcAddress_spt("hipMalloc", &funcPtr_legacy, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress_spt(
        "hipMalloc", &funcPtr_spt, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));

    REQUIRE(funcPtr_default == sym_hipMalloc);
    REQUIRE(funcPtr_legacy == sym_hipMalloc);
    REQUIRE(funcPtr_spt == sym_hipMalloc);
  }

  SECTION("hipGetProcAddress_spt - For hipMemcpy_spt") {
    HIP_CHECK(hipGetProcAddress_spt("hipMemcpy_spt", &funcPtr_default,
                                    currentHipVersion,
                                    HIP_GET_PROC_ADDRESS_DEFAULT, &status));

    HIP_CHECK(hipGetProcAddress_spt(
        "hipMemcpy_spt", &funcPtr_legacy, currentHipVersion,
        HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

    HIP_CHECK(hipGetProcAddress_spt(
        "hipMemcpy_spt", &funcPtr_spt, currentHipVersion,
        HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));

    REQUIRE(funcPtr_default == sym_hipMemcpy_spt);
    REQUIRE(funcPtr_legacy == sym_hipMemcpy_spt);
    REQUIRE(funcPtr_spt == sym_hipMemcpy_spt);
  }

  REQUIRE(dlclose(handle) == 0);
}
#endif

/**
 * End doxygen group DeviceTest.
 * @}
 */
