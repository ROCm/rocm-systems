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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>
#include "hip_vmm_common.hh"
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - Virtual Memory Management APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipGetProcAddress_VMMAPIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_VMM") {
  int value = 0;
  HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (value == 0) {
    HipTest::HIP_SKIP_TEST("Machine does not support VMM. Skipping Test..");
    return;
  }

  hipDeviceProp_t devProp;
  int device;
  bool xnackEnabled = false;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device));
  std::string gfxName(devProp.gcnArchName);

  if (gfxName.find("xnack+") != std::string::npos) {
    xnackEnabled = true;
  }

  void* hipMemGetAllocationGranularity_ptr = nullptr;
  void* hipMemCreate_ptr = nullptr;
  void* hipMemAddressReserve_ptr = nullptr;
  void* hipMemMap_ptr = nullptr;
  void* hipMemSetAccess_ptr = nullptr;
  void* hipMemUnmap_ptr = nullptr;
  void* hipMemAddressFree_ptr = nullptr;
  void* hipMemRelease_ptr = nullptr;
  void* hipMemGetAccess_ptr = nullptr;
  void* hipMemGetAllocationPropertiesFromHandle_ptr = nullptr;
  void* hipMemRetainAllocationHandle_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemGetAllocationGranularity", &hipMemGetAllocationGranularity_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemCreate", &hipMemCreate_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemAddressReserve", &hipMemAddressReserve_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemMap", &hipMemMap_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemSetAccess", &hipMemSetAccess_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemUnmap", &hipMemUnmap_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemAddressFree", &hipMemAddressFree_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemRelease", &hipMemRelease_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemGetAccess", &hipMemGetAccess_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemGetAllocationPropertiesFromHandle",
                              &hipMemGetAllocationPropertiesFromHandle_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemRetainAllocationHandle", &hipMemRetainAllocationHandle_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemGetAllocationGranularity_ptr)(size_t*, const hipMemAllocationProp*,
                                                       hipMemAllocationGranularity_flags) =
      reinterpret_cast<hipError_t (*)(size_t*, const hipMemAllocationProp*,
                                      hipMemAllocationGranularity_flags)>(
          hipMemGetAllocationGranularity_ptr);

  hipError_t (*dyn_hipMemCreate_ptr)(hipMemGenericAllocationHandle_t*, size_t,
                                     const hipMemAllocationProp*, uint64_t) =
      reinterpret_cast<hipError_t (*)(hipMemGenericAllocationHandle_t*, size_t,
                                      const hipMemAllocationProp*, uint64_t)>(hipMemCreate_ptr);

  hipError_t (*dyn_hipMemAddressReserve_ptr)(void**, size_t, size_t, void*, uint64_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t, size_t, void*, uint64_t)>(
          hipMemAddressReserve_ptr);

  hipError_t (*dyn_hipMemMap_ptr)(void*, size_t, size_t, hipMemGenericAllocationHandle_t,
                                  uint64_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, size_t, hipMemGenericAllocationHandle_t,
                                      uint64_t)>(hipMemMap_ptr);

  hipError_t (*dyn_hipMemSetAccess_ptr)(void*, size_t, const hipMemAccessDesc*, size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, const hipMemAccessDesc*, size_t)>(
          hipMemSetAccess_ptr);

  hipError_t (*dyn_hipMemUnmap_ptr)(void*, size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t)>(hipMemUnmap_ptr);

  hipError_t (*dyn_hipMemAddressFree_ptr)(void*, size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t)>(hipMemAddressFree_ptr);

  hipError_t (*dyn_hipMemRelease_ptr)(hipMemGenericAllocationHandle_t) =
      reinterpret_cast<hipError_t (*)(hipMemGenericAllocationHandle_t)>(hipMemRelease_ptr);

  hipError_t (*dyn_hipMemGetAccess_ptr)(uint64_t*, const hipMemLocation*, void*) =
      reinterpret_cast<hipError_t (*)(uint64_t*, const hipMemLocation*, void*)>(
          hipMemGetAccess_ptr);

  hipError_t (*dyn_hipMemGetAllocationPropertiesFromHandle_ptr)(hipMemAllocationProp*,
                                                                hipMemGenericAllocationHandle_t) =
      reinterpret_cast<hipError_t (*)(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>(
          hipMemGetAllocationPropertiesFromHandle_ptr);

  hipError_t (*dyn_hipMemRetainAllocationHandle_ptr)(hipMemGenericAllocationHandle_t*, void*) =
      reinterpret_cast<hipError_t (*)(hipMemGenericAllocationHandle_t*, void*)>(
          hipMemRetainAllocationHandle_ptr);

  const int N = 10;
  const int Nbytes = 10 * sizeof(int);
  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;

  // Validating hipMemGetAllocationGranularity API
  size_t granularity = 0;
  size_t granularityWithFuncPtr = 0;

  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  HIP_CHECK(dyn_hipMemGetAllocationGranularity_ptr(&granularityWithFuncPtr, &prop,
                                                   hipMemAllocationGranularityMinimum));

  REQUIRE(granularity > 0);
  REQUIRE(granularityWithFuncPtr > 0);
  REQUIRE(granularityWithFuncPtr == granularity);

  // Validating hipMemCreate API
  hipMemGenericAllocationHandle_t handle = nullptr;
  HIP_CHECK(dyn_hipMemCreate_ptr(&handle, granularity, &prop, 0));
  REQUIRE(handle != nullptr);

  // Validating hipMemAddressReserve API
  void* ptr = nullptr;
  HIP_CHECK(dyn_hipMemAddressReserve_ptr(&ptr, granularity, 0, 0, 0));
  REQUIRE(ptr != nullptr);

  // Validating hipMemMap API
  HIP_CHECK(dyn_hipMemMap_ptr(ptr, granularity, 0, handle, 0));
  REQUIRE(ptr != nullptr);

  // Validating hipMemSetAccess API
  hipMemAccessDesc desc;
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = 0;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(dyn_hipMemSetAccess_ptr(ptr, granularity, &desc, 1));
  REQUIRE(ptr != nullptr);

  // Performing some operations on ptr, to validate it
  HIP_CHECK(hipMemcpy(ptr, hostMem, Nbytes, hipMemcpyHostToDevice));
  addOneKernel<<<1, 1>>>(reinterpret_cast<int*>(ptr), N);
  HIP_CHECK(hipMemcpy(hostMem, ptr, Nbytes, hipMemcpyDeviceToHost));
  validateHostArray(hostMem, N, 11);

  // Validating hipMemGetAccess API
  uint64_t flags;
  hipMemLocation location = {hipMemLocationTypeDevice, 0};
  HIP_CHECK(dyn_hipMemGetAccess_ptr(&flags, &location, ptr));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);

  // Validating hipMemGetAllocationPropertiesFromHandle API
  hipMemAllocationProp requiredProp;
  HIP_CHECK(dyn_hipMemGetAllocationPropertiesFromHandle_ptr(&requiredProp, handle));
  REQUIRE(requiredProp.type == hipMemAllocationTypePinned);
  REQUIRE(requiredProp.requestedHandleTypes == hipMemHandleTypeNone);
  REQUIRE(requiredProp.location.type == hipMemLocationTypeDevice);
  REQUIRE(requiredProp.location.id == 0);

  // Validating hipMemRetainAllocationHandle API
  hipMemGenericAllocationHandle_t requiredHandle = nullptr;
  HIP_CHECK(dyn_hipMemRetainAllocationHandle_ptr(&requiredHandle, ptr));
  REQUIRE(requiredHandle == handle);

  // Validating hipMemUnmap, hipMemAddressFree, hipMemRelease API's
  HIP_CHECK(dyn_hipMemUnmap_ptr(ptr, granularity));
  HIP_CHECK(dyn_hipMemAddressFree_ptr(ptr, granularity));
  HIP_CHECK(dyn_hipMemRelease_ptr(handle));

  // Performing operation on ptr, to check it is invalidated or not
  if (!xnackEnabled) {
    REQUIRE(hipMemcpy(ptr, hostMem, Nbytes, hipMemcpyHostToDevice) == hipErrorInvalidValue);
  }
  free(hostMem);
}
