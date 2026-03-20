/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* Test Case Description:
   1) This testcase verifies the  basic scenario - supported on
     all devices
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <chrono>
#include <vector>

/*
    This testcase verifies HIP Mem VMM API basic scenario - supported on all devices
 */

HIP_TEST_CASE(Unit_hipMemVmm_Basic) {
  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << vmm);

  if (vmm == 0) {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t granularity = 0;

  hipMemAllocationProp memAllocationProp{};
  memAllocationProp.type = hipMemAllocationTypePinned;
  memAllocationProp.location.id = 0;
  memAllocationProp.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &memAllocationProp,
                                           hipMemAllocationGranularityRecommended));

  size_t size = granularity * 4;
  void* reservedAddress{nullptr};
  HIP_CHECK(hipMemAddressReserve(&reservedAddress, size, 0, nullptr, 0));

  hipMemGenericAllocationHandle_t gaHandle{nullptr};
  HIP_CHECK(hipMemCreate(&gaHandle, size, &memAllocationProp, 0));

  HIP_CHECK(hipMemMap(reservedAddress, size, 0, gaHandle, 0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  hipMemAccessDesc desc;
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  std::vector<char> values(size);
  const char value = 1;

  HIP_CHECK(hipMemSetAccess(reservedAddress, size, &desc, 1));
  HIP_CHECK(hipMemset(reservedAddress, value, size));
  HIP_CHECK(hipMemcpy(&values[0], reservedAddress, size, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; ++i) {
    REQUIRE(values[i] == value);
  }

  HIP_CHECK(hipMemUnmap(reservedAddress, size));

  HIP_CHECK(hipMemRelease(gaHandle));
  HIP_CHECK(hipMemAddressFree(reservedAddress, size));
}

/*
    This testcase verifies HIP Mem VMM API basic scenario, but with Uncached memory -- supported
    only on HIP
 */

#if HT_AMD
HIP_TEST_CASE(Unit_hipMemVmm_Uncached) {
  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << vmm);

  if (vmm == 0) {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t granularity = 0;

  hipMemAllocationProp memAllocationProp{};
  memAllocationProp.type = hipMemAllocationTypeUncached;
  memAllocationProp.location.id = 0;
  memAllocationProp.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &memAllocationProp,
                                           hipMemAllocationGranularityRecommended));

  size_t size = granularity;
  void* reservedAddress{nullptr};
  HIP_CHECK(hipMemAddressReserve(&reservedAddress, size, 0, nullptr, 0));

  hipMemGenericAllocationHandle_t gaHandle{nullptr};
  HIP_CHECK(hipMemCreate(&gaHandle, size, &memAllocationProp, 0));

  HIP_CHECK(hipMemMap(reservedAddress, size, 0, gaHandle, 0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  hipMemAccessDesc desc;
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  std::vector<char> values(size);
  const int value = 1;

  HIP_CHECK(hipMemSetAccess(reservedAddress, size, &desc, 1));
  HIP_CHECK(hipMemset(reservedAddress, value, size));
  HIP_CHECK(hipMemcpy(values.data(), reservedAddress, size, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; ++i) {
    REQUIRE(values[i] == value);
  }

  HIP_CHECK(hipMemUnmap(reservedAddress, size));

  HIP_CHECK(hipMemRelease(gaHandle));
  HIP_CHECK(hipMemAddressFree(reservedAddress, size));
}
#endif
