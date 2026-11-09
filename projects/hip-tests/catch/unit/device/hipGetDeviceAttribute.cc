/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#ifdef __linux__
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

/**
 * @addtogroup hipDeviceGetAttribute hipDeviceGetAttribute
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)` -
 * Query for a specific device attribute.
 */

static hipError_t test_hipDeviceGetAttribute(int deviceId, hipDeviceAttribute_t attr,
                                             int expectedValue = -1) {
  int value = 0;
  std::cout << "Test hipDeviceGetAttribute attribute " << attr;
  if (expectedValue != -1) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != -1) && value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t test_hipDeviceGetHdpAddress(int deviceId, hipDeviceAttribute_t attr,
                                              uint32_t* expectedValue) {
  uint32_t* value = 0;
  std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
  if (expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(reinterpret_cast<int*>(&value), attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) && value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

/**
 * Test Description
 * ------------------------
 *  - Validate various device attributes against device properties.
 *  - Matching attribute and property value shall be equal.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetDeviceAttribute_CheckAttrValues) {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerBlock,
                                       props.maxThreadsPerBlock));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimX, props.maxThreadsDim[0]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimY, props.maxThreadsDim[1]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimZ, props.maxThreadsDim[2]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimX, props.maxGridSize[0]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimY, props.maxGridSize[1]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimZ, props.maxGridSize[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                       props.sharedMemPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTotalConstantMemory,
                                       props.totalConstMem));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeWarpSize, props.warpSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxRegistersPerBlock,
                                       props.regsPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeClockRate, props.clockRate));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryClockRate,
                                       props.memoryClockRate));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryBusWidth, props.memoryBusWidth));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMultiprocessorCount,
                                       props.multiProcessorCount));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIsMultiGpuBoard,
                                       props.isMultiGpuBoard));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeMode, props.computeMode));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeL2CacheSize, props.l2CacheSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerMultiProcessor,
                                       props.maxThreadsPerMultiProcessor));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMajor, props.major));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMinor, props.minor));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeConcurrentKernels,
                                       props.concurrentKernels));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciBusId, props.pciBusID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciDeviceId, props.pciDeviceID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIntegrated, props.integrated));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture1DWidth,
                                       props.maxTexture1D));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DWidth,
                                       props.maxTexture2D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DHeight,
                                       props.maxTexture2D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DWidth,
                                       props.maxTexture3D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DHeight,
                                       props.maxTexture3D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DDepth,
                                       props.maxTexture3D[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeLaunch,
                                       props.cooperativeLaunch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeMultiDeviceLaunch,
                                       props.cooperativeMultiDeviceLaunch));

#if HT_AMD
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                                       props.maxSharedMemoryPerMultiProcessor));
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpMemFlushCntl,
                                        props.hdpMemFlushCntl));
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpRegFlushCntl,
                                        props.hdpRegFlushCntl));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeDirectManagedMemAccessFromHost,
                                       props.directManagedMemAccessFromHost));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeConcurrentManagedAccess,
                                       props.concurrentManagedAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePageableMemoryAccess,
                                       props.pageableMemoryAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
                                       props.pageableMemoryAccessUsesHostPageTables));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
                                       props.cooperativeMultiDeviceUnmatchedFunc));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
                                       props.cooperativeMultiDeviceUnmatchedGridDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
                                       props.cooperativeMultiDeviceUnmatchedBlockDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
                                       props.cooperativeMultiDeviceUnmatchedSharedMem));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeAsicRevision, props.asicRevision));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeManagedMemory, props.managedMemory));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeNumberOfXccs));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeExpertSchedMode));
#endif

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxPitch, props.memPitch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTextureAlignment,
                                       props.textureAlignment));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeKernelExecTimeout,
                                       props.kernelExecTimeoutEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCanMapHostMemory,
                                       props.canMapHostMemory));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeEccEnabled, props.ECCEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTexturePitchAlignment,
                                       props.texturePitchAlignment));
}

/**
 * Test Description
 * ------------------------
 *  - Validates negative scenarios:
 *    -# When pointer to value is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ID is `-1`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ID is out of bounds
 *      - Expected output: do not return `hipSuccess`
 *    -# When attribute is invalid (-1)
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceGetAttribute_NegTst) {
  int deviceCount = 0;
  int pi = -1;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);

  int device;
  HIP_CHECK(hipGetDevice(&device));

  // pi is nullptr
  SECTION("pi is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(nullptr, hipDeviceAttributePciBusId, device));
  }

  // device is -1
  SECTION("device is -1") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, -1));
  }

  // device is Non Existing Device
  SECTION("device is Non Existing Device") {
    REQUIRE_FALSE(hipSuccess ==
                  hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, deviceCount));
  }

  // attr is Invalid Attribute
  SECTION("attr is invalid") {
    REQUIRE_FALSE(hipSuccess ==
                  hipDeviceGetAttribute(&pi, static_cast<hipDeviceAttribute_t>(-1), device));
  }
}

/**
 * Test Description
 * ------------------------
 *  - verify hipDeviceAttributeHostRegisterSupported attribute.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGetDeviceAttribute_hipDevAttrHostRegisterSupported) {
  hipError_t ret_val;
  int hipDevAttr = 0;
  ret_val = hipDeviceGetAttribute(&hipDevAttr, hipDeviceAttributeHostRegisterSupported, 0);
  INFO("hipDeviceAttributeHostRegisterSupported: " << hipDevAttr);

  if (ret_val == hipSuccess) {
    auto x = std::unique_ptr<int>(new int);
    HIP_CHECK(hipHostRegister(x.get(), sizeof(int), hipHostRegisterDefault));

    void* device_memory;
    HIP_CHECK(hipHostGetDevicePointer(&device_memory, x.get(), 0));

    HIP_CHECK(hipHostUnregister(x.get()));
    HIP_CHECK_ERROR(hipHostGetDevicePointer(&device_memory, x.get(), 0), hipErrorInvalidValue);
  } else {
    HIP_SKIP_TEST(
        "Skipping the test as GPU 0 doesn't support "
        "hipDeviceAttributeHostRegisterSupported attribute.\n");
  }
}

HIP_TEST_CASE(Unit_hipGetDeviceAttribute_hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported) {
  int hipVmmSupported = 0, hipDmaBufSupported = 0, hipRDMAWithHipVMMSupported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&hipRDMAWithHipVMMSupported, hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported, 0));
  HIP_CHECK(hipDeviceGetAttribute(&hipVmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  HIP_CHECK(hipDeviceGetAttribute(&hipDmaBufSupported, hipDeviceAttributeDmaBufSupported, 0));
  INFO("hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported: " << hipRDMAWithHipVMMSupported);
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << hipVmmSupported);
  INFO("hipDeviceAttributeDmaBufSupported: " << hipDmaBufSupported);
  REQUIRE(hipRDMAWithHipVMMSupported == (hipVmmSupported && hipDmaBufSupported));
}

HIP_TEST_CASE(Unit_hipGetDeviceAttribute_hipDeviceAttributeHostAllocDmaBufSupported) {
  int hostAllocDmaBufSupported = -1;
  HIP_CHECK(hipDeviceGetAttribute(&hostAllocDmaBufSupported,
                                  hipDeviceAttributeHostAllocDmaBufSupported, 0));
  INFO("hipDeviceAttributeHostAllocDmaBufSupported: " << hostAllocDmaBufSupported);
  REQUIRE((hostAllocDmaBufSupported == 0 || hostAllocDmaBufSupported == 1));
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Query L3 cache size on every visible device. Zero is valid when no cache
 *    information is available. Check null output and invalid device errors.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - AMD platform
 */
HIP_TEST_CASE(Unit_hipGetDeviceAttribute_hipDeviceAttributeL3CacheSize) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);
  for (int device = 0; device < deviceCount; ++device) {
    INFO("device: " << device);
    int l3 = -1;
    HIP_CHECK(hipDeviceGetAttribute(&l3, hipDeviceAttributeL3CacheSize, device));
    INFO("L3CacheSize: " << l3);
    REQUIRE(l3 >= 0);
    HIP_CHECK_ERROR(hipDeviceGetAttribute(nullptr, hipDeviceAttributeL3CacheSize, device),
                    hipErrorInvalidValue);
    (void)hipGetLastError();
  }

  int l3 = -1;
  for (int device : {-1, deviceCount, std::numeric_limits<int>::max()}) {
    INFO("invalid device: " << device);
    HIP_CHECK_ERROR(hipDeviceGetAttribute(&l3, hipDeviceAttributeL3CacheSize, device),
                    hipErrorInvalidDevice);
  }
  // Clear the sticky errors from the intentional negative calls above so they
  // cannot leak into a later case in the same process.
  (void)hipGetLastError();
}

#if HT_LINUX
// The Linux AMD test suite already links HSA. Match agents by UUID, since HIP
// visibility and ordering need not match HSA enumeration order.
HIP_TEST_CASE(Unit_hipGetDeviceAttribute_L3CacheSize_HsaConsistency) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);
  std::vector<hsa_agent_t> agents;
  const hsa_status_t iterateStatus = hsa_iterate_agents(
      [](hsa_agent_t agent, void* data) {
        static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
        return HSA_STATUS_SUCCESS;
      },
      &agents);
  if (iterateStatus != HSA_STATUS_SUCCESS) {
    HIP_SKIP_TEST("HSA agent enumeration is unavailable on this platform");
  }

  int checked = 0;
  for (int device = 0; device < deviceCount; ++device) {
    INFO("device: " << device);
    hipUUID uuid{};
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    hsa_agent_t match{};
    int matches = 0;
    for (auto agent : agents) {
      hsa_device_type_t type;
      REQUIRE(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS);
      if (type != HSA_DEVICE_TYPE_GPU) continue;
      char hsaUuid[32] = {};
      if (hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                             hsaUuid) == HSA_STATUS_SUCCESS &&
          std::memcmp(hsaUuid, "GPU-", 4) == 0 && hsaUuid[20] == '\0' &&
          std::all_of(hsaUuid + 4, hsaUuid + 20, [](char digit) {
            return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f') ||
                   (digit >= 'A' && digit <= 'F');
          }) &&
          std::memcmp(uuid.bytes, hsaUuid + 4, sizeof(uuid.bytes)) == 0) {
        match = agent;
        ++matches;
      }
    }
    // Only compare cache sizes when the agent can be identified unambiguously.
    if (matches != 1) {
      WARN("No unique HSA UUID match for device " << device);
      continue;
    }
    uint32_t cacheSizes[4] = {};
    if (hsa_agent_get_info(match, HSA_AGENT_INFO_CACHE_SIZE, cacheSizes) != HSA_STATUS_SUCCESS) {
      WARN("HSA cache sizes unavailable for device " << device);
      continue;
    }
    int l3 = -1;
    HIP_CHECK(hipDeviceGetAttribute(&l3, hipDeviceAttributeL3CacheSize, device));
    const auto expected = std::min<uint32_t>(cacheSizes[2], std::numeric_limits<int>::max());
    INFO("HSA L3 cache size: " << cacheSizes[2]);
    REQUIRE(l3 == static_cast<int>(expected));
    ++checked;
  }
  if (checked == 0) {
    HIP_SKIP_TEST("No HSA cache information could be matched to a HIP device");
  }
}
#endif  // HT_LINUX
#endif  // HT_AMD

/**
 * End doxygen group DeviceTest.
 * @}
 */
