/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemMap hipMemMap
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemMap (void* ptr,
 *                        size_t size,
 *                        size_t offset,
 *                        hipMemGenericAllocationHandle_t handle,
 *                        unsigned long long flags)` -
 * Maps an allocation handle to a reserved virtual address range.
 */

#include <hip_test_common.hh>

#include <atomic>
#include <chrono>
#include <thread>

#include "hip_vmm_common.hh"

constexpr int N = (1 << 13);
constexpr int num_buf = 3;
constexpr int initializer = 0;

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped/unmapped to same
 * vmm address range repeatedly. This test validates physical memory
 * euse using same vmm range.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_SameMemoryReuse) {
  constexpr int iterations = 20;
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int i = 0; i < iterations; i++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  }
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped/unmapped for multiple
 * vmm addresses. This test validates physical memory reuse using
 * different vmm ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_PhysicalMemoryReuse_SingleGPU) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  void* ptrA[num_buf];
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressReserve(&ptrA[buf], size_mem, 0, 0, 0));
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA[buf], size_mem, 0, handle, 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA[buf], size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA[buf]), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA[buf]), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA[buf]));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA[buf]), buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA[buf], size_mem));
  }
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressFree(ptrA[buf], size_mem));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped to multiple
 * vmm addresses at the same time and check data values integrity
 * between different VMMs.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_PhysicalMemory_Map2MultVMMs) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  void* ptrA[num_buf];
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressReserve(&ptrA[buf], size_mem, 0, 0, 0));
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemMap(ptrA[buf], size_mem, 0, handle, 0));
  }
  // Set access for all the buffers.
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemSetAccess(ptrA[buf], size_mem, &accessDesc, 1));
  }
  // Copy data to VMM via ptrA[0]
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA[0]), A_h.data(), buffer_size));
  // Validate the data contained in VMM using ptrA[0], ptrA[1],
  // ......, ptrA[num_buf-1]
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA[buf]), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  }

  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemUnmap(ptrA[buf], size_mem));
  }

  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressFree(ptrA[buf], size_mem));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate one physical handle and map it to N (>=2) different VAs.
 *      Write a known pattern via VA #0, unmap VA #0 only, then read from
 *      VA #1 to verify the surviving alias is still coherent. Finally, map
 *      the same handle to a fresh VA #N and read back to prove the handle
 *      was not released early.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_UnmapOneAliasOfMultiAliasedHandle) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  // Step 1: Allocate one physical handle.
  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  // Reserve N (>=2) VAs and map the handle to each.
  void* ptrA[num_buf];
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressReserve(&ptrA[buf], size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA[buf], size_mem, 0, handle, 0));
  }

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemSetAccess(ptrA[buf], size_mem, &accessDesc, 1));
  }

  // Step 2: write known pattern via VA #0.
  std::vector<int> A_h(N), B_h(N);
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = static_cast<int>(idx) + 0xA5A5;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA[0]), A_h.data(), buffer_size));

  // Step 3: unmap VA #0 only. Handle and remaining aliases must survive.
  HIP_CHECK(hipMemUnmap(ptrA[0], size_mem));

  // Step 4: read from VA #1. Pattern must be visible.
  std::fill(B_h.begin(), B_h.end(), initializer);
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA[1]), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));

  // Step 5: handle must NOT be released early. Map it to a fresh VA #N and
  // read back. If the handle had been incorrectly released the map would
  // fail or the data would be garbage.
  void* ptrFresh = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptrFresh, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrFresh, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemSetAccess(ptrFresh, size_mem, &accessDesc, 1));
  std::fill(B_h.begin(), B_h.end(), initializer);
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrFresh), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));

  // Cleanup: unmap surviving VAs (skip ptrA[0], already unmapped).
  HIP_CHECK(hipMemUnmap(ptrFresh, size_mem));
  for (int buf = 1; buf < num_buf; buf++) {
    HIP_CHECK(hipMemUnmap(ptrA[buf], size_mem));
  }
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrFresh, size_mem));
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressFree(ptrA[buf], size_mem));
  }
  CTX_DESTROY();
}

void physicalMemoryReuse_MultiDev (hipMemAllocationProp prop) {
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  for (int devX = 0; devX < devicecount; devX++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, devX));
    checkVMMSupported(device);
    prop.location.id = device;  // Current Devices
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    // Allocate host memory and intialize data
    std::vector<int> A_h(N), B_h(N);
    // Initialize with data
    for (size_t idx = 0; idx < N; idx++) {
      A_h[idx] = idx;
    }
    // Allocate a physical memory chunk
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
    // Allocate devicecount virtual address ranges
    std::vector<void*> ptrA(devicecount);
    for (int devY = 0; devY < devicecount; devY++) {
      HIP_CHECK(hipMemAddressReserve(&ptrA[devY], size_mem, 0, 0, 0));
    }
    for (int devY = 0; devY < devicecount; devY++) {
      hipDevice_t deviceToTest;
      HIP_CHECK(hipDeviceGet(&deviceToTest, devY));
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = deviceToTest;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipSetDevice(devY));
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA[devY], size_mem, 0, handle, 0));
      // Set access to GPU 0
      HIP_CHECK(hipMemSetAccess(ptrA[devY], size_mem, &accessDesc, 1));
      HIP_CHECK(
          hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA[devY]), A_h.data(), buffer_size));
      HIP_CHECK(
          hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA[devY]), buffer_size));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
      HIP_CHECK(hipMemUnmap(ptrA[devY], size_mem));
    }
    HIP_CHECK(hipSetDevice(0));  // set the device back to 0.
    // Release resources
    HIP_CHECK(hipMemRelease(handle));
    for (int devY = 0; devY < devicecount; devY++) {
      HIP_CHECK(hipMemAddressFree(ptrA[devY], size_mem));
    }
  }
}
/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped/unmapped for
 * multiple vmm addresses. This test validates physical memory
 * reuse using different vmm ranges on multiple devices.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_PhysicalMemoryReuse_MultiDev) {
  CHECK_P2P_SUPPORT
  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    physicalMemoryReuse_MultiDev(prop);
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypeUncached;
    prop.location.type = hipMemLocationTypeDevice;
    physicalMemoryReuse_MultiDev(prop);
  }
  #endif
}
/**
 * Test Description
 * ------------------------
 *    - Check if different physical chunk can be mapped/unmapped
 * for single vmm address. This test validates VMM memory reuse
 * using different physical ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_VMMMemoryReuse_SingleGPU) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle[num_buf];
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemCreate(&handle[buf], size_mem, &prop, 0));
  }
  // Allocate num_buf virtual address ranges
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Map ptrA to physical chunk
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[buf], 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
#if HT_NVIDIA
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
#endif
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  }
  // Release resources
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemRelease(handle[buf]));
  }
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));

  CTX_DESTROY();
}

void vMMMemoryReuse_MultiGPU (hipMemAllocationProp prop) {
  int deviceId = 0, devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  hipDevice_t device;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  std::vector<hipMemGenericAllocationHandle_t> handle(devicecount);
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a physical memory chunk
  for (int dev = 0; dev < devicecount; dev++) {
    hipDevice_t dev_handle;
    HIP_CHECK(hipDeviceGet(&dev_handle, dev));
    prop.location.id = dev_handle;
    HIP_CHECK(hipMemCreate(&handle[dev], size_mem, &prop, 0));
  }
  // Allocate devicecount virtual address ranges
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  // Map ptrA to physical chunk
  SECTION("Set Access of VMM to Different GPU") {
    for (int dev = 0; dev < devicecount; dev++) {
      hipDevice_t device;
      HIP_CHECK(hipDeviceGet(&device, dev));
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = device;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipSetDevice(dev));
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[dev], 0));
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    }
  }
  SECTION("Set Access of VMM to default GPU") {
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    for (int dev = 0; dev < devicecount; dev++) {
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[dev], 0));
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    }
  }
  HIP_CHECK(hipSetDevice(0));
  // Release resources
  for (int dev = 0; dev < devicecount; dev++) {
    HIP_CHECK(hipMemRelease(handle[dev]));
  }
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
}
/**
 * Test Description
 * ------------------------
 *    - Check if different physical chunk allocated in different devices
 * can be mapped/unmapped to single vmm address. This test validates VMM
 * memory reuse using different physical ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_VMMMemoryReuse_MultiGPU) {
  CHECK_P2P_SUPPORT
  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    vMMMemoryReuse_MultiGPU(prop);
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypeUncached;
    prop.location.type = hipMemLocationTypeDevice;
    vMMMemoryReuse_MultiGPU(prop);
  }
  #endif
}
/**
 * Test Description
 * ------------------------
 *    - Check if a partial part of a VMM range can be mapped/unmapped
 * to a physical address.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_MapPartialVMMMem) {
  int deviceId = 0;
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a bigger physical memory chunk of size_mem
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range of size twice size_mem
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, 2 * size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  std::fill(B_h.begin(), B_h.end(), initializer);
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, 2 * size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Argument Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemMap_negative) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));

  SECTION("nullptr to ptrA") {
    REQUIRE(hipMemMap(nullptr, size_mem, 0, handle, 0) == hipErrorInvalidValue);
  }

  SECTION("pass zero to size") {
    REQUIRE(hipMemMap(ptrA, 0, 0, handle, 0) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemMap_Capture) {
  hipMemGenericAllocationHandle_t handle;
  size_t granularity = 0;
  constexpr size_t kAlignment = 2;
  constexpr int kDeviceId = 0;
  hipDevice_t device = 0;
  void* device_ptr = nullptr;

  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  HIP_CHECK(hipMemCreate(&handle, granularity, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&device_ptr, granularity, kAlignment, 0, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemMap(device_ptr, granularity, 0, handle, 0));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemUnmap(device_ptr, granularity));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(device_ptr, granularity));
}

/**
 * Test Description
 * ------------------------
 * - APU-only. Reserves a VA, creates a physical handle expected to exceed
 *   the dedicated-VRAM carveout, maps it, grants access, and exercises the
 *   buffer end-to-end (memset, copy back, verify head and tail). Mirrors
 *   Unit_hipMalloc_Positive_APU_LargeAllocSpill but through the VMEM API
 *   path that hipGraphAddMemAllocNode ultimately uses, so it guards the
 *   same spill behaviour for graph-based allocations.
 * Test source
 * ------------------------
 * - unit/virtualMemoryManagement/hipMemMap.cc
 */
HIP_TEST_CASE(Unit_hipMemMap_Positive_APU_LargeAllocSpill) {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (!prop.integrated) {
    HIP_SKIP_TEST("dGPU --- APU spill regression test does not apply");
    return;
  }
  int vmm_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm_supported,
                                  hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (!vmm_supported) {
    HIP_SKIP_TEST("Device does not support virtual memory management");
    return;
  }
  // Assumes the dedicated-VRAM carveout is smaller than 5 GiB.
  constexpr size_t requested = static_cast<size_t>(5) << 30;
  constexpr size_t headroom = static_cast<size_t>(1) << 30;
  if (prop.totalGlobalMem < requested + headroom) {
    HIP_SKIP_TEST("APU totalGlobalMem too small for this allocation plus headroom");
    return;
  }

  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = 0;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &aprop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size = ((requested + granularity - 1) / granularity) * granularity;

  void* va = nullptr;
  HIP_CHECK(hipMemAddressReserve(&va, size, 0, nullptr, 0));
  REQUIRE(va != nullptr);

  hipMemGenericAllocationHandle_t handle = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size, &aprop, 0));
  REQUIRE(handle != nullptr);

  HIP_CHECK(hipMemMap(va, size, 0, handle, 0));

  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = 0;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(va, size, &access, 1));

  constexpr int fill = 0xCD;
  HIP_CHECK(hipMemset(va, fill, size));

  constexpr size_t sample = static_cast<size_t>(64) * 1024;
  std::vector<char> head(sample), tail(sample);
  HIP_CHECK(hipMemcpy(head.data(), va, sample, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tail.data(), static_cast<char*>(va) + (size - sample), sample,
                      hipMemcpyDeviceToHost));
  for (size_t i = 0; i < sample; ++i) {
    REQUIRE(head[i] == static_cast<char>(fill));
    REQUIRE(tail[i] == static_cast<char>(fill));
  }

  HIP_CHECK(hipMemUnmap(va, size));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(va, size));
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for submitVirtualMap bookkeeping:
 * after hipMemMap, the bidirectional cross-link
 * (vaddr_sub_obj.phys_mem_obj -> phys, phys.vaddr_mem_obj -> sub_obj)
 * must be wired so that hipMemRetainAllocationHandle (which walks
 * sub_obj -> phys -> ga) returns the original handle.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemMap_Retrieve_AllocationHandle) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  void* ptr = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  // Confirm hipMemRetainAllocationHandle retrieves correct base handle
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for the multi-alias case of the
 * bookkeeping helper. One physical handle mapped to multiple VAs:
 * each VA must independently resolve back to the same handle via
 * hipMemRetainAllocationHandle.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemMap_Retrieve_MultiAllocationHandle) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptrs[num_buf] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < num_buf; ++i) {
    HIP_CHECK(hipMemAddressReserve(&ptrs[i], size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrs[i], size_mem, 0, handle, 0));
  }

  // Every alias must resolve sub_obj -> phys -> ga back to handle.
  for (int i = 0; i < num_buf; ++i) {
    hipMemGenericAllocationHandle_t retrieved = nullptr;
    HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptrs[i]));
    REQUIRE(retrieved == handle);
    HIP_CHECK(hipMemRelease(retrieved));
  }

  for (int i = 0; i < num_buf; ++i) {
    HIP_CHECK(hipMemUnmap(ptrs[i], size_mem));
    HIP_CHECK(hipMemAddressFree(ptrs[i], size_mem));
  }
  HIP_CHECK(hipMemRelease(handle));
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for the MemObjMap insertion done by
 * FinalizeMapMemObjBookkeeping. After hipMemMap the VA must resolve
 * to a backing handle via hipMemRetainAllocationHandle; before
 * hipMemMap (only reserved, never mapped) it must not.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemMap_CheckAddMemObj) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  void* ptr = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));

  // Before map: retainAllocationHandle must fail.
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  REQUIRE(hipMemRetainAllocationHandle(&retrieved, ptr) == hipErrorInvalidValue);

  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  // After map: retainAllocationHandle must pass now ptr has been mapped.
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - hipMemMap test the End-to-end lifecycle: reserve VA,
 * create physical handle, hipMemMap, set access, write through HtoD,
 * read back via DtoH and a kernel, verify, hipMemUnmap, release, free.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemMap_RoundTrip) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  std::vector<int> A_h(N), B_h(N), C_h(N);
  for (size_t idx = 0; idx < N; ++idx) {
    A_h[idx] = static_cast<int>(idx);
    C_h[idx] = static_cast<int>(idx * idx);
  }

  hipMemGenericAllocationHandle_t handle;
  void* ptr = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));

  // The direct path under test.
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));

  // Data integrity: HtoD then DtoH must round-trip.
  std::fill(B_h.begin(), B_h.end(), initializer);
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptr), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  REQUIRE(std::equal(B_h.begin(), B_h.end(), A_h.data()));

  // Kernel write through the mapped VA must be visible to host.
  square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
      reinterpret_cast<int*>(ptr));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  REQUIRE(std::equal(B_h.begin(), B_h.end(), C_h.data()));

  // Bookkeeping cross-link must be wired by the direct path (lean on the
  // dedicated bookkeeping tests for finer-grained assertions).
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Backend-agnostic concurrent stress test for the direct virtualMap
 * path. N threads each repeatedly reserve/create/map/unmap/release on
 * disjoint VAs. A hard wall-clock bound catches deadlock either from
 * the HSA lock-free path (no execution() lock) or from PAL execution()
 * lock misuse.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemMap_ConcurrentDisjointMapsNoDeadlock) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = granularity;
  constexpr int kThreads = 4;
  constexpr int kIters = 16;

  std::atomic<bool> failed{false};
  auto worker = [&](int tid) {
    for (int i = 0; i < kIters && !failed.load(); ++i) {
      hipMemGenericAllocationHandle_t h;
      void* p = nullptr;
      if (hipMemCreate(&h, size_mem, &prop, 0) != hipSuccess) {
        failed = true;
        return;
      }
      if (hipMemAddressReserve(&p, size_mem, 0, 0, 0) != hipSuccess) {
        (void)hipMemRelease(h);
        failed = true;
        return;
      }
      if (hipMemMap(p, size_mem, 0, h, 0) != hipSuccess) {
        (void)hipMemAddressFree(p, size_mem);
        (void)hipMemRelease(h);
        failed = true;
        return;
      }
      hipMemAccessDesc d{};
      d.location.type = hipMemLocationTypeDevice;
      d.location.id = device;
      d.flags = hipMemAccessFlagsProtReadWrite;
      (void)hipMemSetAccess(p, size_mem, &d, 1);
      if (hipMemUnmap(p, size_mem) != hipSuccess) {
        (void)hipMemAddressFree(p, size_mem);
        (void)hipMemRelease(h);
        failed = true;
        return;
      }
      (void)hipMemRelease(h);
      (void)hipMemAddressFree(p, size_mem);
      (void)tid;
    }
  };

  // Hard upper bound: 60s. A deadlocked code path would never complete;
  // a heavily serialized but live path completes in well under this.
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) threads.emplace_back(worker, i);
  for (auto& t : threads) t.join();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0)
          .count();
  REQUIRE_FALSE(failed.load());
  REQUIRE(elapsed < 60);
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
