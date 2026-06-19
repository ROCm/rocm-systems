/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemUnmap hipMemUnmap
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemUnmap (void* ptr, size_t size)` -
 * Unmap memory allocation of a given address range.
 */


#include <hip_test_common.hh>

#include <chrono>

#include "hip_vmm_common.hh"

constexpr int N = (1 << 13);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_negative) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;

  CTX_CREATE();
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
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  SECTION("nullptr to ptrA") { REQUIRE(hipMemUnmap(nullptr, size_mem) == hipErrorInvalidValue); }

  SECTION("pass zero to size") { REQUIRE(hipMemUnmap(ptrA, 0) == hipErrorInvalidValue); }

  SECTION("unmap a smaller size") {
    REQUIRE(hipMemUnmap(ptrA, (size_mem - 1)) == hipErrorInvalidValue);
  }

  SECTION("double unmap of mapped VA") {
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    // Second call: the MemObjMap entry is gone, so the validator now
    // rejects the lookup. Must report hipErrorInvalidValue
    REQUIRE(hipMemUnmap(ptrA, size_mem) == hipErrorInvalidValue);
    // Re-map so the common-path cleanup at the end of the test still
    // finds a live mapping to tear down.
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  }

  SECTION("unmap of reserved-but-never-mapped VA") {
    void* unmappedVa = nullptr;
    HIP_CHECK(hipMemAddressReserve(&unmappedVa, size_mem, 0, 0, 0));
    // No hipMemMap has been issued against unmappedVa, so no entry
    // exists. Validation must reject with hipErrorInvalidValue
    REQUIRE(hipMemUnmap(unmappedVa, size_mem) == hipErrorInvalidValue);
    HIP_CHECK(hipMemAddressFree(unmappedVa, size_mem));
  }

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemUnmap_Capture) {
  CTX_CREATE();
  size_t granularity = 0;
  constexpr size_t kBufferSize = N * sizeof(int);
  int device_id = 0;
  hipDevice_t device;

  HIP_CHECK(hipDeviceGet(&device, device_id));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t mem_size = ((granularity + kBufferSize - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t allocation_handle;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMemCreate(&allocation_handle, mem_size, &allocation_prop, 0));
  HIP_CHECK(hipMemAddressReserve(&device_ptr, mem_size, 0, nullptr, 0));
  HIP_CHECK(hipMemMap(device_ptr, mem_size, 0, allocation_handle, 0));
  HIP_CHECK(hipMemRelease(allocation_handle));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemUnmap(device_ptr, mem_size));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemAddressFree(device_ptr, mem_size));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - After hipMemUnmap, the sub_obj must be removed from MemObjMap and the
 * vaddr<->phys cross-link torn down. hipMemRetainAllocationHandle on the VA must
 * return hipErrorInvalidValue.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_CrossLinksTornDown) {
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

  // Sanity: cross-link is wired pre-unmap.
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // After unmap: MemObjMap::RemoveMemObj must have run and the
  // cross-link must be torn down. retainAllocationHandle on the
  // still-reserved VA must now fail.
  REQUIRE(hipMemRetainAllocationHandle(&retrieved, ptr) == hipErrorInvalidValue);

  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for the map/unmap/remap cycle through
 * the bookkeeping helper. Pin that after an unmap the same VA can
 * be re-mapped (potentially with a different handle) and the
 * cross-link gets rewired to the new handle. Demonstrates the
 * unmap helper fully clears the prior MemObjMap entry.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_RemapCrossLinks) {
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

  hipMemGenericAllocationHandle_t handle1, handle2;
  HIP_CHECK(hipMemCreate(&handle1, size_mem, &prop, 0));
  HIP_CHECK(hipMemCreate(&handle2, size_mem, &prop, 0));

  void* ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));

  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle1, 0));
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle1);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // Helper cleared the slot -- remap with different handle must succeed
  // and the cross-link must point at handle2 (not the stale handle1).
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle2, 0));
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle2);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle1));
  HIP_CHECK(hipMemRelease(handle2));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Square kernel used by the direct-path round-trip and drain tests.
 */
static __global__ void unmap_square_kernel(int* buf) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int v = buf[i];
  buf[i] = v * v;
}

/**
 * Long-running compute kernel for the drain/sync tests below. Has a
 * data dependency on the loop so the compiler cannot DCE it.
 */
static __global__ void unmap_slow_marker_kernel(int* buf, int iters, int marker) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int v = 0;
  for (int k = 0; k < iters; ++k) {
    v += k;
  }
  buf[i] = marker + (v & 0);
}

/**
 * Test Description
 * ------------------------
 *    - Functional sanity that map/write/unmap/remap/write
 * round-trips work through the new direct hipMemUnmap.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_DirectPath_BasicRoundTrip) {
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

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  // First cycle: map, write via kernel, unmap (the direct path under test).
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptr), A_h.data(), buffer_size));
  unmap_square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
      reinterpret_cast<int*>(ptr));
  HIP_CHECK(hipStreamSynchronize(0));
  std::fill(B_h.begin(), B_h.end(), 0);
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  REQUIRE(std::equal(B_h.begin(), B_h.end(), C_h.data()));
  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // Second cycle: same VA + same handle must remap cleanly and read/write
  // through the new mapping must produce fresh data.
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptr), A_h.data(), buffer_size));
  unmap_square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
      reinterpret_cast<int*>(ptr));
  HIP_CHECK(hipStreamSynchronize(0));
  std::fill(B_h.begin(), B_h.end(), 0);
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  REQUIRE(std::equal(B_h.begin(), B_h.end(), C_h.data()));
  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Tests sync stream with access on unmap using a 
 * NON-default stream that hipMemUnmap does not implicitly own.
 *
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_DirectPath_InFlightKernelDrained) {
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

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int kSlowIters = 1 << 20;
  unmap_slow_marker_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, stream>>>(
      reinterpret_cast<int*>(ptr), kSlowIters, 7);

  // No explicit sync -- the direct hipMemUnmap path must drain the
  // access-device queues before tearing the mapping down.
  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // If SyncAllStreams was called for the access device, the stream
  // has no pending work. Otherwise hipStreamQuery returns hipErrorNotReady.
  REQUIRE(hipStreamQuery(stream) == hipSuccess);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
