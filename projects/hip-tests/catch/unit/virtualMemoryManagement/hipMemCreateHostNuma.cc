/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemCreate hipMemCreate
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemCreate (hipMemGenericAllocationHandle_t* handle,
 *                           size_t size,
 *                           const hipMemAllocationProp* prop,
 *                           unsigned long long flags)` -
 * Creates a host-resident, NUMA-aware physical memory allocation using
 * hipMemLocationTypeHostNuma / hipMemLocationTypeHostNumaCurrent.
 */

#include <numa.h>
#include <numaif.h>
#include <sched.h>

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define THREADS_PER_BLOCK 512
#define DATA_SIZE (1 << 13)

namespace {

// Square the input data in place.
__global__ void square_kernel(int* buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = buff[i] * buff[i];
  buff[i] = temp;
}

// Build a host-NUMA allocation property for the given location type / id.
hipMemAllocationProp makeHostNumaProp(hipMemLocationType locType, int id) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = locType;
  prop.location.id = id;
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  return prop;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Query allocation granularity for host-NUMA location types and verify
 * it is non-zero for every NUMA node and for the "current" variant.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_Granularity) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  SECTION("hipMemLocationTypeHostNuma per node") {
    for (int node = 0; node < numNodes; ++node) {
      hipMemAllocationProp prop = makeHostNumaProp(hipMemLocationTypeHostNuma, node);
      size_t granularity = 0;
      HIP_CHECK(
          hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
      REQUIRE(granularity > 0);
    }
  }

  SECTION("hipMemLocationTypeHostNumaCurrent") {
    hipMemAllocationProp prop = makeHostNumaProp(hipMemLocationTypeHostNumaCurrent, 0);
    size_t granularity = 0;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate and release host-NUMA physical memory for different multiples
 * of granularity, for every NUMA node and the "current" variant.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_BasicAllocDealloc) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  hipMemLocationType locType = hipMemLocationTypeHostNuma;
  int locId = 0;
  SECTION("hipMemLocationTypeHostNuma node 0") {
    locType = hipMemLocationTypeHostNuma;
    locId = 0;
  }
  SECTION("hipMemLocationTypeHostNuma last node") {
    locType = hipMemLocationTypeHostNuma;
    locId = numNodes - 1;
  }
  SECTION("hipMemLocationTypeHostNumaCurrent") {
    locType = hipMemLocationTypeHostNumaCurrent;
    locId = 0;
  }

  hipMemAllocationProp prop = makeHostNumaProp(locType, locId);
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t handle;
  for (int mul = 1; mul < 16; ++mul) {
    HIP_CHECK(hipMemCreate(&handle, granularity * mul, &prop, 0));
    HIP_CHECK(hipMemRelease(handle));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate host-NUMA physical memory, map it to a VA range, grant device
 * access, copy host->VMM->host and verify the data round-trips.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_ChkDev2HstMemcpy) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  hipMemLocationType locType = hipMemLocationTypeHostNuma;
  int locId = 0;
  SECTION("hipMemLocationTypeHostNuma node 0") {
    locType = hipMemLocationTypeHostNuma;
    locId = 0;
  }
  SECTION("hipMemLocationTypeHostNuma last node") {
    locType = hipMemLocationTypeHostNuma;
    locId = numNodes - 1;
  }
  SECTION("hipMemLocationTypeHostNumaCurrent") {
    locType = hipMemLocationTypeHostNumaCurrent;
    locId = 0;
  }

  hipMemAllocationProp prop = makeHostNumaProp(locType, locId);
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = ((buffer_size + granularity - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptrA = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; ++idx) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate host-NUMA physical memory, map it, grant device access, copy
 * data to it, launch a kernel to square the values and verify the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_ChkWithKerLaunch) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  hipMemLocationType locType = hipMemLocationTypeHostNuma;
  int locId = 0;
  SECTION("hipMemLocationTypeHostNuma node 0") {
    locType = hipMemLocationTypeHostNuma;
    locId = 0;
  }
  SECTION("hipMemLocationTypeHostNuma last node") {
    locType = hipMemLocationTypeHostNuma;
    locId = numNodes - 1;
  }
  SECTION("hipMemLocationTypeHostNumaCurrent") {
    locType = hipMemLocationTypeHostNumaCurrent;
    locId = 0;
  }

  hipMemAllocationProp prop = makeHostNumaProp(locType, locId);
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = ((buffer_size + granularity - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptrA = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemRelease(handle));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

  std::vector<int> A_h(N), B_h(N), C_h(N);
  for (int idx = 0; idx < N; ++idx) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     reinterpret_cast<int*>(ptrA));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate host-NUMA memory, grant both host and device access, then verify
 * the CPU can directly read/write the mapping and the GPU observes the same
 * memory (host writes -> kernel squares in place -> CPU reads result). This
 * exercises the host-resident, CPU-accessible nature of the allocation.
 *
 *    - NOTE: the actual NUMA node residency is NOT asserted here. The backing is
 * a dma-buf / GTT mapping managed by the amdgpu driver, not a normal anonymous
 * VMA, so userspace NUMA introspection (move_pages(), get_mempolicy(),
 * /proc/self/numa_maps) cannot resolve its node. Node selection is a code-level
 * guarantee in ROCclr (hostVmemAlloc picks cpu_agents_[node]'s pool); verifying
 * it from userspace would require root-only KFD topology/debugfs.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_HostAndDeviceAccess) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  hipMemLocationType locType = hipMemLocationTypeHostNuma;
  int locId = 0;
  SECTION("hipMemLocationTypeHostNuma node 0") {
    locType = hipMemLocationTypeHostNuma;
    locId = 0;
  }
  SECTION("hipMemLocationTypeHostNuma last node") {
    locType = hipMemLocationTypeHostNuma;
    locId = numNodes - 1;
  }
  SECTION("hipMemLocationTypeHostNumaCurrent") {
    locType = hipMemLocationTypeHostNumaCurrent;
    locId = 0;
  }

  hipMemAllocationProp prop = makeHostNumaProp(locType, locId);
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = ((buffer_size + granularity - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptrA = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  hipMemAccessDesc descs[2] = {};
  descs[0].location.type = hipMemLocationTypeHost;
  descs[0].location.id = 0;
  descs[0].flags = hipMemAccessFlagsProtReadWrite;
  descs[1].location.type = hipMemLocationTypeDevice;
  descs[1].location.id = device;
  descs[1].flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &descs[0], 1));
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &descs[1], 1));

  // CPU writes directly into the host-resident mapping.
  int* hostPtr = reinterpret_cast<int*>(ptrA);
  for (int idx = 0; idx < N; ++idx) {
    hostPtr[idx] = idx;
  }

  // GPU squares the same memory in place.
  hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     reinterpret_cast<int*>(ptrA));
  HIP_CHECK(hipDeviceSynchronize());

  // CPU reads the GPU's results back directly.
  for (int idx = 0; idx < N; ++idx) {
    REQUIRE(hostPtr[idx] == idx * idx);
  }

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative tests for host-NUMA hipMemCreate: invalid NUMA node ids and
 * invalid flags.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_Negative) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  if (numa_available() < 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostNumaUnavailable);
  }
  const int numNodes = numa_max_node() + 1;

  hipMemAllocationProp prop = makeHostNumaProp(hipMemLocationTypeHostNuma, 0);
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t handle;

  SECTION("HostNuma id out of range") {
    prop.location.id = numNodes;  // one past the last valid node
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("HostNuma id negative") {
    prop.location.id = -1;
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("HostNuma non-zero flags") {
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 1) == hipErrorInvalidValue);
  }

  SECTION("HostNuma size not a multiple of granularity") {
    REQUIRE(hipMemCreate(&handle, granularity - 1, &prop, 0) == hipErrorInvalidValue);
  }

  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
