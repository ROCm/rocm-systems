/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemCreate hipMemCreate
 * @{
 * @ingroup VirtualMemoryManagementTest
 * Tests for host-NUMA-backed allocations via hipMemCreate, exercising
 *   - hipMemLocationTypeHostNuma         (id = NUMA node id)
 *   - hipMemLocationTypeHostNumaCurrent  (id ignored; node closest to caller's CPU)
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#if __has_include(<numa.h>)
#include <numa.h>
#define HAVE_LIBNUMA 1
#else
#define HAVE_LIBNUMA 0
#endif

#define THREADS_PER_BLOCK 512
#define DATA_SIZE (1 << 13)

namespace {

// Return a NUMA node id that is known to exist on the system.
// Falls back to 0 (which is always present on NUMA-aware Linux).
int firstAvailableNumaNode() {
#if HAVE_LIBNUMA
  if (numa_available() != -1) {
    for (int n = 0; n <= numa_max_node(); ++n) {
      if (numa_bitmask_isbitset(numa_get_mems_allowed(), n)) return n;
    }
  }
#endif
  return 0;
}

// Return an id guaranteed to NOT correspond to any configured NUMA node.
int invalidNumaNode() {
#if HAVE_LIBNUMA
  if (numa_available() != -1) {
    return numa_max_node() + 1024;
  }
#endif
  return 1 << 20;  // wildly out of range
}

// Kernel that squares each element. Reused across tests below.
__global__ void square_kernel(int* buf) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  buf[i] = buf[i] * buf[i];
}

// Build a host-NUMA allocation prop with the given location type / id.
hipMemAllocationProp makeHostNumaProp(hipMemLocationType locType, int locId) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = locType;
  prop.location.id = locId;
  return prop;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Allocate and release physical memory of multiples of granularity for
 *      both hipMemLocationTypeHostNuma (specific node) and
 *      hipMemLocationTypeHostNumaCurrent (current CPU's node).
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreateHostNuma.cc
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_BasicAllocDealloc) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop{};
  SECTION("HostNuma node 0") {
    prop = makeHostNumaProp(hipMemLocationTypeHostNuma, firstAvailableNumaNode());
  }
  SECTION("HostNumaCurrent (id ignored)") {
    prop = makeHostNumaProp(hipMemLocationTypeHostNumaCurrent, 0);
  }

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t handle;
  for (int mul = 1; mul < 8; ++mul) {
    HIP_CHECK(hipMemCreate(&handle, granularity * mul, &prop, 0));
    HIP_CHECK(hipMemRelease(handle));
  }
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Verify that the props recovered via
 *      hipMemGetAllocationPropertiesFromHandle round-trip the host-NUMA
 *      location info.
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_PropRoundTrip) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  const int node = firstAvailableNumaNode();
  hipMemAllocationProp prop = makeHostNumaProp(hipMemLocationTypeHostNuma, node);

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity, &prop, 0));

  hipMemAllocationProp got{};
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&got, handle));
  REQUIRE(got.location.type == hipMemLocationTypeHostNuma);
  REQUIRE(got.location.id == node);

  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Map a host-NUMA backed physical allocation into a virtual address
 *      range, grant device R/W access, run HtoD/DtoH copies and verify.
 *      Repeated for both HostNuma and HostNumaCurrent.
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_MapAndMemcpyRoundTrip) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop{};
  SECTION("HostNuma node 0") {
    prop = makeHostNumaProp(hipMemLocationTypeHostNuma, firstAvailableNumaNode());
  }
  SECTION("HostNumaCurrent") {
    prop = makeHostNumaProp(hipMemLocationTypeHostNumaCurrent, 0);
  }

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));

  std::vector<int> A_h(N), B_h(N);
  for (int i = 0; i < N; ++i) A_h[i] = i;
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptr), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  REQUIRE(std::equal(A_h.begin(), A_h.end(), B_h.begin()));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Launch a kernel against a host-NUMA backed mapping and validate the
 *      result. Confirms the GPU can read/write the host-resident, NUMA-bound
 *      physical allocation.
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_KernelLaunch) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop =
      makeHostNumaProp(hipMemLocationTypeHostNuma, firstAvailableNumaNode());

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

  void* ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemRelease(handle));

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));

  std::vector<int> A_h(N), B_h(N), Expected(N);
  for (int i = 0; i < N; ++i) {
    A_h[i] = i;
    Expected[i] = i * i;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptr), A_h.data(), buffer_size));
  hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     reinterpret_cast<int*>(ptr));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptr), buffer_size));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(std::equal(Expected.begin(), Expected.end(), B_h.begin()));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative tests for the NUMA host location types.
 */
HIP_TEST_CASE(Unit_hipMemCreate_HostNuma_Negative) {
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop =
      makeHostNumaProp(hipMemLocationTypeHostNuma, firstAvailableNumaNode());
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t handle;

  SECTION("HostNuma with negative node id") {
    prop.location.id = -1;
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("HostNuma with out-of-range node id") {
    prop.location.id = invalidNumaNode();
    const hipError_t err = hipMemCreate(&handle, granularity, &prop, 0);
    REQUIRE((err == hipErrorInvalidValue || err == hipErrorInvalidDevice));
  }

  SECTION("HostNuma with size not multiple of granularity") {
    REQUIRE(hipMemCreate(&handle, granularity - 1, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("HostNumaCurrent with non-zero flags") {
    prop.location.type = hipMemLocationTypeHostNumaCurrent;
    prop.location.id = 0;
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 1) == hipErrorInvalidValue);
  }

  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
