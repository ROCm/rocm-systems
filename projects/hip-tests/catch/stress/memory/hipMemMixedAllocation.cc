/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace {
constexpr size_t kChunkSize = size_t{32} << 20;
constexpr size_t kPairs = 16;

struct MixedAllocation {
  void* address = nullptr;
  void* ordinary = nullptr;
  hipMemGenericAllocationHandle_t handle{};
  bool created = false;
  bool mapped = false;

  void release() {
    if (ordinary != nullptr) {
      CHECK(hipFree(ordinary) == hipSuccess);
      ordinary = nullptr;
    }
    if (mapped) {
      CHECK(hipMemUnmap(address, kChunkSize) == hipSuccess);
      mapped = false;
    }
    if (created) {
      CHECK(hipMemRelease(handle) == hipSuccess);
      created = false;
    }
    if (address != nullptr) {
      CHECK(hipMemAddressFree(address, kChunkSize) == hipSuccess);
      address = nullptr;
    }
  }

  ~MixedAllocation() { release(); }
};

void checkPattern(void* address, size_t size, unsigned char expected,
                  std::vector<unsigned char>& readback) {
  for (size_t offset = 0; offset < size; offset += readback.size()) {
    INFO("byte offset " << offset);
    const size_t bytes = std::min(readback.size(), size - offset);
    HIP_CHECK(hipMemcpy(readback.data(), static_cast<unsigned char*>(address) + offset, bytes,
                        hipMemcpyDeviceToHost));
    REQUIRE(std::all_of(readback.begin(), readback.begin() + bytes,
                        [expected](unsigned char value) { return value == expected; }));
  }
}
}  // namespace

// This bounded stress test does not attempt to exhaust system RAM. The oversized
// request checks argument rejection, not allocator/residency exhaustion or a
// driver's behavior near the system commit limit.
HIP_TEST_CASE(Stress_hipMemMixedAllocation_Recovery) {
  CTX_CREATE();
  HIP_CHECK(hipSetDevice(0));
  int vmm_supported = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&vmm_supported, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (vmm_supported == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
  }
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity != 0);
  if (kChunkSize % granularity != 0) {
    HIP_SKIP_TEST("The 32 MiB regression fixture requires compatible VMM granularity.");
  }
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  std::vector<unsigned char> readback(64 * 1024);
  size_t released_bytes = 0;
  {
    // Touch all host pages while keeping up to 1 GiB of mixed GPU allocations.
    std::vector<unsigned char> host_ballast(size_t{256} << 20, 0x6d);
    std::array<MixedAllocation, kPairs> allocations{};
    size_t completed = 0;
    for (size_t pair = 0; pair < kPairs; ++pair) {
      INFO("allocation pair " << pair);
      auto& allocation = allocations[pair];
      HIP_CHECK(hipMemAddressReserve(&allocation.address, kChunkSize, 0, nullptr, 0));
      const hipError_t create_result = hipMemCreate(&allocation.handle, kChunkSize, &prop, 0);
      INFO("hipMemCreate result: " << hipGetErrorName(create_result));
      if (create_result == hipErrorOutOfMemory) {
        (void)hipGetLastError();
        break;
      }
      HIP_CHECK(create_result);
      allocation.created = true;
      HIP_CHECK(hipMemMap(allocation.address, kChunkSize, 0, allocation.handle, 0));
      allocation.mapped = true;
      HIP_CHECK(hipMemSetAccess(allocation.address, kChunkSize, &access, 1));
      HIP_CHECK(hipMemset(allocation.address, static_cast<int>(pair + 1), kChunkSize));
      released_bytes += kChunkSize;

      const hipError_t malloc_result = hipMalloc(&allocation.ordinary, kChunkSize);
      INFO("hipMalloc result: " << hipGetErrorName(malloc_result));
      if (malloc_result == hipErrorOutOfMemory) {
        (void)hipGetLastError();
        break;
      }
      HIP_CHECK(malloc_result);
      HIP_CHECK(hipMemset(allocation.ordinary, static_cast<int>(pair + 65), kChunkSize));
      released_bytes += kChunkSize;
      ++completed;
    }
    HIP_CHECK(hipDeviceSynchronize());
    INFO("completed mixed allocation pairs: " << completed);
    REQUIRE(completed > 0);

    // Reject an impossible allocation while valid VMM and hipMalloc allocations
    // remain live. Verify they survive the error before releasing any of them.
    void* rejected = nullptr;
    const hipError_t rejected_result = hipMalloc(&rejected, std::numeric_limits<size_t>::max());
    if (rejected_result == hipSuccess) HIP_CHECK(hipFree(rejected));
    HIP_CHECK_ERROR(rejected_result, hipErrorOutOfMemory);
    (void)hipGetLastError();
    for (size_t pair = 0; pair < kPairs; ++pair) {
      INFO("verify allocation pair " << pair);
      auto& allocation = allocations[pair];
      if (allocation.mapped) {
        checkPattern(allocation.address, kChunkSize, static_cast<unsigned char>(pair + 1),
                     readback);
      }
      if (allocation.ordinary != nullptr) {
        checkPattern(allocation.ordinary, kChunkSize, static_cast<unsigned char>(pair + 65),
                     readback);
      }
    }
    REQUIRE(std::all_of(host_ballast.begin(), host_ballast.end(),
                        [](unsigned char value) { return value == 0x6d; }));
  }

  // Reacquire all released GPU bytes, including any partially completed pair.
  // The host ballast has also been released. This checks capacity recovery but
  // is not a leak detector on a device with ample spare memory.
  MixedAllocation recovery;
  HIP_CHECK(hipMalloc(&recovery.ordinary, released_bytes));
  HIP_CHECK(hipMemset(recovery.ordinary, 0xa7, released_bytes));
  HIP_CHECK(hipDeviceSynchronize());
  checkPattern(recovery.ordinary, released_bytes, 0xa7, readback);
  recovery.release();
  HIP_CHECK(hipGetLastError());
  CTX_DESTROY();
}
