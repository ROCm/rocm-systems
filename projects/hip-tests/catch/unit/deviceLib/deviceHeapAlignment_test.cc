/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <hip_test_common.hh>

// This is the OCKL request used to grow the device heap. Calling it directly
// exposes the slab base; malloc returns a suballocation within a slab instead.
extern "C" __device__ unsigned long long __ockl_devmem_request(unsigned long long address,
                                                            unsigned long long size);

namespace {
constexpr size_t kSlabBytes = 2 * 1024 * 1024;
constexpr size_t kRequests = 4;

struct SlabResult {
  unsigned long long address;
  unsigned int first;
  unsigned int last;
};

__global__ void requestHeapSlabs(SlabResult* results) {
  for (size_t i = 0; i < kRequests; ++i) {
    const auto address = __ockl_devmem_request(0, kSlabBytes);
    results[i] = {address, 0, 0};
    if (address != 0) {
      auto* words = reinterpret_cast<volatile unsigned int*>(address);
      words[0] = static_cast<unsigned int>(i + 1);
      words[kSlabBytes / sizeof(unsigned int) - 1] = static_cast<unsigned int>(i + 17);
      results[i].first = words[0];
      results[i].last = words[kSlabBytes / sizeof(unsigned int) - 1];
      __ockl_devmem_request(address, 0);
    }
  }
}
}  // namespace

HIP_TEST_CASE(Unit_deviceHeap_HostcallSlabAlignment) {
  CHECK_PCIE_ATOMIC_SUPPORT;

  std::array<SlabResult, kRequests> results{};
  SlabResult* device_results = nullptr;
  HIP_CHECK(hipMalloc(&device_results, sizeof(results)));
  requestHeapSlabs<<<1, 1>>>(device_results);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(results.data(), device_results, sizeof(results), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_results));

  for (size_t i = 0; i < results.size(); ++i) {
    INFO("slab request " << i << ", address " << results[i].address);
    REQUIRE(results[i].address != 0);
    REQUIRE(results[i].address % kSlabBytes == 0);
    REQUIRE(results[i].first == i + 1);
    REQUIRE(results[i].last == i + 17);
  }
}
