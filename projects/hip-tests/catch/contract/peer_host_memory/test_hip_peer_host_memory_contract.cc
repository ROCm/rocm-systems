/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kElementCount = 1024;
constexpr size_t kByteCount = kElementCount * sizeof(uint32_t);

void RequireMultiGpu(int* count) {
  HIP_CHECK(hipGetDeviceCount(count));
  if (*count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
}
}  // namespace

// Regression test for ROCM-34275/ROCM-34281: portable+mapped host memory must
// be accessible from a peer GPU via hipHostGetDevicePointer.  A previous change
// skipped MakeResident for peer GART mappings, leaving the peer GPU's page
// tables unpopulated and causing TDR (2x discrete) or silent data corruption
// (APU + discrete).

// @asserts: hipHostGetDevicePointer - peer device pointer for Portable|Mapped host memory round-trips bytes via hipMemset and hipMemcpy
HIP_TEST_CASE(Contract_PeerHostMemory_PortableMapped_PeerMemsetRoundTrips) {
  hip::contract::ContractCleanup cleanup;
  int device_count = 0;
  RequireMultiGpu(&device_count);

  const int owner_device = 0;
  const int peer_device = 1;

  // Allocate portable+mapped host memory on the owner device
  HIP_CHECK(hipSetDevice(owner_device));
  void* host_ptr = nullptr;
  HIP_CHECK(hipHostMalloc(&host_ptr, kByteCount,
                          hipHostMallocPortable | hipHostMallocMapped));
  cleanup.Add([host_ptr] { (void)hipHostFree(host_ptr); });
  REQUIRE(host_ptr != nullptr);
  std::memset(host_ptr, 0, kByteCount);

  // Get device-visible pointer on the peer device
  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&peer_dptr, host_ptr, 0));
  REQUIRE(peer_dptr != nullptr);

  // Write from the peer GPU via hipMemsetD8
  HIP_CHECK(hipMemset(peer_dptr, 0xAB, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify on the host
  auto* bytes = static_cast<const uint8_t*>(host_ptr);
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(bytes[i] == 0xAB);
  }
}

// @asserts: hipHostGetDevicePointer - peer device pointer for Portable|Mapped host memory round-trips structured data via hipMemcpy
HIP_TEST_CASE(Contract_PeerHostMemory_PortableMapped_PeerMemcpyRoundTrips) {
  hip::contract::ContractCleanup cleanup;
  int device_count = 0;
  RequireMultiGpu(&device_count);

  const int owner_device = 0;
  const int peer_device = 1;

  HIP_CHECK(hipSetDevice(owner_device));
  void* host_ptr = nullptr;
  HIP_CHECK(hipHostMalloc(&host_ptr, kByteCount,
                          hipHostMallocPortable | hipHostMallocMapped));
  cleanup.Add([host_ptr] { (void)hipHostFree(host_ptr); });
  std::memset(host_ptr, 0, kByteCount);

  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&peer_dptr, host_ptr, 0));

  // Prepare a pattern on the host and copy it TO the peer device pointer
  std::array<uint32_t, kElementCount> src_pattern{};
  for (size_t i = 0; i < kElementCount; ++i) {
    src_pattern[i] = static_cast<uint32_t>(0xDEAD0000u + i);
  }
  HIP_CHECK(hipMemcpy(peer_dptr, src_pattern.data(), kByteCount, hipMemcpyHostToDevice));

  // Read back FROM the peer device pointer
  std::array<uint32_t, kElementCount> dst{};
  HIP_CHECK(hipMemcpy(dst.data(), peer_dptr, kByteCount, hipMemcpyDeviceToHost));

  REQUIRE(dst == src_pattern);
}

// @asserts: hipHostGetDevicePointer - both owner and peer device pointers see the same data after either writes
HIP_TEST_CASE(Contract_PeerHostMemory_PortableMapped_OwnerAndPeerSeeConsistentData) {
  hip::contract::ContractCleanup cleanup;
  int device_count = 0;
  RequireMultiGpu(&device_count);

  const int owner_device = 0;
  const int peer_device = 1;

  HIP_CHECK(hipSetDevice(owner_device));
  void* host_ptr = nullptr;
  HIP_CHECK(hipHostMalloc(&host_ptr, kByteCount,
                          hipHostMallocPortable | hipHostMallocMapped));
  cleanup.Add([host_ptr] { (void)hipHostFree(host_ptr); });

  // Get device pointers for both devices
  void* owner_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&owner_dptr, host_ptr, 0));

  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&peer_dptr, host_ptr, 0));

  // Owner writes 0x11
  HIP_CHECK(hipSetDevice(owner_device));
  HIP_CHECK(hipMemset(owner_dptr, 0x11, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Peer reads — should see 0x11
  std::array<uint8_t, kByteCount> readback{};
  HIP_CHECK(hipSetDevice(peer_device));
  HIP_CHECK(hipMemcpy(readback.data(), peer_dptr, kByteCount, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(readback[i] == 0x11);
  }

  // Peer writes 0x22
  HIP_CHECK(hipMemset(peer_dptr, 0x22, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Host reads — should see 0x22
  auto* bytes = static_cast<const uint8_t*>(host_ptr);
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(bytes[i] == 0x22);
  }
}
