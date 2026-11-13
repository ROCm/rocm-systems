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
//
// All tests exercise the peer GPU's mapping by running operations on the peer
// device (hipMemset runs on the current device) or by copying through a device
// buffer allocated on the peer (avoiding ihipMemcpyCommand's redirect of
// pageable-host copies to the owning device).

// @asserts: hipHostGetDevicePointer - peer GPU can write mapped host memory via hipMemset, visible to the CPU
HIP_TEST_CASE(Contract_PeerHostMemory_PortableMapped_PeerMemsetRoundTrips) {
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
  REQUIRE(host_ptr != nullptr);
  std::memset(host_ptr, 0, kByteCount);

  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&peer_dptr, host_ptr, 0));
  REQUIRE(peer_dptr != nullptr);

  // hipMemset runs on the current device (peer_device), exercising the peer
  // GPU's page table mapping of the host memory.
  HIP_CHECK(hipMemset(peer_dptr, 0xAB, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify directly on the CPU — no hipMemcpy needed for mapped memory.
  auto* bytes = static_cast<const uint8_t*>(host_ptr);
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(bytes[i] == 0xAB);
  }
}

// @asserts: hipHostGetDevicePointer - peer GPU can transfer data through mapped host memory via a peer-device buffer
HIP_TEST_CASE(Contract_PeerHostMemory_PortableMapped_PeerDeviceCopyRoundTrips) {
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

  // Allocate a device buffer on the peer GPU to route copies through it,
  // ensuring the peer GPU's engine accesses peer_dptr (not the owner's).
  void* peer_dev_buf = nullptr;
  HIP_CHECK(hipMalloc(&peer_dev_buf, kByteCount));
  cleanup.Add([peer_dev_buf] { (void)hipFree(peer_dev_buf); });

  // Write a pattern into the host buffer from the CPU.
  auto* host_u32 = static_cast<uint32_t*>(host_ptr);
  for (size_t i = 0; i < kElementCount; ++i) {
    host_u32[i] = static_cast<uint32_t>(0xDEAD0000u + i);
  }

  // D2D copy on peer: peer_dptr → peer_dev_buf.  This makes the peer GPU
  // read from peer_dptr, exercising the peer GART mapping.
  HIP_CHECK(hipMemcpy(peer_dev_buf, peer_dptr, kByteCount, hipMemcpyDeviceToDevice));

  // Clear host buffer, then D2D copy on peer: peer_dev_buf → peer_dptr.
  // This makes the peer GPU write to peer_dptr.
  std::memset(host_ptr, 0, kByteCount);
  HIP_CHECK(hipMemcpy(peer_dptr, peer_dev_buf, kByteCount, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify the round-trip on the CPU.
  for (size_t i = 0; i < kElementCount; ++i) {
    REQUIRE(host_u32[i] == static_cast<uint32_t>(0xDEAD0000u + i));
  }
}

// @asserts: hipHostGetDevicePointer - owner and peer GPUs see consistent data through mapped host memory
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

  void* owner_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&owner_dptr, host_ptr, 0));

  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&peer_dptr, host_ptr, 0));

  // Owner GPU writes 0x11 via hipMemset (runs on owner_device).
  HIP_CHECK(hipSetDevice(owner_device));
  HIP_CHECK(hipMemset(owner_dptr, 0x11, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Peer GPU reads through a peer-device buffer to exercise peer mapping.
  HIP_CHECK(hipSetDevice(peer_device));
  void* peer_dev_buf = nullptr;
  HIP_CHECK(hipMalloc(&peer_dev_buf, kByteCount));
  cleanup.Add([peer_dev_buf] { (void)hipFree(peer_dev_buf); });

  // D2D on peer: peer_dptr → peer_dev_buf (peer GPU reads host memory).
  HIP_CHECK(hipMemcpy(peer_dev_buf, peer_dptr, kByteCount, hipMemcpyDeviceToDevice));
  // Copy result to host for verification.
  std::array<uint8_t, kByteCount> readback{};
  HIP_CHECK(hipMemcpy(readback.data(), peer_dev_buf, kByteCount, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(readback[i] == 0x11);
  }

  // Peer GPU writes 0x22 via hipMemset (runs on peer_device).
  HIP_CHECK(hipMemset(peer_dptr, 0x22, kByteCount));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify directly on the CPU.
  auto* bytes = static_cast<const uint8_t*>(host_ptr);
  for (size_t i = 0; i < kByteCount; ++i) {
    REQUIRE(bytes[i] == 0x22);
  }
}
