/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kByteCount = 256;

int DeviceCount() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count;
}

// Returns ordered pairs for which the runtime does not expose peer access. On a
// symmetric fabric this is empty and the tests below skip.
std::vector<std::pair<int, int>> DeniedPairs() {
  std::vector<std::pair<int, int>> pairs;
  const int count = DeviceCount();
  for (int src = 0; src < count; ++src) {
    for (int dst = 0; dst < count; ++dst) {
      if (src == dst) {
        continue;
      }
      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
      if (can_access == 0) {
        pairs.emplace_back(src, dst);
      }
    }
  }
  return pairs;
}

std::array<uint8_t, kByteCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kByteCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

void* AllocateOn(int device, hip::contract::ContractCleanup* cleanup) {
  HIP_CHECK(hipSetDevice(device));
  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, kByteCount));
  cleanup->Add([device, ptr] {
    (void)hipSetDevice(device);
    (void)hipFree(ptr);
  });
  return ptr;
}
}  // namespace

// A direction that hipDeviceCanAccessPeer rejects has no direct peer mapping,
// but the runtime has always completed these copies through the host-staged
// P2P path. It must not fail or abort.
// @asserts: hipMemcpyPeer - a peer direction rejected by hipDeviceCanAccessPeer completes through the host-staged path
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeer_DeniedDirection_StagesCopy) {
  const auto denied = DeniedPairs();
  if (denied.empty()) {
    HIP_SKIP_TEST("No asymmetric peer direction on this topology.");
  }

  for (const auto& [src, dst] : denied) {
    INFO("Staged direction: " << src << " -> " << dst);
    hip::contract::ContractCleanup cleanup;
    void* src_buf = AllocateOn(src, &cleanup);
    void* dst_buf = AllocateOn(dst, &cleanup);
    const auto pattern = MakePattern(0x5a);

    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipMemcpy(src_buf, pattern.data(), kByteCount, hipMemcpyHostToDevice));

    // Execute on the source. The denied direction src -> dst has no direct peer
    // mapping, so this copy must go through the host-staged path.
    HIP_CHECK(hipMemcpyPeer(dst_buf, dst, src_buf, src, kByteCount));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipSetDevice(dst));
    std::array<uint8_t, kByteCount> result{};
    HIP_CHECK(hipMemcpy(result.data(), dst_buf, kByteCount, hipMemcpyDeviceToHost));
    REQUIRE(result == pattern);
  }
}

// The async form must stage on the caller's stream and complete before the
// stream is synchronized.
// @asserts: hipMemcpyPeerAsync - a peer direction rejected by hipDeviceCanAccessPeer stages on the caller stream and completes
HIP_TEST_CASE(Contract_PeerCopy_HipMemcpyPeerAsync_DeniedDirection_StagesCopy) {
  const auto denied = DeniedPairs();
  if (denied.empty()) {
    HIP_SKIP_TEST("No asymmetric peer direction on this topology.");
  }

  for (const auto& [src, dst] : denied) {
    INFO("Staged direction: " << src << " -> " << dst);
    hip::contract::ContractCleanup cleanup;
    void* src_buf = AllocateOn(src, &cleanup);
    void* dst_buf = AllocateOn(dst, &cleanup);
    const auto pattern = MakePattern(0xa5);

    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipMemcpy(src_buf, pattern.data(), kByteCount, hipMemcpyHostToDevice));

    // A stream on the source makes the source the executing device, so the
    // denied direction must stage instead of issuing a direct peer blit.
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    cleanup.Add([src, stream] {
      (void)hipSetDevice(src);
      (void)hipStreamDestroy(stream);
    });

    HIP_CHECK(hipMemcpyPeerAsync(dst_buf, dst, src_buf, src, kByteCount, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipSetDevice(dst));
    std::array<uint8_t, kByteCount> result{};
    HIP_CHECK(hipMemcpy(result.data(), dst_buf, kByteCount, hipMemcpyDeviceToHost));
    REQUIRE(result == pattern);
  }
}
