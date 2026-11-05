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
constexpr size_t kByteCount = 64;

int DeviceCount() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count;
}

// Returns ordered pairs for which KFD does not expose peer access. On a
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

// @asserts: hipMemcpyPeer - a direction that hipDeviceCanAccessPeer rejects returns
// hipErrorPeerAccessUnsupported instead of aborting the process
HIP_TEST_CASE(Contract_PeerCopy_DeniedDirection_ReturnsPeerAccessUnsupported) {
  const auto denied = DeniedPairs();
  if (denied.empty()) {
    HIP_SKIP_TEST("No asymmetric peer direction on this topology.");
  }

  for (const auto& [src, dst] : denied) {
    INFO("Denied direction: " << src << " -> " << dst);
    hip::contract::ContractCleanup cleanup;
    void* src_buf = AllocateOn(src, &cleanup);
    void* dst_buf = AllocateOn(dst, &cleanup);

    HIP_CHECK_ERROR(hipMemcpyPeer(dst_buf, dst, src_buf, src, kByteCount),
                    hipErrorPeerAccessUnsupported);
    (void)hipGetLastError();

    // The copy stream can live on either endpoint. Both must reject the
    // direction before touching the peer BAR.
    for (const int stream_device : {src, dst}) {
      HIP_CHECK(hipSetDevice(stream_device));
      hipStream_t stream = nullptr;
      HIP_CHECK(hipStreamCreate(&stream));
      cleanup.Add([stream_device, stream] {
        (void)hipSetDevice(stream_device);
        (void)hipStreamDestroy(stream);
      });
      HIP_CHECK_ERROR(hipMemcpyPeerAsync(dst_buf, dst, src_buf, src, kByteCount, stream),
                      hipErrorPeerAccessUnsupported);
      (void)hipGetLastError();
    }
  }
}

// @asserts: hipMemcpy - a cross-device copy in a direction that is not peer
// addressable completes through the host-staged P2P path
HIP_TEST_CASE(Contract_PeerCopy_DeniedDirection_ImplicitCopyStages) {
  const auto denied = DeniedPairs();
  if (denied.empty()) {
    HIP_SKIP_TEST("No asymmetric peer direction on this topology.");
  }

  const auto [src, dst] = denied.front();
  INFO("Staged direction: " << src << " -> " << dst);
  hip::contract::ContractCleanup cleanup;
  void* src_buf = AllocateOn(src, &cleanup);
  void* dst_buf = AllocateOn(dst, &cleanup);

  std::array<uint8_t, kByteCount> pattern{};
  std::array<uint8_t, kByteCount> result{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(0x5a + i);
  }

  HIP_CHECK(hipSetDevice(src));
  HIP_CHECK(hipMemcpy(src_buf, pattern.data(), kByteCount, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dst_buf, src_buf, kByteCount, hipMemcpyDeviceToDevice));

  HIP_CHECK(hipSetDevice(dst));
  HIP_CHECK(hipMemcpy(result.data(), dst_buf, kByteCount, hipMemcpyDeviceToHost));
  REQUIRE(result == pattern);
}
