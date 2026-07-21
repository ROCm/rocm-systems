/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpyBatchAsync hipMemcpyBatchAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes,
 * size_t count, hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
 * size_t* failIdx, hipStream_t stream __dparm(0))`
 *
 * Perform a batch of 1D copies.
 */

namespace {

constexpr size_t kOneKiB = 1024;
constexpr size_t kSmallCopySize = 4 * kOneKiB;
constexpr size_t kMediumCopySize = 32 * kOneKiB;
constexpr size_t kLargeCopySize = 512 * kOneKiB;
constexpr int kPatternValue = 0x42;

struct BatchConfig {
  size_t copy_count;
  size_t copy_size;
};

enum class PointerPattern {
  kBasePointers,
  kOffsetPointers,
  kUnalignedPointers,
  kBroadcastSource,
};

std::vector<std::pair<int, int>> GetPeerAccessibleDevicePairs() {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const int device_count = HipTest::getDeviceCount();
  std::vector<std::pair<int, int>> peer_pairs;
  for (int src_device = 0; src_device < device_count; ++src_device) {
    for (int dst_device = 0; dst_device < device_count; ++dst_device) {
      if (src_device == dst_device) {
        continue;
      }
      int can_access_peer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
      if (can_access_peer != 0) {
        peer_pairs.emplace_back(src_device, dst_device);
      }
    }
  }

  if (peer_pairs.empty()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }
  return peer_pairs;
}

void EnablePeerAccess(const std::vector<std::pair<int, int>>& peer_pairs) {
  for (const auto& [src_device, dst_device] : peer_pairs) {
    HIP_CHECK(hipSetDevice(src_device));
    hipError_t peer_status = hipDeviceEnablePeerAccess(dst_device, 0);
    if (peer_status != hipSuccess && peer_status != hipErrorPeerAccessAlreadyEnabled) {
      HIP_CHECK(peer_status);
    }
  }
}

void DisablePeerAccess(const std::vector<std::pair<int, int>>& peer_pairs) {
  for (const auto& [src_device, dst_device] : peer_pairs) {
    HIP_CHECK(hipSetDevice(src_device));
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  }
}

std::vector<LinearAllocGuard<int>> AllocateBatchBuffers(LinearAllocs allocation_type,
                                                        const BatchConfig& config,
                                                        size_t extra_bytes = 0) {
  std::vector<LinearAllocGuard<int>> allocations;

  for (size_t i = 0; i < config.copy_count; ++i) {
    allocations.emplace_back(allocation_type, config.copy_size + extra_bytes);
  }

  return allocations;
}

std::vector<void*> MakeBatchPtrs(std::vector<LinearAllocGuard<int>>& allocations,
                                 size_t offset_bytes = 0) {
  std::vector<void*> ptrs;

  for (LinearAllocGuard<int>& allocation : allocations) {
    ptrs.push_back(static_cast<void*>(reinterpret_cast<char*>(allocation.ptr()) + offset_bytes));
  }

  return ptrs;
}

void FillDeviceBuffers(const std::vector<void*>& ptrs, size_t copy_size, int value) {
  const size_t copy_elements = copy_size / sizeof(int);
  std::vector<int> source(copy_elements);

  for (size_t i = 0; i < ptrs.size(); ++i) {
    std::fill(source.begin(), source.end(), value + static_cast<int>(i));
    HIP_CHECK(hipMemcpy(ptrs[i], source.data(), copy_size, hipMemcpyHostToDevice));
  }
}

void FillHostBuffers(std::vector<LinearAllocGuard<int>>& buffers, size_t copy_size,
                     int value = kPatternValue) {
  const size_t copy_elements = copy_size / sizeof(int);

  for (size_t i = 0; i < buffers.size(); ++i) {
    std::fill_n(buffers[i].host_ptr(), copy_elements, value + static_cast<int>(i));
  }
}

void VerifyArrayFromBothEnds(const int* values, size_t copy_elements, int expected,
                             size_t copy_index) {
  for (size_t offset = 0; offset < (copy_elements + 1) / 2; ++offset) {
    const size_t front_index = offset;
    const size_t back_index = copy_elements - 1 - offset;

    INFO("Array failure at copy index " << copy_index << ", element " << front_index);
    REQUIRE(values[front_index] == expected);

    if (front_index == back_index) {
      continue;
    }

    INFO("Array failure at copy index " << copy_index << ", element " << back_index);
    REQUIRE(values[back_index] == expected);
  }
}

void VerifyDeviceBuffers(const std::vector<void*>& ptrs, size_t copy_size,
                         int expected = kPatternValue, bool add_index = true) {
  const size_t copy_elements = copy_size / sizeof(int);
  std::vector<int> result(copy_elements);

  for (size_t i = 0; i < ptrs.size(); ++i) {
    HIP_CHECK(hipMemcpy(result.data(), ptrs[i], copy_size, hipMemcpyDeviceToHost));
    const int value = expected + (add_index ? static_cast<int>(i) : 0);
    VerifyArrayFromBothEnds(result.data(), copy_elements, value, i);
  }
}

void VerifyHostBuffers(std::vector<LinearAllocGuard<int>>& buffers, size_t copy_size,
                       int expected = kPatternValue) {
  const size_t copy_elements = copy_size / sizeof(int);

  for (size_t i = 0; i < buffers.size(); ++i) {
    VerifyArrayFromBothEnds(buffers[i].host_ptr(), copy_elements, expected + static_cast<int>(i),
                            i);
  }
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Verifies API-level negative validation for hipMemcpyBatchAsync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Negative) {
  constexpr size_t kCount = 3;

  StreamGuard stream_guard(Streams::created);
  std::array<LinearAllocGuard<int>, kCount> src_allocs;
  std::array<LinearAllocGuard<int>, kCount> dst_allocs;
  std::array<void*, kCount> src_ptrs{};
  std::array<void*, kCount> dst_ptrs{};
  std::array<size_t, kCount> sizes{};

  for (size_t i = 0; i < kCount; ++i) {
    src_allocs[i] = LinearAllocGuard<int>(LinearAllocs::hipMalloc, kSmallCopySize);
    dst_allocs[i] = LinearAllocGuard<int>(LinearAllocs::hipMalloc, kSmallCopySize);
    src_ptrs[i] = src_allocs[i].ptr();
    dst_ptrs[i] = dst_allocs[i].ptr();
    sizes[i] = kSmallCopySize;
  }

  size_t attrs_idxs[1] = {0};

  SECTION("Null destination array") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(nullptr, src_ptrs.data(), sizes.data(), kCount, nullptr,
                                        attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Null source array") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), nullptr, sizes.data(), kCount, nullptr,
                                        attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Null sizes array") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), nullptr, kCount, nullptr,
                                        attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Zero count") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), 0, nullptr,
                                        attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Null destination element") {
    dst_ptrs[1] = nullptr;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Null source element") {
    src_ptrs[1] = nullptr;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Zero size first copy") {
    sizes[0] = 0;
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, &fail_idx, stream_guard.stream()),
                    hipErrorInvalidValue);
    REQUIRE(fail_idx == 0);
  }

  SECTION("Zero size middle copy") {
    sizes[1] = 0;
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, &fail_idx, stream_guard.stream()),
                    hipErrorInvalidValue);
    REQUIRE(fail_idx == 1);
  }

  SECTION("Zero size last copy") {
    sizes[2] = 0;
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, &fail_idx, stream_guard.stream()),
                    hipErrorInvalidValue);
    REQUIRE(fail_idx == 2);
  }

  SECTION("Null fail index on zero size") {
    sizes[1] = 0;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Source range exceeds allocation") {
    src_ptrs[1] = static_cast<void*>(src_allocs[1].ptr() + 1);
    sizes[1] = kSmallCopySize;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Destination range exceeds allocation") {
    dst_ptrs[1] = static_cast<void*>(dst_allocs[1].ptr() + 1);
    sizes[1] = kSmallCopySize;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        nullptr, attrs_idxs, 0, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies attribute array validation for hipMemcpyBatchAsync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Attrs_Negative) {
  constexpr size_t kCount = 2;
  BatchConfig config{kCount, kSmallCopySize};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);
  std::array<hipMemcpyAttributes, 3> attrs{
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
  };
  std::array<size_t, 3> attrs_idxs{0, 1, 2};

  SECTION("Null attrs with nonzero numAttrs") {
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, nullptr,
                            attrs_idxs.data(), 1, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Null attrsIdxs with nonzero numAttrs") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                        attrs.data(), nullptr, 1, nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("First attrsIdxs entry is not zero") {
    attrs_idxs[0] = 1;
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, attrs.data(),
                            attrs_idxs.data(), 1, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("numAttrs exceeds count") {
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, attrs.data(),
                            attrs_idxs.data(), kCount + 1, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("attrsIdxs is not monotonic") {
    attrs_idxs[1] = 0;
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, attrs.data(),
                            attrs_idxs.data(), 2, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("attrsIdxs entry is out of range") {
    attrs_idxs[1] = kCount;
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, attrs.data(),
                            attrs_idxs.data(), 2, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Invalid source access order") {
    attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderInvalid;
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, attrs.data(),
                            attrs_idxs.data(), 1, nullptr, stream_guard.stream()),
        hipErrorInvalidValue);
  }
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 * - Verifies D2D batch copies with hipMemcpyFlagExtOpSwap.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_Swap) {
  constexpr size_t copy_count = 3;
  constexpr size_t copy_size = kSmallCopySize;
  constexpr int kSwapSrcValue = 23;
  constexpr int kSwapDstValue = 47;
  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(src_ptrs.size(), copy_size);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpSwap};
  size_t attrs_idxs[1] = {0};

  FillDeviceBuffers(src_ptrs, copy_size, kSwapSrcValue);
  FillDeviceBuffers(dst_ptrs, copy_size, kSwapDstValue);

  hipError_t status =
      hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), src_ptrs.size(), &attr,
                          attrs_idxs, 1, nullptr, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    SUCCEED("hipMemcpyFlagExtOpSwap is not supported on this device");
  } else {
    HIP_CHECK(status);
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    VerifyDeviceBuffers(src_ptrs, copy_size, kSwapDstValue);
    VerifyDeviceBuffers(dst_ptrs, copy_size, kSwapSrcValue);
  }
}
#endif

/**
 * Test Description
 * ------------------------
 * - Verifies D2D batch copies across generated copy sizes, counts, pointer
 * patterns, and copy flags.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_Functional) {
  const size_t copy_count = GENERATE(1, 8);
  const size_t copy_size = GENERATE(kSmallCopySize, kMediumCopySize, kLargeCopySize);
  const PointerPattern pointer_pattern =
      GENERATE(PointerPattern::kBasePointers, PointerPattern::kOffsetPointers,
               PointerPattern::kUnalignedPointers, PointerPattern::kBroadcastSource);
#if HT_AMD
  const hipMemcpyFlags flag = GENERATE(hipMemcpyFlagDefault, hipMemcpyFlagExtPreferCE);
#else
  const hipMemcpyFlags flag = hipMemcpyFlagDefault;
#endif
  const size_t offset_bytes = pointer_pattern == PointerPattern::kOffsetPointers      ? sizeof(int)
                              : pointer_pattern == PointerPattern::kUnalignedPointers ? 1
                                                                                      : 0;

  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src =
      AllocateBatchBuffers(LinearAllocs::hipMalloc, config, offset_bytes);
  std::vector<LinearAllocGuard<int>> dst =
      AllocateBatchBuffers(LinearAllocs::hipMalloc, config, offset_bytes);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src, offset_bytes);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst, offset_bytes);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);
  size_t attrs_idxs[1] = {0};
  hipMemcpyAttributes attr{
      hipMemcpySrcAccessOrderAny, {}, {}, static_cast<unsigned int>(flag)};

  if (pointer_pattern == PointerPattern::kBroadcastSource) {
    FillDeviceBuffers(src_ptrs, copy_size, kPatternValue);
    void* broadcast_src = src_ptrs.front();
    std::fill(src_ptrs.begin(), src_ptrs.end(), broadcast_src);
  } else {
    FillDeviceBuffers(src_ptrs, copy_size, kPatternValue);
  }

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, &attr,
                                attrs_idxs, 1, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  if (pointer_pattern == PointerPattern::kBroadcastSource) {
    VerifyDeviceBuffers(dst_ptrs, copy_size, kPatternValue, false);
  } else {
    VerifyDeviceBuffers(dst_ptrs, copy_size);
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies H2D batch copies across generated host source allocation types,
 * copy counts, and copy sizes.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_H2D_Functional) {
  const LinearAllocs host_alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const size_t copy_count = GENERATE(1, 8);
  const size_t copy_size = GENERATE(kSmallCopySize, kMediumCopySize, kLargeCopySize);

  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(host_alloc_type, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);

  FillHostBuffers(src, copy_size);

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, nullptr,
                                nullptr, 0, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyDeviceBuffers(dst_ptrs, copy_size);
}

/**
 * Test Description
 * ------------------------
 * - Verifies that pageable H2D source access is complete before
 * hipMemcpyBatchAsync returns when srcAccessOrder is DuringApiCall.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_H2D_Pageable_DuringApiCall_SourceAccess) {
  constexpr size_t copy_count = 8;
  constexpr size_t copy_size = kLargeCopySize;
  constexpr int kOriginalValue = 17;
  constexpr int kAlteredValue = 23;
  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::malloc, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);
  size_t attrs_idxs[1] = {0};
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderDuringApiCall, {}, {}, hipMemcpyFlagDefault};

  FillHostBuffers(src, copy_size, kOriginalValue);

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, &attr,
                                attrs_idxs, 1, nullptr, stream_guard.stream()));
  FillHostBuffers(src, copy_size, kAlteredValue);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDeviceBuffers(dst_ptrs, copy_size, kOriginalValue);
  VerifyHostBuffers(src, copy_size, kAlteredValue);
}

/**
 * Test Description
 * ------------------------
 * - Verifies that pageable H2D source access observes previous same-stream
 * writes to the source when srcAccessOrder is Stream.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_H2D_Pageable_Stream_SourceAccess) {
  constexpr size_t copy_count = 1;
  size_t copy_size = GENERATE(kSmallCopySize, kLargeCopySize);
  constexpr int kStreamProducedValue = 47;
  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> producer =
      AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::malloc, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);
  size_t attrs_idxs[1] = {0};
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault};

  FillDeviceBuffers(MakeBatchPtrs(producer), copy_size, kStreamProducedValue);

  for (size_t i = 0; i < copy_count; ++i) {
    HIP_CHECK(hipMemcpyAsync(src_ptrs[i], producer[i].ptr(), copy_size, hipMemcpyDeviceToHost,
                             stream_guard.stream()));
  }
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, &attr,
                                attrs_idxs, 1, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyDeviceBuffers(dst_ptrs, copy_size, kStreamProducedValue);
}

/**
 * Test Description
 * ------------------------
 * - Verifies D2H batch copies across generated host destination allocation
 * types, copy counts, and copy sizes.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_D2H_Functional) {
  const LinearAllocs host_alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const size_t copy_count = GENERATE(1, 8);
  const size_t copy_size = GENERATE(kSmallCopySize, kMediumCopySize, kLargeCopySize);

  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(host_alloc_type, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);

  FillDeviceBuffers(src_ptrs, copy_size, kPatternValue);

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, nullptr,
                                nullptr, 0, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyHostBuffers(dst, copy_size);
}

/**
 * Test Description
 * ------------------------
 * - Verifies H2H batch copies across generated host allocation types, copy
 * counts, and copy sizes.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_H2H_Functional) {
  const LinearAllocs host_alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const size_t copy_count = GENERATE(1, 8);
  const size_t copy_size = kSmallCopySize;

  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(host_alloc_type, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(host_alloc_type, config);
  std::vector<void*> src_ptrs = MakeBatchPtrs(src);
  std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
  std::vector<size_t> sizes(config.copy_count, config.copy_size);

  FillHostBuffers(src, copy_size);

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, nullptr,
                                nullptr, 0, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyHostBuffers(dst, copy_size);
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 * - [Group A] Verify hipMemcpyBatchAsync with hipMemcpyFlagExtOpSwap
 *   exchanges the contents of a hipHostMalloc host buffer and a hipMalloc
 *   device buffer (H->D direction: host src <-> device dst).
 *
 *   Host src is allocated with hipHostMalloc so it resolves as an amd::Memory
 *   object of Host type.  A plain malloc pointer would yield nullptr from the
 *   memory lookup and be rejected at the read/write-only check
 *   (hip_memory.cpp:3069-3073).
 *
 *   Expected pre-implementation baseline:
 *     PASS  on swap-capable HW (gfx94x / gfx95x / gfx125x) — serviced by
 *           SDMA swap.
 *     SKIP  (kSdmaSwapUnsupported) on all other architectures — the front-end
 *           gate at hip_memory.cpp:3011 returns hipErrorNotSupported before
 *           the shader kernel is wired in.
 *   After Phase 2 this test must pass on every architecture.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_H2D_Functional) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValHost = 42;
  constexpr int kValDev = 99;

  // Host buffer via hipHostMalloc so it is a tracked amd::Memory (Host type).
  void* h_src = nullptr;
  HIP_CHECK(hipHostMalloc(&h_src, kSizeBytes));

  // Device buffer via hipMalloc.
  void* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_dst, kSizeBytes));

  // Fill host side with kValHost, device side with kValDev.
  std::vector<int> hostInit(kNumElements, kValHost);
  std::memcpy(h_src, hostInit.data(), kSizeBytes);

  std::vector<int> devInit(kNumElements, kValDev);
  HIP_CHECK(hipMemcpy(d_dst, devInit.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_dst};
  void* srcs[] = {h_src};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipHostFree(h_src));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // After swap: host side should hold kValDev, device side kValHost.
  std::vector<int> resultDev(kNumElements);
  HIP_CHECK(hipMemcpy(resultDev.data(), d_dst, kSizeBytes, hipMemcpyDeviceToHost));

  const int* resultHost = static_cast<const int*>(h_src);
  for (size_t i = 0; i < kNumElements; i++) {
    INFO("host[" << i << "] = " << resultHost[i] << " (expected " << kValDev << ")");
    REQUIRE(resultHost[i] == kValDev);
    INFO("dev[" << i << "] = " << resultDev[i] << " (expected " << kValHost << ")");
    REQUIRE(resultDev[i] == kValHost);
  }

  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group A] Verify hipMemcpyBatchAsync swap in the reverse direction:
 *   device src (hipMalloc) <-> host dst (hipHostMalloc).
 *
 *   Expected pre-implementation baseline:
 *     PASS  on swap-capable HW (gfx94x / gfx95x / gfx125x).
 *     SKIP  (kSdmaSwapUnsupported) elsewhere.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_D2H_Functional) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValDev = 11;
  constexpr int kValHost = 77;

  // Device buffer.
  void* d_src = nullptr;
  HIP_CHECK(hipMalloc(&d_src, kSizeBytes));

  // Host buffer via hipHostMalloc.
  void* h_dst = nullptr;
  HIP_CHECK(hipHostMalloc(&h_dst, kSizeBytes));

  // Fill device side with kValDev, host side with kValHost.
  std::vector<int> devInit(kNumElements, kValDev);
  HIP_CHECK(hipMemcpy(d_src, devInit.data(), kSizeBytes, hipMemcpyHostToDevice));

  std::vector<int> hostInit(kNumElements, kValHost);
  std::memcpy(h_dst, hostInit.data(), kSizeBytes);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // dst=h_dst (host), src=d_src (device).
  void* dsts[] = {h_dst};
  void* srcs[] = {d_src};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipHostFree(h_dst));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // After swap: device side holds kValHost, host side holds kValDev.
  std::vector<int> resultDev(kNumElements);
  HIP_CHECK(hipMemcpy(resultDev.data(), d_src, kSizeBytes, hipMemcpyDeviceToHost));

  const int* resultHost = static_cast<const int*>(h_dst);
  for (size_t i = 0; i < kNumElements; i++) {
    INFO("dev[" << i << "] = " << resultDev[i] << " (expected " << kValHost << ")");
    REQUIRE(resultDev[i] == kValHost);
    INFO("host[" << i << "] = " << resultHost[i] << " (expected " << kValDev << ")");
    REQUIRE(resultHost[i] == kValDev);
  }

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipHostFree(h_dst));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group A] Verify hipMemcpyBatchAsync with multiple H<->D swap ops in a
 *   single call.  Exercises the multi-entry descriptor path in the batch
 *   emitter (rocblit.cpp:884-897 swapPending/num_entries loop).
 *
 *   Four independent host<->device pairs are swapped in one call; all must be
 *   correct after synchronization.
 *
 *   Expected pre-implementation baseline:
 *     PASS  on swap-capable HW (gfx94x / gfx95x / gfx125x).
 *     SKIP  (kSdmaSwapUnsupported) elsewhere.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_MultiOp) {
  constexpr size_t kNumOps = 4;
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);

  // Each pair has distinct sentinel values to confirm independence.
  constexpr int kHostVals[kNumOps] = {10, 20, 30, 40};
  constexpr int kDevVals[kNumOps] = {55, 66, 77, 88};

  void* h_bufs[kNumOps] = {};
  void* d_bufs[kNumOps] = {};

  for (size_t op = 0; op < kNumOps; ++op) {
    HIP_CHECK(hipHostMalloc(&h_bufs[op], kSizeBytes));
    HIP_CHECK(hipMalloc(&d_bufs[op], kSizeBytes));

    std::vector<int> hi(kNumElements, kHostVals[op]);
    std::memcpy(h_bufs[op], hi.data(), kSizeBytes);

    std::vector<int> di(kNumElements, kDevVals[op]);
    HIP_CHECK(hipMemcpy(d_bufs[op], di.data(), kSizeBytes, hipMemcpyHostToDevice));
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // All ops share the same swap attribute (index 0).
  size_t sizes[kNumOps];
  size_t attrsIdxs[kNumOps];
  for (size_t op = 0; op < kNumOps; ++op) {
    sizes[op] = kSizeBytes;
    attrsIdxs[op] = 0;
  }

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipMemcpyBatchAsync(d_bufs, h_bufs, sizes, kNumOps, &attr,
                                       attrsIdxs, 1, &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    for (size_t op = 0; op < kNumOps; ++op) {
      HIP_CHECK(hipFree(d_bufs[op]));
      HIP_CHECK(hipHostFree(h_bufs[op]));
    }
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify every pair independently.
  for (size_t op = 0; op < kNumOps; ++op) {
    std::vector<int> resultDev(kNumElements);
    HIP_CHECK(hipMemcpy(resultDev.data(), d_bufs[op], kSizeBytes, hipMemcpyDeviceToHost));
    const int* resultHost = static_cast<const int*>(h_bufs[op]);
    for (size_t i = 0; i < kNumElements; i++) {
      INFO("op=" << op << " dev[" << i << "]=" << resultDev[i]
                 << " (expected " << kHostVals[op] << ")");
      REQUIRE(resultDev[i] == kHostVals[op]);
      INFO("op=" << op << " host[" << i << "]=" << resultHost[i]
                 << " (expected " << kDevVals[op] << ")");
      REQUIRE(resultHost[i] == kDevVals[op]);
    }
  }

  for (size_t op = 0; op < kNumOps; ++op) {
    HIP_CHECK(hipFree(d_bufs[op]));
    HIP_CHECK(hipHostFree(h_bufs[op]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group A] Verify hipMemcpyBatchAsync swap with hipMemcpyFlagExtPreferCE
 *   combined with hipMemcpyFlagExtOpSwap.  hipMemcpyFlagExtPreferCE maps to
 *   CopyEnginePreference::SDMA (hip_memory.cpp:2917-2919), which keeps the
 *   swap on the SDMA path even when the new size-based threshold would route
 *   it to the shader kernel.  This test therefore keeps the SDMA swap path
 *   under test after Phase 2 introduces the shader routing fork.
 *
 *   Expected pre-implementation baseline:
 *     PASS  on swap-capable HW (gfx94x / gfx95x / gfx125x) — hipMemcpyFlagExtPreferCE
 *           forces SDMA; SDMA swap already works there.
 *     SKIP  (kSdmaSwapUnsupported) elsewhere.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_PreferCE) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValHost = 13;
  constexpr int kValDev = 37;

  void* h_src = nullptr;
  HIP_CHECK(hipHostMalloc(&h_src, kSizeBytes));

  void* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_dst, kSizeBytes));

  std::vector<int> hostInit(kNumElements, kValHost);
  std::memcpy(h_src, hostInit.data(), kSizeBytes);

  std::vector<int> devInit(kNumElements, kValDev);
  HIP_CHECK(hipMemcpy(d_dst, devInit.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_dst};
  void* srcs[] = {h_src};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  // Combine swap flag with PreferCE to force SDMA copy-engine path.
  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap | hipMemcpyFlagExtPreferCE;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipHostFree(h_src));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // After swap: host holds kValDev, device holds kValHost.
  std::vector<int> resultDev(kNumElements);
  HIP_CHECK(hipMemcpy(resultDev.data(), d_dst, kSizeBytes, hipMemcpyDeviceToHost));

  const int* resultHost = static_cast<const int*>(h_src);
  for (size_t i = 0; i < kNumElements; i++) {
    INFO("host[" << i << "] = " << resultHost[i] << " (expected " << kValDev << ")");
    REQUIRE(resultHost[i] == kValDev);
    INFO("dev[" << i << "] = " << resultDev[i] << " (expected " << kValHost << ")");
    REQUIRE(resultDev[i] == kValHost);
  }

  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group A] Verify that a swap issued asynchronously on a stream is
 *   correctly ordered relative to dependent work enqueued before and after
 *   it on the same stream.
 *
 *   Sequence on stream:
 *     1. hipMemcpy (H2D) to pre-fill device buffer with sentinel value B.
 *     2. hipMemcpyBatchAsync swap (host A <-> device B).
 *     3. hipMemcpy (D2H) to read device buffer into a second host vector.
 *   hipStreamSynchronize is called once at the end; all three operations
 *   must have serialised correctly for the final verification to pass.
 *
 *   This guards the barrier/signal bookkeeping and the attach_signal /
 *   system-scope handling in the batch command submission path.
 *
 *   Expected pre-implementation baseline:
 *     PASS  on swap-capable HW (gfx94x / gfx95x / gfx125x).
 *     SKIP  (kSdmaSwapUnsupported) elsewhere.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_Async_Ordering) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValHost = 111;
  constexpr int kValDev = 222;
  constexpr int kValPre = 7;  // value written to device before the swap

  // Host buffer (hipHostMalloc so it is a tracked amd::Memory).
  void* h_buf = nullptr;
  HIP_CHECK(hipHostMalloc(&h_buf, kSizeBytes));

  // Device buffer.
  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kSizeBytes));

  // Fill host side with kValHost now (before stream work begins).
  std::vector<int> hostInit(kNumElements, kValHost);
  std::memcpy(h_buf, hostInit.data(), kSizeBytes);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Operation 1 (before swap): write kValPre to device via H2D memcpy on stream,
  // then overwrite with kValDev so the swap sees a known value.
  // We use two enqueued memcpys to confirm ordering: first write kValPre, then
  // immediately overwrite with kValDev.  Both are stream-ordered before the swap.
  std::vector<int> preFill(kNumElements, kValPre);
  HIP_CHECK(hipMemcpyAsync(d_buf, preFill.data(), kSizeBytes, hipMemcpyHostToDevice, stream));

  std::vector<int> devFill(kNumElements, kValDev);
  HIP_CHECK(hipMemcpyAsync(d_buf, devFill.data(), kSizeBytes, hipMemcpyHostToDevice, stream));

  // Operation 2: swap on stream — host kValHost <-> device kValDev.
  void* dsts[] = {d_buf};
  void* srcs[] = {h_buf};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipHostFree(h_buf));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);

  // Operation 3 (after swap): read device buffer back to a result vector on stream.
  std::vector<int> resultDev(kNumElements, 0);
  HIP_CHECK(hipMemcpyAsync(resultDev.data(), d_buf, kSizeBytes, hipMemcpyDeviceToHost, stream));

  // Single synchronize covers all three operations.
  HIP_CHECK(hipStreamSynchronize(stream));

  // After correct ordering:
  //   device holds kValHost (written there by the swap).
  //   host   holds kValDev  (written there by the swap).
  //   resultDev reflects the post-swap device contents = kValHost.
  const int* resultHost = static_cast<const int*>(h_buf);
  for (size_t i = 0; i < kNumElements; i++) {
    INFO("resultDev[" << i << "] = " << resultDev[i] << " (expected " << kValHost << ")");
    REQUIRE(resultDev[i] == kValHost);
    INFO("host[" << i << "] = " << resultHost[i] << " (expected " << kValDev << ")");
    REQUIRE(resultHost[i] == kValDev);
  }

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipHostFree(h_buf));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Primary shader-kernel test.  Issues a small H<->D swap
 *   (hipHostMalloc host buffer <-> hipMalloc device buffer) and verifies
 *   both sides exchanged correctly.
 *
 *   On NON-swap-capable HW (sdma_swap_supported_ == false) this is the
 *   genuine proof that the shader kernel ran: after Phase 2, correct output
 *   is only achievable via the shader path because SDMA swap is absent.
 *   The test MUST RUN (not skip) on non-swap-capable architectures — that is
 *   precisely the case it protects.
 *
 *   On swap-capable HW (gfx94x / gfx95x / gfx125x) both the shader path and
 *   the SDMA path produce identical correct data, so the test is
 *   correctness-only there (routing is unobservable in-process).
 *
 *   NO env manipulation: GPU_FORCE_BLIT_SWAP_SIZE is read once at device init
 *   and the Catch binary shares one device, so setenv mid-test has no effect.
 *   Non-swap-capable HW takes the shader path automatically — no flag needed.
 *
 *   Expected pre-implementation state (unmodified runtime):
 *     On non-swap-capable HW: hipMemcpyBatchAsync returns hipErrorNotSupported
 *     (front-end gate at hip_memory.cpp:3011 not yet relaxed) → this test
 *     FAILS pre-implementation and PASSES after Phase 2.
 *     On swap-capable HW: PASSES (SDMA swap already works; routing not forced).
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_ForcedShader) {
  // Small size — well below the default 16 KB sdmaSwapThreshold_.
  // On swap-capable HW this may or may not go to the shader (unobservable);
  // on non-swap-capable HW the shader is the only possible path.
  constexpr size_t kNumElements = 512;  // 2 KB
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValHost = 0xAB;
  constexpr int kValDev = 0xCD;

  void* h_buf = nullptr;
  HIP_CHECK(hipHostMalloc(&h_buf, kSizeBytes));

  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kSizeBytes));

  // Fill host with kValHost, device with kValDev.
  std::vector<int> hostInit(kNumElements, kValHost);
  std::memcpy(h_buf, hostInit.data(), kSizeBytes);

  std::vector<int> devInit(kNumElements, kValDev);
  HIP_CHECK(hipMemcpy(d_buf, devInit.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_buf};
  void* srcs[] = {h_buf};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  // Do NOT skip on hipErrorNotSupported — that failure IS the pre-implementation
  // state on non-swap-capable HW and must be visible as a test failure.
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1,
                                &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // After swap: host holds kValDev, device holds kValHost.
  std::vector<int> resultDev(kNumElements);
  HIP_CHECK(hipMemcpy(resultDev.data(), d_buf, kSizeBytes, hipMemcpyDeviceToHost));

  const int* resultHost = static_cast<const int*>(h_buf);
  for (size_t i = 0; i < kNumElements; i++) {
    INFO("host[" << i << "] = " << resultHost[i] << " (expected " << kValDev << ")");
    REQUIRE(resultHost[i] == kValDev);
    INFO("dev[" << i << "] = " << resultDev[i] << " (expected " << kValHost << ")");
    REQUIRE(resultDev[i] == kValHost);
  }

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipHostFree(h_buf));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Threshold-straddling correctness test.  Issues two H<->D swap
 *   ops in a single hipMemcpyBatchAsync call: one below the default 16 KB
 *   sdmaSwapThreshold_ (8 KB → shader path on swap-capable HW after Phase 2)
 *   and one above it (1 MB → SDMA path on swap-capable HW after Phase 2).
 *   Both ops must produce correct results.
 *
 *   Uses the DEFAULT threshold (GPU_FORCE_BLIT_SWAP_SIZE = 16 KB).  No env
 *   manipulation is performed — GPU_FORCE_BLIT_SWAP_SIZE is read once at
 *   device init and setenv inside the test has no effect in the shared process.
 *
 *   IMPORTANT: This is a CORRECTNESS check, NOT a routing-fork assertion.
 *   On swap-capable HW routing is unobservable in-process (both shader and
 *   SDMA produce identical correct data).  The threshold straddling ensures
 *   both code paths are exercised post-Phase-2 without being able to assert
 *   which path was taken for each op.
 *
 *   Expected pre-implementation state (unmodified runtime):
 *     On non-swap-capable HW: hipErrorNotSupported from front-end gate →
 *     test FAILS pre-implementation, PASSES after Phase 2.
 *     On swap-capable HW: PASSES (both ops routed to SDMA today; after Phase 2
 *     the small op additionally exercises the shader path).
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_ThresholdSelection) {
  // Op 0: 8 KB — below default 16 KB threshold (shader path on swap-capable HW
  //   after Phase 2; on non-swap-capable HW shader path always).
  // Op 1: 1 MB — above default 16 KB threshold (SDMA path on swap-capable HW).
  constexpr size_t kSmallElems = 8 * 1024 / sizeof(int);   // 8 KB
  constexpr size_t kLargeElems = 1024 * 1024 / sizeof(int);  // 1 MB
  constexpr size_t kSmallBytes = kSmallElems * sizeof(int);
  constexpr size_t kLargeBytes = kLargeElems * sizeof(int);

  constexpr int kSmallValHost = 0x11;
  constexpr int kSmallValDev  = 0x22;
  constexpr int kLargeValHost = 0x33;
  constexpr int kLargeValDev  = 0x44;

  void* h_small = nullptr;
  void* h_large = nullptr;
  HIP_CHECK(hipHostMalloc(&h_small, kSmallBytes));
  HIP_CHECK(hipHostMalloc(&h_large, kLargeBytes));

  void* d_small = nullptr;
  void* d_large = nullptr;
  HIP_CHECK(hipMalloc(&d_small, kSmallBytes));
  HIP_CHECK(hipMalloc(&d_large, kLargeBytes));

  // Initialise buffers.
  std::vector<int> hs(kSmallElems, kSmallValHost);
  std::memcpy(h_small, hs.data(), kSmallBytes);

  std::vector<int> hl(kLargeElems, kLargeValHost);
  std::memcpy(h_large, hl.data(), kLargeBytes);

  std::vector<int> ds(kSmallElems, kSmallValDev);
  HIP_CHECK(hipMemcpy(d_small, ds.data(), kSmallBytes, hipMemcpyHostToDevice));

  std::vector<int> dl(kLargeElems, kLargeValDev);
  HIP_CHECK(hipMemcpy(d_large, dl.data(), kLargeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_small, d_large};
  void* srcs[] = {h_small, h_large};
  size_t sizes[] = {kSmallBytes, kLargeBytes};
  size_t attrsIdxs[] = {0, 0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  // On non-swap-capable HW today this returns hipErrorNotSupported → test FAILS
  // pre-implementation (do NOT skip; the failure is the expected baseline state).
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 2, &attr, attrsIdxs, 1,
                                &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify small op.
  {
    std::vector<int> resultDev(kSmallElems);
    HIP_CHECK(hipMemcpy(resultDev.data(), d_small, kSmallBytes, hipMemcpyDeviceToHost));
    const int* resultHost = static_cast<const int*>(h_small);
    for (size_t i = 0; i < kSmallElems; i++) {
      INFO("small host[" << i << "] = " << resultHost[i]
                         << " (expected " << kSmallValDev << ")");
      REQUIRE(resultHost[i] == kSmallValDev);
      INFO("small dev[" << i << "] = " << resultDev[i]
                        << " (expected " << kSmallValHost << ")");
      REQUIRE(resultDev[i] == kSmallValHost);
    }
  }

  // Verify large op.
  {
    std::vector<int> resultDev(kLargeElems);
    HIP_CHECK(hipMemcpy(resultDev.data(), d_large, kLargeBytes, hipMemcpyDeviceToHost));
    const int* resultHost = static_cast<const int*>(h_large);
    for (size_t i = 0; i < kLargeElems; i++) {
      INFO("large host[" << i << "] = " << resultHost[i]
                         << " (expected " << kLargeValDev << ")");
      REQUIRE(resultHost[i] == kLargeValDev);
      INFO("large dev[" << i << "] = " << resultDev[i]
                        << " (expected " << kLargeValHost << ")");
      REQUIRE(resultDev[i] == kLargeValHost);
    }
  }

  HIP_CHECK(hipFree(d_small));
  HIP_CHECK(hipFree(d_large));
  HIP_CHECK(hipHostFree(h_small));
  HIP_CHECK(hipHostFree(h_large));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Swap-kernel branch coverage via size/alignment variants.
 *   Three swap ops in a single batch, each targeting a different branch of
 *   the swap kernel:
 *
 *   Op 0 — ulong2 16-byte aligned fast path: size is a multiple of 16 bytes
 *     and base pointers are 16-byte aligned (guaranteed by hipHostMalloc /
 *     hipMalloc).  The kernel body uses ulong2 loads/stores throughout.
 *
 *   Op 1 — uint fallback path: transfer region is shifted by a 4-byte-aligned
 *     (but not 16-byte-aligned) offset into a hipHostMalloc / hipMalloc buffer,
 *     making the effective base address unaligned to ulong2.  The kernel must
 *     fall back to uint (4-byte) granularity for the body.
 *
 *   Op 2 — non-zero trailing-byte tail: size is deliberately chosen so it is
 *     NOT a multiple of the kernel body granularity (e.g. 1025 bytes),
 *     forcing the trailing-byte tail path that swaps remaining bytes one at a
 *     time in work-group 0 / work-item 0.
 *
 *   Also directly exercises the "both endpoints get system scope" concern
 *   (ShaderSwapBufferBatch must set needs_system_scope symmetrically for src
 *   and dst — see design doc item 5).
 *
 *   NO env manipulation.  No routing-fork assertion (routing unobservable).
 *
 *   Coverage note: this test only exercises 0-offset and 4-byte-aligned
 *   offsets.  Sub-word (1/2/3-byte) offsets and mismatched src/dst alignments
 *   are covered by Unit_hipMemcpyBatchAsync_Swap_UnalignedMatrix, and large
 *   (> threshold) unaligned swaps are covered by
 *   Unit_hipMemcpyBatchAsync_Swap_LargeUnalignedRouting.
 *
 *   Expected state:
 *     On swap-capable HW its small ops (<= sdmaSwapThreshold_) route to the
 *     shader path and pass.  On non-swap-capable HW hipMemcpyBatchAsync returns
 *     hipErrorNotSupported and the test fails (do NOT skip).
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_AlignmentCoverage) {
  // Op 0: fully aligned, multiple of 16 bytes (ulong2 fast path).
  constexpr size_t kAlignedBytes = 4096;  // 256 ulong2 elements
  // Op 1: we allocate extra space and hand an offset pointer to the batch so the
  // effective address exercises the uint (4-byte) fallback path: 4-byte aligned
  // but NOT 16-byte (ulong2) aligned. Sub-word offsets (e.g. +1) are deliberately
  // NOT tested — the kernel's uint fallback casts the base to uint* and indexes,
  // which requires >=4-byte alignment (and SDMA has the same limitation).
  constexpr size_t kUnalignedOffset = 4;           // 4-byte aligned, not 16-byte aligned
  constexpr size_t kUnalignedTransferBytes = 256;  // transfer length (no alignment req.)
  constexpr size_t kUnalignedAllocBytes = kUnalignedTransferBytes + 16;  // padding for offset
  // Op 2: non-multiple-of-body-granularity (trailing bytes forced).
  constexpr size_t kTailBytes = 1025;  // 1024 + 1 trailing byte

  constexpr int kVal0Host = 0x01010101;
  constexpr int kVal0Dev  = 0x02020202;
  constexpr uint8_t kVal1Host = 0xAA;
  constexpr uint8_t kVal1Dev  = 0xBB;
  constexpr uint8_t kVal2Host = 0xCC;
  constexpr uint8_t kVal2Dev  = 0xDD;

  // Op 0 — aligned buffers.
  void* h0 = nullptr;
  void* d0 = nullptr;
  HIP_CHECK(hipHostMalloc(&h0, kAlignedBytes));
  HIP_CHECK(hipMalloc(&d0, kAlignedBytes));
  std::memset(h0, kVal0Host & 0xFF, kAlignedBytes);
  HIP_CHECK(hipMemset(d0, kVal0Dev & 0xFF, kAlignedBytes));

  // Op 1 — allocate with padding; expose pointer offset by kUnalignedOffset.
  void* h1_base = nullptr;
  void* d1_base = nullptr;
  HIP_CHECK(hipHostMalloc(&h1_base, kUnalignedAllocBytes));
  HIP_CHECK(hipMalloc(&d1_base, kUnalignedAllocBytes));
  // Fill the full allocation, then expose the offset region.
  std::memset(h1_base, kVal1Host, kUnalignedAllocBytes);
  HIP_CHECK(hipMemset(d1_base, kVal1Dev, kUnalignedAllocBytes));
  void* h1 = static_cast<uint8_t*>(h1_base) + kUnalignedOffset;
  void* d1 = static_cast<uint8_t*>(d1_base) + kUnalignedOffset;

  // Op 2 — trailing-byte tail (odd size, naturally aligned base).
  void* h2 = nullptr;
  void* d2 = nullptr;
  HIP_CHECK(hipHostMalloc(&h2, kTailBytes));
  HIP_CHECK(hipMalloc(&d2, kTailBytes));
  std::memset(h2, kVal2Host, kTailBytes);
  HIP_CHECK(hipMemset(d2, kVal2Dev, kTailBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d0, d1, d2};
  void* srcs[] = {h0, h1, h2};
  size_t sizes[] = {kAlignedBytes, kUnalignedTransferBytes, kTailBytes};
  size_t attrsIdxs[] = {0, 0, 0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  // On non-swap-capable HW today returns hipErrorNotSupported → test FAILS
  // pre-implementation (do NOT skip).
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 3, &attr, attrsIdxs, 1,
                                &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify op 0 (aligned).
  {
    std::vector<uint8_t> resultDev(kAlignedBytes);
    HIP_CHECK(hipMemcpy(resultDev.data(), d0, kAlignedBytes, hipMemcpyDeviceToHost));
    const uint8_t* resultHost = static_cast<const uint8_t*>(h0);
    for (size_t i = 0; i < kAlignedBytes; i++) {
      INFO("aligned host[" << i << "] = " << static_cast<int>(resultHost[i])
                           << " (expected " << (kVal0Dev & 0xFF) << ")");
      REQUIRE(resultHost[i] == (kVal0Dev & 0xFF));
      INFO("aligned dev[" << i << "] = " << static_cast<int>(resultDev[i])
                          << " (expected " << (kVal0Host & 0xFF) << ")");
      REQUIRE(resultDev[i] == (kVal0Host & 0xFF));
    }
  }

  // Verify op 1 (unaligned offset).
  {
    // Only check the transferred region; the leading bytes (before the offset)
    // are outside the transfer window and are allowed to be any value.
    std::vector<uint8_t> fullDev(kUnalignedAllocBytes);
    HIP_CHECK(hipMemcpy(fullDev.data(), d1_base, kUnalignedAllocBytes, hipMemcpyDeviceToHost));
    const uint8_t* resultDev = fullDev.data() + kUnalignedOffset;   // matches the offset
    const uint8_t* resultHost = static_cast<const uint8_t*>(h1_base) + kUnalignedOffset;
    for (size_t i = 0; i < kUnalignedTransferBytes; i++) {
      INFO("unaligned host[" << i << "] = " << static_cast<int>(resultHost[i])
                             << " (expected " << static_cast<int>(kVal1Dev) << ")");
      REQUIRE(resultHost[i] == kVal1Dev);
      INFO("unaligned dev[" << i << "] = " << static_cast<int>(resultDev[i])
                            << " (expected " << static_cast<int>(kVal1Host) << ")");
      REQUIRE(resultDev[i] == kVal1Host);
    }
  }

  // Verify op 2 (trailing-byte tail).
  {
    std::vector<uint8_t> resultDev(kTailBytes);
    HIP_CHECK(hipMemcpy(resultDev.data(), d2, kTailBytes, hipMemcpyDeviceToHost));
    const uint8_t* resultHost = static_cast<const uint8_t*>(h2);
    for (size_t i = 0; i < kTailBytes; i++) {
      INFO("tail host[" << i << "] = " << static_cast<int>(resultHost[i])
                        << " (expected " << static_cast<int>(kVal2Dev) << ")");
      REQUIRE(resultHost[i] == kVal2Dev);
      INFO("tail dev[" << i << "] = " << static_cast<int>(resultDev[i])
                       << " (expected " << static_cast<int>(kVal2Host) << ")");
      REQUIRE(resultDev[i] == kVal2Host);
    }
  }

  HIP_CHECK(hipFree(d0));
  HIP_CHECK(hipFree(d1_base));
  HIP_CHECK(hipFree(d2));
  HIP_CHECK(hipHostFree(h0));
  HIP_CHECK(hipHostFree(h1_base));
  HIP_CHECK(hipHostFree(h2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Parameterized H<->D swap correctness over mismatched src/dst
 *   byte offsets and two transfer sizes.  For each (src_off, dst_off) pair the
 *   difference d = (dst_addr - src_addr) selects the largest shareable op width
 *   per the design doc's per-copy op-size algorithm, exercising op widths
 *   1/2/4/8/16 plus head/tail handling:
 *
 *     (0,0)   -> 16-byte body, no head/tail
 *     (4,4)   -> 4-byte-aligned, aligned fallback
 *     (0,15)  -> d=15 (odd) -> 1-byte body
 *     (1,15)  -> d=14 -> 2-byte body, head=1
 *     (3,11)  -> d=8  -> 8-byte body, head=5
 *     (3,9)   -> d=6  -> 2-byte body, head=1
 *     (1,0)   -> d=-1 (odd) -> 1-byte body
 *     (8,8)   -> 8-byte-aligned mismatch-free
 *
 *   Sizes: 4096 (<= 16 KB threshold -> shader path) and 131072 (128 KiB,
 *   > threshold -> SDMA path at baseline).
 *
 *   Each host/device allocation is size+64 bytes.  The full host allocation is
 *   filled with 0xAA and the full device allocation with 0xBB.  A single swap of
 *   `size` bytes is issued from host+src_off <-> device+dst_off.  After sync the
 *   transfer region must be swapped (host holds 0xBB, device holds 0xAA) and all
 *   bytes OUTSIDE the transfer region (the padding) must be UNCHANGED (host
 *   padding still 0xAA, device padding still 0xBB).
 *
 *   Tagged [!shouldfail]: on the unmodified runtime the sub-word / mismatched
 *   offsets (shader path) and the large unaligned case (routed to SDMA which
 *   cannot handle non-64B-aligned offsets) produce wrong data, so the overall
 *   test fails.  [!shouldfail] makes Catch2 report the test as passing at this
 *   baseline; a later commit removes the tag once the kernel rewrite makes every
 *   case pass.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_UnalignedMatrix) {
  constexpr uint8_t kHostFill = 0xAA;
  constexpr uint8_t kDevFill = 0xBB;
  constexpr size_t kPad = 64;

  const std::pair<size_t, size_t> off =
      GENERATE(std::make_pair(size_t{0}, size_t{0}), std::make_pair(size_t{4}, size_t{4}),
               std::make_pair(size_t{0}, size_t{15}), std::make_pair(size_t{1}, size_t{15}),
               std::make_pair(size_t{3}, size_t{11}), std::make_pair(size_t{3}, size_t{9}),
               std::make_pair(size_t{1}, size_t{0}), std::make_pair(size_t{8}, size_t{8}));
  const size_t size = GENERATE(size_t{4096}, size_t{131072});
  const size_t src_off = off.first;
  const size_t dst_off = off.second;

  INFO("src_off=" << src_off << " dst_off=" << dst_off << " size=" << size);

  void* h_base = nullptr;
  void* d_base = nullptr;
  HIP_CHECK(hipHostMalloc(&h_base, size + kPad));
  HIP_CHECK(hipMalloc(&d_base, size + kPad));

  std::memset(h_base, kHostFill, size + kPad);
  HIP_CHECK(hipMemset(d_base, kDevFill, size + kPad));

  void* src = static_cast<uint8_t*>(h_base) + src_off;
  void* dst = static_cast<uint8_t*>(d_base) + dst_off;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {dst};
  void* srcs[] = {src};
  size_t sizes[] = {size};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  // Do NOT skip on hipErrorNotSupported: at baseline the large unaligned case
  // routes to SDMA and fails, which is the state [!shouldfail] absorbs.
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy both buffers back in full so padding can be verified too.
  std::vector<uint8_t> devResult(size + kPad);
  HIP_CHECK(hipMemcpy(devResult.data(), d_base, size + kPad, hipMemcpyDeviceToHost));
  const uint8_t* hostResult = static_cast<const uint8_t*>(h_base);

  for (size_t i = 0; i < size + kPad; i++) {
    const bool in_host_region = (i >= src_off) && (i < src_off + size);
    const uint8_t expected_host = in_host_region ? kDevFill : kHostFill;
    INFO("host byte " << i << " (region=" << in_host_region << ") = "
                      << static_cast<int>(hostResult[i]) << " expected "
                      << static_cast<int>(expected_host));
    REQUIRE(hostResult[i] == expected_host);

    const bool in_dev_region = (i >= dst_off) && (i < dst_off + size);
    const uint8_t expected_dev = in_dev_region ? kHostFill : kDevFill;
    INFO("dev byte " << i << " (region=" << in_dev_region << ") = "
                     << static_cast<int>(devResult[i]) << " expected "
                     << static_cast<int>(expected_dev));
    REQUIRE(devResult[i] == expected_dev);
  }

  HIP_CHECK(hipFree(d_base));
  HIP_CHECK(hipHostFree(h_base));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Imbalanced-batch swap correctness routed through the SHADER path.
 *   A single hipMemcpyBatchAsync call with 8 H<->D swap ops of alternating
 *   sizes: large = 8192 bytes at even indices (0,2,4,6) and small = 512 bytes
 *   at odd indices (1,3,5,7).  Every size is <= the default 16 KB
 *   sdmaSwapThreshold_ so all ops take the shader path.  All pointers are
 *   16-byte aligned (base pointers from hipHostMalloc / hipMalloc, offset 0).
 *
 *   Each host op is filled with a distinct pattern (0x10 + index) and each
 *   device op with another distinct pattern (0x80 + index).  After sync every
 *   op's data must have swapped correctly on both sides.
 *
 *   This exercises the unified imbalanced dispatch (mixed body-op counts in one
 *   batch).  It should PASS at baseline and after the rewrite (correctness is
 *   identical; only the thread-to-work distribution changes) — NOT tagged.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_ImbalancedBatch) {
  constexpr size_t kNumOps = 8;
  constexpr size_t kLargeBytes = 8192;
  constexpr size_t kSmallBytes = 512;

  auto op_size = [](size_t i) -> size_t { return (i % 2 == 0) ? kLargeBytes : kSmallBytes; };

  std::vector<void*> h_bufs(kNumOps, nullptr);
  std::vector<void*> d_bufs(kNumOps, nullptr);
  std::vector<size_t> sizes(kNumOps);
  std::vector<uint8_t> host_pattern(kNumOps);
  std::vector<uint8_t> dev_pattern(kNumOps);

  for (size_t i = 0; i < kNumOps; i++) {
    sizes[i] = op_size(i);
    host_pattern[i] = static_cast<uint8_t>(0x10 + i);
    dev_pattern[i] = static_cast<uint8_t>(0x80 + i);
    HIP_CHECK(hipHostMalloc(&h_bufs[i], sizes[i]));
    HIP_CHECK(hipMalloc(&d_bufs[i], sizes[i]));
    std::memset(h_bufs[i], host_pattern[i], sizes[i]);
    HIP_CHECK(hipMemset(d_bufs[i], dev_pattern[i], sizes[i]));
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<void*> dsts(d_bufs);
  std::vector<void*> srcs(h_bufs);
  std::vector<size_t> attrsIdxs(kNumOps, 0);

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), kNumOps, &attr,
                                attrsIdxs.data(), 1, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < kNumOps; i++) {
    std::vector<uint8_t> devResult(sizes[i]);
    HIP_CHECK(hipMemcpy(devResult.data(), d_bufs[i], sizes[i], hipMemcpyDeviceToHost));
    const uint8_t* hostResult = static_cast<const uint8_t*>(h_bufs[i]);
    for (size_t b = 0; b < sizes[i]; b++) {
      INFO("op " << i << " host byte " << b << " = " << static_cast<int>(hostResult[b])
                 << " expected " << static_cast<int>(dev_pattern[i]));
      REQUIRE(hostResult[b] == dev_pattern[i]);
      INFO("op " << i << " dev byte " << b << " = " << static_cast<int>(devResult[b])
                 << " expected " << static_cast<int>(host_pattern[i]));
      REQUIRE(devResult[b] == host_pattern[i]);
    }
  }

  for (size_t i = 0; i < kNumOps; i++) {
    HIP_CHECK(hipFree(d_bufs[i]));
    HIP_CHECK(hipHostFree(h_bufs[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Focused single-op LARGE unaligned H<->D swap.  size = 262144
 *   (256 KiB, > the default 16 KB sdmaSwapThreshold_), src offset = 4 and dst
 *   offset = 4 (4-byte aligned but NOT 64-byte aligned).  At baseline a
 *   > threshold swap routes to SDMA, which cannot service non-64B-aligned
 *   offsets, so this fails.
 *
 *   Host (hipHostMalloc) and device (hipMalloc) allocations are size+64 bytes.
 *   The full host allocation is filled with 0xAA and the full device allocation
 *   with 0xBB.  After the swap the transfer region must be swapped (host holds
 *   0xBB, device holds 0xAA) and the padding OUTSIDE the transfer region must be
 *   UNCHANGED (host still 0xAA, device still 0xBB).
 *
 *   Tagged [!shouldfail]: the baseline SDMA route fails this case.  The tag is
 *   removed once the routing gate sends non-64B-aligned swaps to the shader.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_LargeUnalignedRouting) {
  constexpr uint8_t kHostFill = 0xAA;
  constexpr uint8_t kDevFill = 0xBB;
  constexpr size_t kPad = 64;
  constexpr size_t kSize = 262144;  // 256 KiB, > threshold
  constexpr size_t kSrcOff = 4;     // 4-byte aligned, not 64-byte aligned
  constexpr size_t kDstOff = 4;

  void* h_base = nullptr;
  void* d_base = nullptr;
  HIP_CHECK(hipHostMalloc(&h_base, kSize + kPad));
  HIP_CHECK(hipMalloc(&d_base, kSize + kPad));

  std::memset(h_base, kHostFill, kSize + kPad);
  HIP_CHECK(hipMemset(d_base, kDevFill, kSize + kPad));

  void* src = static_cast<uint8_t*>(h_base) + kSrcOff;
  void* dst = static_cast<uint8_t*>(d_base) + kDstOff;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {dst};
  void* srcs[] = {src};
  size_t sizes[] = {kSize};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<uint8_t> devResult(kSize + kPad);
  HIP_CHECK(hipMemcpy(devResult.data(), d_base, kSize + kPad, hipMemcpyDeviceToHost));
  const uint8_t* hostResult = static_cast<const uint8_t*>(h_base);

  for (size_t i = 0; i < kSize + kPad; i++) {
    const bool in_host_region = (i >= kSrcOff) && (i < kSrcOff + kSize);
    const uint8_t expected_host = in_host_region ? kDevFill : kHostFill;
    INFO("host byte " << i << " = " << static_cast<int>(hostResult[i]) << " expected "
                      << static_cast<int>(expected_host));
    REQUIRE(hostResult[i] == expected_host);

    const bool in_dev_region = (i >= kDstOff) && (i < kDstOff + kSize);
    const uint8_t expected_dev = in_dev_region ? kHostFill : kDevFill;
    INFO("dev byte " << i << " = " << static_cast<int>(devResult[i]) << " expected "
                     << static_cast<int>(expected_dev));
    REQUIRE(devResult[i] == expected_dev);
  }

  HIP_CHECK(hipFree(d_base));
  HIP_CHECK(hipHostFree(h_base));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - [Group B] Mixed-batch test: a single hipMemcpyBatchAsync call that
 *   contains BOTH linear copy ops AND H<->D swap ops, forcing the three-way
 *   partition introduced in Phase 2:
 *
 *     d2dCopyOps     — shader linear copies  (hipMemcpyFlagExtOpLinear default)
 *     swapShaderOps  — shader swap ops        (hipMemcpyFlagExtOpSwap)
 *     p2pCopyOps     — SDMA path (large swaps + everything else)
 *
 *   Batch layout (4 ops, two attribute entries):
 *     Op 0  attr[0] = linear copy  — D->H, device src → host dst
 *     Op 1  attr[1] = swap         — host src <-> device dst
 *     Op 2  attr[0] = linear copy  — H->D, host src → device dst
 *     Op 3  attr[1] = swap         — device src <-> host dst
 *
 *   Every op uses independent buffers with distinct sentinel values.  After
 *   synchronization:
 *     - Copy ops: dst holds the value that was in src at submission time.
 *     - Swap ops: both sides exchange their initial values.
 *
 *   This exercises the partition + dispatch logic together in one call and
 *   confirms the three-way split does not corrupt or drop any op.
 *
 *   NO env manipulation.  No routing-fork assertion (routing unobservable).
 *
 *   Expected pre-implementation state (unmodified runtime):
 *     On non-swap-capable HW: hipErrorNotSupported from the front-end gate →
 *     test FAILS pre-implementation, PASSES after Phase 2.
 *     On swap-capable HW: The swap ops are serviced by SDMA today and the
 *     copy ops by the shader linear path; the test PASSES (mix already works
 *     for the SDMA case).  After Phase 2 small swaps move to the shader path
 *     but correctness is identical.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap_MixedBatch) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);

  // Sentinel values, one per op × side, all distinct.
  constexpr int kCopy0SrcVal  = 0x10;  // op 0 src (device) → dst (host)
  constexpr int kCopy0DstInit = 0x11;  // op 0 dst initial (overwritten by copy)
  constexpr int kSwap1HostVal = 0x20;  // op 1 host side before swap
  constexpr int kSwap1DevVal  = 0x21;  // op 1 device side before swap
  constexpr int kCopy2SrcVal  = 0x30;  // op 2 src (host) → dst (device)
  constexpr int kCopy2DstInit = 0x31;  // op 2 dst initial (overwritten by copy)
  constexpr int kSwap3HostVal = 0x40;  // op 3 host side before swap
  constexpr int kSwap3DevVal  = 0x41;  // op 3 device side before swap

  // --- Op 0: linear copy, device src → host dst ---
  void* d_copy0_src = nullptr;
  HIP_CHECK(hipMalloc(&d_copy0_src, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kCopy0SrcVal);
    HIP_CHECK(hipMemcpy(d_copy0_src, init.data(), kSizeBytes, hipMemcpyHostToDevice));
  }
  void* h_copy0_dst = nullptr;
  HIP_CHECK(hipHostMalloc(&h_copy0_dst, kSizeBytes));
  std::memset(h_copy0_dst, kCopy0DstInit & 0xFF, kSizeBytes);

  // --- Op 1: swap, host src <-> device dst ---
  void* h_swap1 = nullptr;
  HIP_CHECK(hipHostMalloc(&h_swap1, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kSwap1HostVal);
    std::memcpy(h_swap1, init.data(), kSizeBytes);
  }
  void* d_swap1 = nullptr;
  HIP_CHECK(hipMalloc(&d_swap1, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kSwap1DevVal);
    HIP_CHECK(hipMemcpy(d_swap1, init.data(), kSizeBytes, hipMemcpyHostToDevice));
  }

  // --- Op 2: linear copy, host src → device dst ---
  void* h_copy2_src = nullptr;
  HIP_CHECK(hipHostMalloc(&h_copy2_src, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kCopy2SrcVal);
    std::memcpy(h_copy2_src, init.data(), kSizeBytes);
  }
  void* d_copy2_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_copy2_dst, kSizeBytes));
  HIP_CHECK(hipMemset(d_copy2_dst, kCopy2DstInit & 0xFF, kSizeBytes));

  // --- Op 3: swap, device src <-> host dst ---
  void* d_swap3 = nullptr;
  HIP_CHECK(hipMalloc(&d_swap3, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kSwap3DevVal);
    HIP_CHECK(hipMemcpy(d_swap3, init.data(), kSizeBytes, hipMemcpyHostToDevice));
  }
  void* h_swap3 = nullptr;
  HIP_CHECK(hipHostMalloc(&h_swap3, kSizeBytes));
  {
    std::vector<int> init(kNumElements, kSwap3HostVal);
    std::memcpy(h_swap3, init.data(), kSizeBytes);
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Two attribute entries: [0] = linear copy, [1] = swap.
  hipMemcpyAttributes attrs[2] = {};
  attrs[0].flags = 0;  // linear (default)
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = hipMemcpyFlagExtOpSwap;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;

  // op0=copy(D->H), op1=swap(H<->D), op2=copy(H->D), op3=swap(D<->H).
  void* dsts[] = {h_copy0_dst, d_swap1,     d_copy2_dst, h_swap3};
  void* srcs[] = {d_copy0_src, h_swap1,     h_copy2_src, d_swap3};
  size_t sizes[] = {kSizeBytes, kSizeBytes, kSizeBytes,  kSizeBytes};
  size_t attrsIdxs[] = {0, 1, 0, 1};  // linear, swap, linear, swap

  size_t failIdx = 0;
  // On non-swap-capable HW today returns hipErrorNotSupported → test FAILS
  // pre-implementation (do NOT skip).
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 4, attrs, attrsIdxs, 2,
                                &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify op 0 (copy D->H): h_copy0_dst must hold kCopy0SrcVal.
  {
    const int* result = static_cast<const int*>(h_copy0_dst);
    for (size_t i = 0; i < kNumElements; i++) {
      INFO("copy0 dst[" << i << "] = " << result[i]
                        << " (expected " << kCopy0SrcVal << ")");
      REQUIRE(result[i] == kCopy0SrcVal);
    }
  }

  // Verify op 1 (swap H<->D): host holds kSwap1DevVal, device holds kSwap1HostVal.
  {
    std::vector<int> resultDev(kNumElements);
    HIP_CHECK(hipMemcpy(resultDev.data(), d_swap1, kSizeBytes, hipMemcpyDeviceToHost));
    const int* resultHost = static_cast<const int*>(h_swap1);
    for (size_t i = 0; i < kNumElements; i++) {
      INFO("swap1 host[" << i << "] = " << resultHost[i]
                         << " (expected " << kSwap1DevVal << ")");
      REQUIRE(resultHost[i] == kSwap1DevVal);
      INFO("swap1 dev[" << i << "] = " << resultDev[i]
                        << " (expected " << kSwap1HostVal << ")");
      REQUIRE(resultDev[i] == kSwap1HostVal);
    }
  }

  // Verify op 2 (copy H->D): d_copy2_dst must hold kCopy2SrcVal.
  {
    std::vector<int> result(kNumElements);
    HIP_CHECK(hipMemcpy(result.data(), d_copy2_dst, kSizeBytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kNumElements; i++) {
      INFO("copy2 dst[" << i << "] = " << result[i]
                        << " (expected " << kCopy2SrcVal << ")");
      REQUIRE(result[i] == kCopy2SrcVal);
    }
  }

  // Verify op 3 (swap D<->H): device holds kSwap3HostVal, host holds kSwap3DevVal.
  {
    std::vector<int> resultDev(kNumElements);
    HIP_CHECK(hipMemcpy(resultDev.data(), d_swap3, kSizeBytes, hipMemcpyDeviceToHost));
    const int* resultHost = static_cast<const int*>(h_swap3);
    for (size_t i = 0; i < kNumElements; i++) {
      INFO("swap3 dev[" << i << "] = " << resultDev[i]
                        << " (expected " << kSwap3HostVal << ")");
      REQUIRE(resultDev[i] == kSwap3HostVal);
      INFO("swap3 host[" << i << "] = " << resultHost[i]
                         << " (expected " << kSwap3DevVal << ")");
      REQUIRE(resultHost[i] == kSwap3DevVal);
    }
  }

  HIP_CHECK(hipFree(d_copy0_src));
  HIP_CHECK(hipFree(d_swap1));
  HIP_CHECK(hipFree(d_copy2_dst));
  HIP_CHECK(hipFree(d_swap3));
  HIP_CHECK(hipHostFree(h_copy0_dst));
  HIP_CHECK(hipHostFree(h_swap1));
  HIP_CHECK(hipHostFree(h_copy2_src));
  HIP_CHECK(hipHostFree(h_swap3));
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

/**
 * Test Description
 * ------------------------
 * - Verifies one batch containing H2D, D2D, D2H, and H2H copies.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Mixed_Functional) {
  constexpr size_t copy_size = kMediumCopySize;
  const size_t copy_elements = copy_size / sizeof(int);

  if (HipTest::getDeviceCount() < 3) {
    HIP_SKIP_TEST("Test requires at least three GPUs.");
  }

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);

  LinearAllocGuard<int> h2d_src(LinearAllocs::malloc, copy_size);
  LinearAllocGuard<int> h2d_dst(LinearAllocs::hipMalloc, copy_size);

  HIP_CHECK(hipSetDevice(1));
  LinearAllocGuard<int> d2d_src(LinearAllocs::hipMalloc, copy_size);
  LinearAllocGuard<int> d2d_dst(LinearAllocs::hipMalloc, copy_size);

  HIP_CHECK(hipSetDevice(2));
  LinearAllocGuard<int> d2h_src(LinearAllocs::hipMalloc, copy_size);

  LinearAllocGuard<int> d2h_dst(LinearAllocs::malloc, copy_size);
  LinearAllocGuard<int> h2h_src(LinearAllocs::malloc, copy_size);
  LinearAllocGuard<int> h2h_dst(LinearAllocs::malloc, copy_size);

  HIP_CHECK(hipSetDevice(0));
  std::array<void*, 4> src_ptrs{h2d_src.ptr(), d2d_src.ptr(), d2h_src.ptr(), h2h_src.ptr()};
  std::array<void*, 4> dst_ptrs{h2d_dst.ptr(), d2d_dst.ptr(), d2h_dst.ptr(), h2h_dst.ptr()};
  std::array<size_t, 4> sizes{copy_size, copy_size, copy_size, copy_size};

  std::fill_n(h2d_src.host_ptr(), copy_elements, kPatternValue);
  std::fill_n(h2h_src.host_ptr(), copy_elements, kPatternValue + 3);
  std::vector<int> source(copy_elements);
  std::fill(source.begin(), source.end(), kPatternValue + 1);
  HIP_CHECK(hipMemcpy(d2d_src.ptr(), source.data(), copy_size, hipMemcpyHostToDevice));
  std::fill(source.begin(), source.end(), kPatternValue + 2);
  HIP_CHECK(hipMemcpy(d2h_src.ptr(), source.data(), copy_size, hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), sizes.size(),
                                nullptr, nullptr, 0, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  std::vector<int> result(copy_elements);
  HIP_CHECK(hipMemcpy(result.data(), h2d_dst.ptr(), copy_size, hipMemcpyDeviceToHost));
  VerifyArrayFromBothEnds(result.data(), copy_elements, kPatternValue, 0);
  HIP_CHECK(hipMemcpy(result.data(), d2d_dst.ptr(), copy_size, hipMemcpyDeviceToHost));
  VerifyArrayFromBothEnds(result.data(), copy_elements, kPatternValue + 1, 1);
  VerifyArrayFromBothEnds(d2h_dst.host_ptr(), copy_elements, kPatternValue + 2, 2);
  VerifyArrayFromBothEnds(h2h_dst.host_ptr(), copy_elements, kPatternValue + 3, 3);
}

/**
 * Test Description
 * ------------------------
 * - Verifies default stream behavior, same-stream ordering, and event
 * dependency ordering for hipMemcpyBatchAsync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Stream) {
  constexpr size_t kCount = 2;
  constexpr size_t kCopyElements = kSmallCopySize / sizeof(int);
  BatchConfig config{kCount, kSmallCopySize};
  std::vector<size_t> sizes(config.copy_count, config.copy_size);

  SECTION("Default stream") {
    std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<void*> src_ptrs = MakeBatchPtrs(src);
    std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
    FillDeviceBuffers(src_ptrs, kSmallCopySize, kPatternValue);

    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, nullptr,
                                  nullptr, 0, nullptr, nullptr));
    HIP_CHECK(hipStreamSynchronize(nullptr));

    VerifyDeviceBuffers(dst_ptrs, kSmallCopySize);
  }

  SECTION("Ordering after prior kernel on same stream") {
    StreamGuard stream_guard(Streams::created);
    std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<void*> src_ptrs = MakeBatchPtrs(src);
    std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);

    static_cast<void>(hipGetLastError());
    for (size_t i = 0; i < kCount; ++i) {
      VectorSet<<<1, 256, 0, stream_guard.stream()>>>(
          static_cast<int*>(src_ptrs[i]), kPatternValue + static_cast<int>(i), kCopyElements);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, nullptr,
                                  nullptr, 0, nullptr, stream_guard.stream()));
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

    VerifyDeviceBuffers(dst_ptrs, kSmallCopySize);
  }

  SECTION("Ordering via event dependency") {
    StreamGuard producer_stream(Streams::created);
    StreamGuard copy_stream(Streams::created);
    EventsGuard events(1);
    std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(LinearAllocs::hipMalloc, config);
    std::vector<void*> src_ptrs = MakeBatchPtrs(src);
    std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);

    static_cast<void>(hipGetLastError());
    for (size_t i = 0; i < kCount; ++i) {
      VectorSet<<<1, 256, 0, producer_stream.stream()>>>(
          static_cast<int*>(src_ptrs[i]), kPatternValue + static_cast<int>(i), kCopyElements);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(events[0], producer_stream.stream()));
    HIP_CHECK(hipStreamWaitEvent(copy_stream.stream(), events[0], 0));
    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount, nullptr,
                                  nullptr, 0, nullptr, copy_stream.stream()));
    HIP_CHECK(hipStreamSynchronize(copy_stream.stream()));

    VerifyDeviceBuffers(dst_ptrs, kSmallCopySize);
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies peer-to-peer batches across peer-accessible device pairs.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Functional) {
  const std::vector<std::pair<int, int>> peer_pairs = GetPeerAccessibleDevicePairs();
  const int stream_device = peer_pairs.front().first;
#if HT_AMD
  const hipMemcpyFlags flag = GENERATE(hipMemcpyFlagDefault, hipMemcpyFlagExtPreferCE);
#else
  const hipMemcpyFlags flag = hipMemcpyFlagDefault;
#endif

  constexpr size_t copy_size = kSmallCopySize;
  constexpr size_t batch_count = 2;
  const size_t total_copy_count = peer_pairs.size() * batch_count;
  std::vector<LinearAllocGuard<int>> src_allocations;
  std::vector<LinearAllocGuard<int>> dst_allocations;
  std::vector<void*> src_ptrs;
  std::vector<void*> dst_ptrs;
  std::vector<size_t> sizes(total_copy_count, copy_size);
  hipMemcpyAttributes attr{
      hipMemcpySrcAccessOrderStream, {}, {}, static_cast<unsigned int>(flag)};
  size_t attrs_idxs[1] = {0};

  EnablePeerAccess(peer_pairs);
  for (const auto& [src_device, dst_device] : peer_pairs) {
    for (size_t i = 0; i < batch_count; ++i) {
      HIP_CHECK(hipSetDevice(src_device));
      src_allocations.emplace_back(LinearAllocs::hipMalloc, copy_size);
      src_ptrs.push_back(src_allocations.back().ptr());

      HIP_CHECK(hipSetDevice(dst_device));
      dst_allocations.emplace_back(LinearAllocs::hipMalloc, copy_size);
      dst_ptrs.push_back(dst_allocations.back().ptr());
    }
  }

  HIP_CHECK(hipSetDevice(stream_device));
  StreamGuard stream_guard(Streams::created);
  FillDeviceBuffers(src_ptrs, copy_size, kPatternValue);
  HIP_CHECK(hipSetDevice(stream_device));
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), sizes.size(), &attr,
                                attrs_idxs, 1, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyDeviceBuffers(dst_ptrs, copy_size);
  DisablePeerAccess(peer_pairs);
  HIP_CHECK(hipSetDevice(stream_device));
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 * - Verifies H2D hipMemcpyBatchAsync with hipMemcpyFlagExtOpIndirectSrc copies
 * from the host buffer referenced by a pinned pointer slot.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_IndirectSrc) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipHostMalloc, copy_size);
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, copy_size);
  LinearAllocGuard<char> src_slot(LinearAllocs::hipHostMalloc, sizeof(void*));

  const size_t copy_elements = copy_size / sizeof(int);
  std::fill_n(src.host_ptr(), copy_elements, kPatternValue);

  void* src_ptr = src.ptr();
  std::memcpy(src_slot.ptr(), &src_ptr, sizeof(void*));

  std::vector<void*> dst_ptrs{dst.ptr()};
  std::vector<void*> src_ptrs{src_slot.ptr()};
  std::vector<size_t> sizes{copy_size};
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpIndirectSrc};
  size_t attrs_idx = 0;

  hipError_t status = hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), 1, &attr,
                                          &attrs_idx, 1, nullptr, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    SUCCEED("hipMemcpyFlagExtOpIndirectSrc is not supported on this device");
  } else {
    HIP_CHECK(status);
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    VerifyDeviceBuffers(dst_ptrs, copy_size);
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies D2H hipMemcpyBatchAsync with hipMemcpyFlagExtOpIndirectDst copies
 * into the host buffer referenced by a pinned pointer slot.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_IndirectDst) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, copy_size);
  LinearAllocGuard<int> dst(LinearAllocs::hipHostMalloc, copy_size);
  LinearAllocGuard<char> dst_slot(LinearAllocs::hipHostMalloc, sizeof(void*));

  const size_t copy_elements = copy_size / sizeof(int);
  std::vector<int> host_pattern(copy_elements, kPatternValue);
  HIP_CHECK(hipMemcpy(src.ptr(), host_pattern.data(), copy_size, hipMemcpyHostToDevice));
  std::fill_n(dst.host_ptr(), copy_elements, 0);

  void* dst_ptr = dst.ptr();
  std::memcpy(dst_slot.ptr(), &dst_ptr, sizeof(void*));

  std::vector<void*> src_ptrs{src.ptr()};
  std::vector<void*> dst_ptrs{dst_slot.ptr()};
  std::vector<size_t> sizes{copy_size};
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpIndirectDst};
  size_t attrs_idx = 0;

  hipError_t status = hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), 1, &attr,
                                          &attrs_idx, 1, nullptr, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    SUCCEED("hipMemcpyFlagExtOpIndirectDst is not supported on this device");
  } else {
    HIP_CHECK(status);
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    VerifyArrayFromBothEnds(dst.host_ptr(), copy_elements, kPatternValue, 0);
  }
}
#endif

/**
 * End doxygen group MemoryTest.
 * @}
 */
