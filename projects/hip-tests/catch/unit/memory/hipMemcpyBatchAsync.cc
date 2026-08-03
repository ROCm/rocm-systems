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
 * - Verify hipMemcpyBatchAsync with hipMemcpyFlagExtOpSwap
 *   exchanges the contents of a hipHostMalloc host buffer and a hipMalloc
 *   device buffer (H->D direction: host src <-> device dst).
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
 * - Verify hipMemcpyBatchAsync swap in the reverse direction:
 *   device src (hipMalloc) <-> host dst (hipHostMalloc).
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
 * - Verify hipMemcpyBatchAsync with multiple H<->D swap ops in a
 *   single call.
 *
 *   Four independent host<->device pairs are swapped in one call; all must be
 *   correct after synchronization.
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
 * - Parameterized H<->D swap correctness over mismatched src/dst byte offsets and two
 *   transfer sizes. For each (src_off, dst_off) pair, the implementation selects the
 *   largest power-of-two element width (<= 16) that divides both endpoint addresses;
 *   any remainder is handled as a trailing byte tail.
 *
 *   Sizes: 4096 and 131072 above and below SDMA threshold
 *
 *   Each host/device allocation is size+64 bytes.  The full host allocation is
 *   filled with 0xAA and the full device allocation with 0xBB.  A single swap of
 *   `size` bytes is issued from host+src_off <-> device+dst_off.  After sync the
 *   transfer region must be swapped (host holds 0xBB, device holds 0xAA) and all
 *   bytes OUTSIDE the transfer region (the padding) must be UNCHANGED (host
 *   padding still 0xAA, device padding still 0xBB).
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
               std::make_pair(size_t{1}, size_t{0}), std::make_pair(size_t{8}, size_t{8}),
               std::make_pair(size_t{2}, size_t{2}));
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
