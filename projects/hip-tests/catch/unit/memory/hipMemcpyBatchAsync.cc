/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>
#include <utils.hh>
#include "memcpyBatchAsync_common.hh"

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
 * For each batch entry, the contents of two buffers are exchanged using
 * hipMemcpyFlagExtOpSwap across generated per-side allocation types, copy counts, and sizes.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap) {
  const size_t count = GENERATE(2, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  const LinearAllocs allocTypeA =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const LinearAllocs allocTypeB =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  CAPTURE(count, size_in_bytes, allocTypeA, allocTypeB);

  const hipError_t expectedError = getSwapExpectedReturn(allocTypeA, allocTypeB);

  std::vector<std::vector<unsigned char>> initialValuesA(
      count, std::vector<unsigned char>(size_in_bytes, 10));
  std::vector<std::vector<unsigned char>> initialValuesB(
      count, std::vector<unsigned char>(size_in_bytes, 4));
  std::vector<void*> swapPtrsA(count);
  std::vector<void*> swapPtrsB(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);
  for (size_t i = 0; i < count; ++i) {
    LinearAllocGuard<unsigned char> allocB(allocTypeB, size_in_bytes);
    swapPtrsB[i] = allocB.ptr();
    allocations.push_back(std::move(allocB));
    fillBuffer(swapPtrsB[i], initialValuesB[i], allocTypeB);

    LinearAllocGuard<unsigned char> allocA(allocTypeA, size_in_bytes);
    swapPtrsA[i] = allocA.ptr();
    allocations.push_back(std::move(allocA));
    fillBuffer(swapPtrsA[i], initialValuesA[i], allocTypeA);
  }

  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpSwap};
  size_t attrs_idxs[1] = {0};
  size_t fail_index = 0;

  HIP_CHECK_ERROR(hipMemcpyBatchAsync(swapPtrsA.data(), swapPtrsB.data(), sizes.data(), count,
                                      &attr, attrs_idxs, 1, &fail_index, stream_guard.stream()),
                  expectedError);

  // Unsupported allocation/device combinations are asserted to fail above; only the supported
  // combinations reach a real exchange worth verifying.
  if (expectedError == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(swapPtrsA[i], initialValuesB[i], allocTypeA);
      requireBufferEquals(swapPtrsB[i], initialValuesA[i], allocTypeB);
    }
  }
}

// Batched multicast copy: one shared source, multiple destinations.
static void RunMulticastCopyTest(size_t count, size_t size_in_bytes, LinearAllocs srcAllocType,
                                 LinearAllocs dstAllocType) {
  std::vector<unsigned char> initialValues(size_in_bytes, 10);
  std::vector<void*> srcPtrs(count);
  std::vector<void*> dstPtrs(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<unsigned char> srcAlloc(srcAllocType, size_in_bytes);
  void* srcMem = srcAlloc.ptr();
  fillBuffer(srcMem, initialValues, srcAllocType);
  allocations.push_back(std::move(srcAlloc));

  for (size_t i = 0; i < count; ++i) {
    srcPtrs[i] = srcMem;
    LinearAllocGuard<unsigned char> dstAlloc(dstAllocType, size_in_bytes);
    dstPtrs[i] = dstAlloc.ptr();
    allocations.push_back(std::move(dstAlloc));
  }

  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault};
  size_t attrs_idxs[1] = {0};
  size_t fail_index = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dstPtrs.data(), srcPtrs.data(), sizes.data(), count, &attr,
                                attrs_idxs, 1, &fail_index, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (size_t i = 0; i < count; ++i) {
    requireBufferEquals(dstPtrs[i], initialValues, dstAllocType);
  }
}

/**
 * Batched multicast copy: one shared source, multiple destinations.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Multicast) {
  const size_t count = GENERATE(2, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  const LinearAllocs allocTypeSrc =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const LinearAllocs allocTypeDst =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  CAPTURE(count, size_in_bytes, allocTypeSrc, allocTypeDst);

  RunMulticastCopyTest(count, size_in_bytes, allocTypeSrc, allocTypeDst);
}

/**
 * Batched multicast copy with large per-operation size.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Multicast_Large) {
  const size_t count = GENERATE(2, 3, 8);
  const LinearAllocs allocTypeSrc =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const LinearAllocs allocTypeDst =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const size_t size_in_bytes = 1024 * 1024;
  CAPTURE(count, size_in_bytes, allocTypeSrc, allocTypeDst);

  RunMulticastCopyTest(count, size_in_bytes, allocTypeSrc, allocTypeDst);
}

/**
 * Batch D2D copies where most entries share one source (multicast-friendly) but one entry uses a
 * different source, e.g. srcA, srcA, srcA, srcB, srcA, srcA, srcA. Validates correctness when the
 * batch cannot be lowered to a single multicast operation.
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_MixedMulticastSources) {
  constexpr int k_count = 7;
  const size_t size_in_bytes = 4096;

  HIP_CHECK(hipSetDevice(0));
  std::vector<unsigned char> pattern_a(size_in_bytes, 10);
  std::vector<unsigned char> pattern_b(size_in_bytes, 4);

  LinearAllocGuard<unsigned char> srcAllocA(LinearAllocs::hipMalloc, size_in_bytes);
  LinearAllocGuard<unsigned char> srcAllocB(LinearAllocs::hipMalloc, size_in_bytes);
  void* const srcMemA = srcAllocA.ptr();
  void* const srcMemB = srcAllocB.ptr();

  std::vector<LinearAllocGuard<unsigned char>> allocations;
  allocations.push_back(std::move(srcAllocA));
  allocations.push_back(std::move(srcAllocB));

  fillBuffer(srcMemA, pattern_a, LinearAllocs::hipMalloc);
  fillBuffer(srcMemB, pattern_b, LinearAllocs::hipMalloc);

  std::vector<void*> dst_ptrs;
  for (int i = 0; i < k_count; ++i) {
    LinearAllocGuard<unsigned char> dstAlloc(LinearAllocs::hipMalloc, size_in_bytes);
    HIP_CHECK(hipMemset(dstAlloc.ptr(), 0, size_in_bytes));
    dst_ptrs.push_back(dstAlloc.ptr());
    allocations.push_back(std::move(dstAlloc));
  }

  std::vector<void*> src_ptrs = {srcMemA, srcMemA, srcMemA, srcMemB, srcMemA, srcMemA, srcMemA};
  std::vector<size_t> sizes(k_count, size_in_bytes);

  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault};
  size_t attrs_idxs[1] = {0};

  StreamGuard stream_guard(Streams::created);

  size_t fail_index = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), k_count, &attr,
                                attrs_idxs, 1, &fail_index, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (int i = 0; i < k_count; ++i) {
    const std::vector<unsigned char>& expected = (i == 3) ? pattern_b : pattern_a;
    requireBufferEquals(dst_ptrs[i], expected, LinearAllocs::hipMalloc);
  }
}

// Batch of copies that reach their buffers through a pointer slot on the sides selected by `flags`.
// Each slot is allocated with the same allocation type as the buffer it references, so an indirect
// side keeps the source and destination pairing of the direct copy it replaces.
static void RunIndirectCopyTest(unsigned int flags, size_t count, size_t size_in_bytes,
                                LinearAllocs alloc_type_src, LinearAllocs alloc_type_dst) {
  const bool indirect_src = (flags & hipMemcpyFlagExtOpIndirectSrc) != 0;
  const bool indirect_dst = (flags & hipMemcpyFlagExtOpIndirectDst) != 0;
  const hipError_t expected_error = getIndirectExpectedReturn(alloc_type_src, alloc_type_dst);

  const std::vector<unsigned char> zeros(size_in_bytes, 0);
  std::vector<std::vector<unsigned char>> initial_values;
  std::vector<void*> src_ptrs(count);
  std::vector<void*> dst_ptrs(count);
  std::vector<void*> batch_src_ptrs(count);
  std::vector<void*> batch_dst_ptrs(count);
  std::vector<LinearAllocGuard<unsigned char>> allocations;
  std::vector<LinearAllocGuard<void*>> slots;

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);

  for (size_t i = 0; i < count; ++i) {
    initial_values.emplace_back(size_in_bytes, static_cast<unsigned char>(10 + i));

    LinearAllocGuard<unsigned char> src_alloc(alloc_type_src, size_in_bytes);
    src_ptrs[i] = src_alloc.ptr();
    allocations.push_back(std::move(src_alloc));
    fillBuffer(src_ptrs[i], initial_values[i], alloc_type_src);

    LinearAllocGuard<unsigned char> dst_alloc(alloc_type_dst, size_in_bytes);
    dst_ptrs[i] = dst_alloc.ptr();
    allocations.push_back(std::move(dst_alloc));
    fillBuffer(dst_ptrs[i], zeros, alloc_type_dst);

    batch_src_ptrs[i] = src_ptrs[i];
    batch_dst_ptrs[i] = dst_ptrs[i];

    if (indirect_src) {
      batch_src_ptrs[i] = addPointerSlot(slots, src_ptrs[i], alloc_type_src);
    }

    if (indirect_dst) {
      batch_dst_ptrs[i] = addPointerSlot(slots, dst_ptrs[i], alloc_type_dst);
    }
  }

  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  size_t attrs_idxs[1] = {0};

  HIP_CHECK_ERROR(
      hipMemcpyBatchAsync(batch_dst_ptrs.data(), batch_src_ptrs.data(), sizes.data(), count, &attr,
                          attrs_idxs, 1, nullptr, stream_guard.stream()),
      expected_error);

  // Unsupported allocation combinations are asserted to fail above; only the supported ones move
  // data worth verifying.
  if (expected_error == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(dst_ptrs[i], initial_values[i], alloc_type_dst);
    }
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies batched copies that reach their buffers through pointer slots across generated
 * indirect flag combinations, copy counts, sizes and per-side allocation types.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_IndirectCopy) {
  const unsigned int flags =
      GENERATE(as<unsigned int>{}, hipMemcpyFlagExtOpIndirectSrc, hipMemcpyFlagExtOpIndirectDst,
               hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst);
  const size_t count = GENERATE(1, 3, 8);
  const size_t size_in_bytes = GENERATE(as<size_t>{}, 1, 63, 4096);
  const LinearAllocs alloc_type_src =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const LinearAllocs alloc_type_dst =
      GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  CAPTURE(flags, count, size_in_bytes, alloc_type_src, alloc_type_dst);

  RunIndirectCopyTest(flags, count, size_in_bytes, alloc_type_src, alloc_type_dst);
}

/**
 * Test Description
 * ------------------------
 * - Verifies that hipMemcpyBatchAsync rejects the ExtOp attribute combinations that no device
 * accepts: swap paired with an indirect flag in one entry, and a pointer slot too small to hold a
 * pointer.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_ExtOp_Negative) {
  constexpr size_t kSizeInBytes = 4096;

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<unsigned char> src(LinearAllocs::hipHostMalloc, kSizeInBytes);
  LinearAllocGuard<unsigned char> dst(LinearAllocs::hipMalloc, kSizeInBytes);

  std::array<void*, 1> src_ptrs{src.ptr()};
  std::array<void*, 1> dst_ptrs{dst.ptr()};
  std::array<size_t, 1> sizes{kSizeInBytes};
  std::array<hipMemcpyAttributes, 1> attrs{
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault}};
  std::array<size_t, 1> attrs_idxs{0};

  SECTION("Swap combined with indirect in one attribute entry") {
    const unsigned int indirect_flags =
        GENERATE(as<unsigned int>{}, hipMemcpyFlagExtOpIndirectSrc, hipMemcpyFlagExtOpIndirectDst,
                 hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst);
    attrs[0].flags = hipMemcpyFlagExtOpSwap | indirect_flags;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                        dst_ptrs.size(), attrs.data(), attrs_idxs.data(),
                                        attrs.size(), nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  // An indirect side is range-checked against sizeof(void*) rather than against the copy size, so
  // this is rejected for holding less than a pointer even though it is smaller than the copy too.
  SECTION("Pointer slot smaller than a pointer") {
    LinearAllocGuard<unsigned char> short_slot(LinearAllocs::hipHostMalloc, sizeof(void*) / 2);
    src_ptrs[0] = short_slot.ptr();
    attrs[0].flags = hipMemcpyFlagExtOpIndirectSrc;
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                        dst_ptrs.size(), attrs.data(), attrs_idxs.data(),
                                        attrs.size(), nullptr, stream_guard.stream()),
                    hipErrorInvalidValue);
  }
}

namespace {

// Signal memory that blocks `waiting_stream` until `release_stream` releases it. The destructor
// releases the signal and drains both streams, so a failed assertion cannot leave the signal
// blocked: destroying a stream that is still waiting never returns, and the signal memory has to
// outlive the wait packet that references it.
class SignalGuard {
 public:
  SignalGuard(hipStream_t waiting_stream, hipStream_t release_stream)
      : waiting_stream_{waiting_stream}, release_stream_{release_stream} {
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal_), sizeof(uint64_t),
                                    hipMallocSignalMemory));
    uint64_t blocked = kBlocked;
    __atomic_store(signal_, &blocked, __ATOMIC_RELEASE);
  }

  SignalGuard(const SignalGuard&) = delete;
  SignalGuard& operator=(const SignalGuard&) = delete;

  ~SignalGuard() {
    // Cast to void to suppress nodiscard warnings; a destructor cannot assert.
    static_cast<void>(release());
    static_cast<void>(hipStreamSynchronize(release_stream_));
    static_cast<void>(hipStreamSynchronize(waiting_stream_));
    static_cast<void>(hipFree(signal_));
  }

  hipError_t enqueueWait() {
    return hipStreamWaitValue64(waiting_stream_, signal_, kReleased, hipStreamWaitValueEq);
  }

  hipError_t release() {
    return hipStreamWriteValue64(release_stream_, signal_, kReleased, hipStreamWriteValueDefault);
  }

 private:
  static constexpr uint64_t kBlocked = 1;
  static constexpr uint64_t kReleased = 0;

  hipStream_t waiting_stream_;
  hipStream_t release_stream_;
  uint64_t* signal_ = nullptr;
};

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Verifies that an indirect copy reads its pointer slot when the copy runs rather than when
 * hipMemcpyBatchAsync is called: the batch is enqueued behind a stream wait and a second stream
 * points the slots at the real buffers afterwards. Each slot starts out pointing at a decoy buffer,
 * so a copy that reads its slot before the publishing writes moves the wrong data instead of
 * dereferencing an unwritten slot.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_IndirectCopy_StreamOrderedPointer) {
  HIP_CHECK(hipSetDevice(0));
  int wait_value_supported = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&wait_value_supported, hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (wait_value_supported == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  const size_t count = GENERATE(1, 3, 8);
  const unsigned int flags =
      GENERATE(as<unsigned int>{}, hipMemcpyFlagExtOpIndirectSrc, hipMemcpyFlagExtOpIndirectDst,
               hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst);
  CAPTURE(count, flags);

  // The ExtOp path accepts only device and pinned host pairings on a single GPU, and stream memory
  // operations can publish into a slot allocated on either side of it.
  constexpr LinearAllocs kAllocTypeSrc = LinearAllocs::hipHostMalloc;
  constexpr LinearAllocs kAllocTypeDst = LinearAllocs::hipMalloc;
  constexpr size_t kSizeInBytes = 4096;

  const bool indirect_src = (flags & hipMemcpyFlagExtOpIndirectSrc) != 0;
  const bool indirect_dst = (flags & hipMemcpyFlagExtOpIndirectDst) != 0;
  const hipError_t expected_error = getIndirectExpectedReturn(kAllocTypeSrc, kAllocTypeDst);

  const std::vector<unsigned char> zeros(kSizeInBytes, 0);
  const std::vector<unsigned char> decoy_values(kSizeInBytes, 0xCD);
  std::vector<std::vector<unsigned char>> initial_values;
  std::vector<void*> src_ptrs(count);
  std::vector<void*> dst_ptrs(count);
  std::vector<void*> batch_src_ptrs(count);
  std::vector<void*> batch_dst_ptrs(count);
  std::vector<std::pair<void*, void*>> slots_to_publish;
  std::vector<LinearAllocGuard<unsigned char>> allocations;
  std::vector<LinearAllocGuard<void*>> slots;

  StreamGuard copy_stream(Streams::created);
  StreamGuard publish_stream(Streams::created);
  SignalGuard signal(copy_stream.stream(), publish_stream.stream());

  for (size_t i = 0; i < count; ++i) {
    initial_values.emplace_back(kSizeInBytes, static_cast<unsigned char>(10 + i));

    LinearAllocGuard<unsigned char> src_alloc(kAllocTypeSrc, kSizeInBytes);
    src_ptrs[i] = src_alloc.ptr();
    allocations.push_back(std::move(src_alloc));
    fillBuffer(src_ptrs[i], initial_values[i], kAllocTypeSrc);

    LinearAllocGuard<unsigned char> dst_alloc(kAllocTypeDst, kSizeInBytes);
    dst_ptrs[i] = dst_alloc.ptr();
    allocations.push_back(std::move(dst_alloc));
    fillBuffer(dst_ptrs[i], zeros, kAllocTypeDst);

    batch_src_ptrs[i] = src_ptrs[i];
    batch_dst_ptrs[i] = dst_ptrs[i];

    // A slot published early reads the decoy, which the verification below catches: the source
    // decoy carries a pattern the destination must not receive, and a destination decoy diverts the
    // copy away from the destination buffer, which stays zeroed.
    if (indirect_src) {
      LinearAllocGuard<unsigned char> decoy(kAllocTypeSrc, kSizeInBytes);
      fillBuffer(decoy.ptr(), decoy_values, kAllocTypeSrc);
      batch_src_ptrs[i] = addPointerSlot(slots, decoy.ptr(), kAllocTypeSrc);
      slots_to_publish.emplace_back(batch_src_ptrs[i], src_ptrs[i]);
      allocations.push_back(std::move(decoy));
    }

    if (indirect_dst) {
      LinearAllocGuard<unsigned char> decoy(kAllocTypeDst, kSizeInBytes);
      batch_dst_ptrs[i] = addPointerSlot(slots, decoy.ptr(), kAllocTypeDst);
      slots_to_publish.emplace_back(batch_dst_ptrs[i], dst_ptrs[i]);
      allocations.push_back(std::move(decoy));
    }
  }

  std::vector<size_t> sizes(count, kSizeInBytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  size_t attrs_idxs[1] = {0};

  HIP_CHECK(signal.enqueueWait());
  HIP_CHECK_ERROR(
      hipMemcpyBatchAsync(batch_dst_ptrs.data(), batch_src_ptrs.data(), sizes.data(), count, &attr,
                          attrs_idxs, 1, nullptr, copy_stream.stream()),
      expected_error);

  for (const auto& [slot, buffer] : slots_to_publish) {
    HIP_CHECK(hipStreamWriteValue64(publish_stream.stream(), slot,
                                    static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(buffer)),
                                    hipStreamWriteValueDefault));
  }
  HIP_CHECK(signal.release());
  HIP_CHECK(hipStreamSynchronize(publish_stream.stream()));
  HIP_CHECK(hipStreamSynchronize(copy_stream.stream()));

  if (expected_error == hipSuccess) {
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(dst_ptrs[i], initial_values[i], kAllocTypeDst);
    }
  }
}

/**
 * Test Description
 * ------------------------
 * - Verifies one batch that carries plain copies, swaps and indirect-source copies, each kind under
 * its own attribute entry spanning several copies.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_MixedAttributes_DefaultSwapIndirect) {
  constexpr size_t kSizeInBytes = 4096;
  // Each attribute entry covers a run of copies rather than a single one, so the batch exercises
  // the walk from a copy index to the attribute entry that governs it.
  constexpr size_t kCopiesPerAttr = 2;

  // The batch is rejected as a whole when either ExtOp packet is missing on this device.
  const hipError_t swap_error =
      getSwapExpectedReturn(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc);
  const hipError_t indirect_error =
      getIndirectExpectedReturn(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const hipError_t expected_error = (swap_error == hipSuccess && indirect_error == hipSuccess)
                                        ? hipSuccess
                                        : hipErrorNotSupported;

  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream_guard(Streams::created);

  struct Expectation {
    void* buffer;
    LinearAllocs alloc_type;
    std::vector<unsigned char> contents;
  };

  std::vector<LinearAllocGuard<unsigned char>> allocations;
  std::vector<LinearAllocGuard<void*>> slots;
  std::vector<void*> src_ptrs;
  std::vector<void*> dst_ptrs;
  std::vector<Expectation> expectations;

  const std::vector<unsigned char> zeros(kSizeInBytes, 0);
  const auto addBuffer = [&](const LinearAllocs alloc_type,
                             const std::vector<unsigned char>& contents) {
    LinearAllocGuard<unsigned char> alloc(alloc_type, kSizeInBytes);
    void* ptr = alloc.ptr();
    allocations.push_back(std::move(alloc));
    fillBuffer(ptr, contents, alloc_type);
    return ptr;
  };

  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> pattern(kSizeInBytes, static_cast<unsigned char>(10 + i));
    src_ptrs.push_back(addBuffer(LinearAllocs::hipMalloc, pattern));
    dst_ptrs.push_back(addBuffer(LinearAllocs::hipMalloc, zeros));
    expectations.push_back({dst_ptrs.back(), LinearAllocs::hipMalloc, pattern});
  }

  // A swap leaves each endpoint holding what the other one started with.
  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> device_pattern(kSizeInBytes,
                                                    static_cast<unsigned char>(20 + i));
    const std::vector<unsigned char> host_pattern(kSizeInBytes, static_cast<unsigned char>(30 + i));
    dst_ptrs.push_back(addBuffer(LinearAllocs::hipMalloc, device_pattern));
    src_ptrs.push_back(addBuffer(LinearAllocs::hipHostMalloc, host_pattern));
    expectations.push_back({dst_ptrs.back(), LinearAllocs::hipMalloc, host_pattern});
    expectations.push_back({src_ptrs.back(), LinearAllocs::hipHostMalloc, device_pattern});
  }

  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> pattern(kSizeInBytes, static_cast<unsigned char>(40 + i));
    void* indirect_src = addBuffer(LinearAllocs::hipHostMalloc, pattern);
    src_ptrs.push_back(addPointerSlot(slots, indirect_src, LinearAllocs::hipHostMalloc));
    dst_ptrs.push_back(addBuffer(LinearAllocs::hipMalloc, zeros));
    expectations.push_back({dst_ptrs.back(), LinearAllocs::hipMalloc, pattern});
  }

  std::vector<size_t> sizes(dst_ptrs.size(), kSizeInBytes);
  std::array<hipMemcpyAttributes, 3> attrs{
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpSwap},
      hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpIndirectSrc},
  };
  std::array<size_t, 3> attrs_idxs{0, kCopiesPerAttr, 2 * kCopiesPerAttr};

  HIP_CHECK_ERROR(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                      dst_ptrs.size(), attrs.data(), attrs_idxs.data(),
                                      attrs.size(), nullptr, stream_guard.stream()),
                  expected_error);

  if (expected_error == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    for (const Expectation& expectation : expectations) {
      requireBufferEquals(expectation.buffer, expectation.contents, expectation.alloc_type);
    }
  }
}
#endif
/**
 * End doxygen group MemoryTest.
 * @}
 */
