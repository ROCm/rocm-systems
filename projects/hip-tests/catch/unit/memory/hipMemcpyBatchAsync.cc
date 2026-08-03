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
 * hipStream_t stream __dparm(0))`
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

    if (values[front_index] != expected) {
      INFO("Array failure at copy index " << copy_index << ", element " << front_index);
      REQUIRE(values[front_index] == expected);
    }

    if (front_index != back_index && values[back_index] != expected) {
      INFO("Array failure at copy index " << copy_index << ", element " << back_index);
      REQUIRE(values[back_index] == expected);
    }
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
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDeviceBuffers(src_ptrs, copy_size, kSwapDstValue);
  VerifyDeviceBuffers(dst_ptrs, copy_size, kSwapSrcValue);
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
 * - Verifies H2H batch copies across independently generated source and
 * destination host allocation types and copy counts.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_H2H_Functional) {
  const LinearAllocs src_alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const LinearAllocs dst_alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const size_t copy_count = GENERATE(1, 8);
  const size_t copy_size = kSmallCopySize;

  BatchConfig config{copy_count, copy_size};
  StreamGuard stream_guard(Streams::created);
  std::vector<LinearAllocGuard<int>> src = AllocateBatchBuffers(src_alloc_type, config);
  std::vector<LinearAllocGuard<int>> dst = AllocateBatchBuffers(dst_alloc_type, config);
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
  skipMemcpyBatchAsyncIfAnyGfx1250();

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

static void RunIndirectCopyTest(unsigned int flags, size_t count, size_t size_in_bytes,
                                LinearAllocs alloc_type_src, LinearAllocs alloc_type_dst) {
  const bool is_indirect_src = flags & hipMemcpyFlagExtOpIndirectSrc;
  const bool is_indirect_dst = flags & hipMemcpyFlagExtOpIndirectDst;
  const hipError_t expected_error = getIndirectExpectedReturn(alloc_type_src, alloc_type_dst);

  IndirectCopyBuffers buffers =
      makeIndirectCopyBuffers(count, size_in_bytes, alloc_type_src, alloc_type_dst);

  for (size_t i = 0; i < count; ++i) {
    if (is_indirect_src) {
      buffers.batch_src_ptrs[i] =
          addPointerSlot(buffers.slots, buffers.src_ptrs[i], alloc_type_src);
    }

    if (is_indirect_dst) {
      buffers.batch_dst_ptrs[i] =
          addPointerSlot(buffers.slots, buffers.dst_ptrs[i], alloc_type_dst);
    }
  }

  StreamGuard stream_guard(Streams::created);
  std::vector<size_t> sizes(count, size_in_bytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  size_t attrs_idxs[1] = {0};

  HIP_CHECK_ERROR(hipMemcpyBatchAsync(buffers.batch_dst_ptrs.data(), buffers.batch_src_ptrs.data(),
                                      sizes.data(), count, &attr, attrs_idxs, 1, nullptr,
                                      stream_guard.stream()),
                  expected_error);

  // Unsupported allocation combinations are asserted to fail above; only the supported ones move
  // data worth verifying.
  if (expected_error == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(buffers.dst_ptrs[i], buffers.initial_values[i], alloc_type_dst);
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

/**
 * Test Description
 * ------------------------
 * Verifies that an indirect copy dereferences its pointer slot on the device when the copy runs,
 * not on the host when hipMemcpyBatchAsync is called. The other indirect tests fill their slots
 * before the call and so pass under either behavior; here the batch is enqueued behind a stream
 * wait and a second stream publishes the real buffer addresses into the slots afterwards, so only
 * a copy that reads its slot at execution time finds them.
 *
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_IndirectCopy_StreamOrderedPointer) {
  int wait_value_supported = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&wait_value_supported, hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (wait_value_supported == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  const size_t count = GENERATE(1, 8);
  const unsigned int flags =
      GENERATE(as<unsigned int>{}, hipMemcpyFlagExtOpIndirectSrc, hipMemcpyFlagExtOpIndirectDst,
               hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst);

  const LinearAllocs alloc_type_src =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  const LinearAllocs alloc_type_dst =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
  CAPTURE(count, flags, alloc_type_src, alloc_type_dst);

  constexpr size_t kSizeInBytes = 4096;
  constexpr uint64_t kBlocked = 1;
  constexpr uint64_t kReleased = 0;

  const bool is_indirect_src = flags & hipMemcpyFlagExtOpIndirectSrc;
  const bool is_indirect_dst = flags & hipMemcpyFlagExtOpIndirectDst;
  const hipError_t expected_return = getIndirectExpectedReturn(alloc_type_src, alloc_type_dst);

  IndirectCopyBuffers buffers =
      makeIndirectCopyBuffers(count, kSizeInBytes, alloc_type_src, alloc_type_dst);

  const std::vector<unsigned char> decoy_values(kSizeInBytes, 0xCD);
  // Each entry pairs a slot with the real address it must receive once the batch is enqueued.
  std::vector<std::pair<void*, void*>> slots_to_publish;

  for (size_t i = 0; i < count; ++i) {
    // The slots start out pointing at decoys instead of the real buffers, so a slot read taken
    // before the addresses are published gives a wrong result rather than a crash.
    if (is_indirect_src) {
      LinearAllocGuard<unsigned char> decoy(alloc_type_src, kSizeInBytes);
      fillBuffer(decoy.ptr(), decoy_values, alloc_type_src);
      buffers.batch_src_ptrs[i] = addPointerSlot(buffers.slots, decoy.ptr(), alloc_type_src);
      slots_to_publish.emplace_back(buffers.batch_src_ptrs[i], buffers.src_ptrs[i]);
      buffers.allocations.push_back(std::move(decoy));
    }

    if (is_indirect_dst) {
      LinearAllocGuard<unsigned char> decoy(alloc_type_dst, kSizeInBytes);
      buffers.batch_dst_ptrs[i] = addPointerSlot(buffers.slots, decoy.ptr(), alloc_type_dst);
      slots_to_publish.emplace_back(buffers.batch_dst_ptrs[i], buffers.dst_ptrs[i]);
      buffers.allocations.push_back(std::move(decoy));
    }
  }

  StreamGuard copy_stream(Streams::created);
  StreamGuard publish_stream(Streams::created);

  // The value hipStreamWaitValue64 polls has to live in signal memory.
  uint64_t* signal = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  __atomic_store_n(signal, kBlocked, __ATOMIC_RELEASE);

  std::vector<size_t> sizes(count, kSizeInBytes);
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  size_t attrs_idxs[1] = {0};

  // The publishing writes have to be submitted only after hipMemcpyBatchAsync has returned.
  // Submitting them first would let them land before the call, and a slot read at call time would
  // then find the real address anyway. Blocking copy_stream is what keeps the batch from running
  // before writes.
  HIP_CHECK(hipStreamWaitValue64(copy_stream.stream(), signal, kReleased, hipStreamWaitValueEq));

  const hipError_t batch_status =
      hipMemcpyBatchAsync(buffers.batch_dst_ptrs.data(), buffers.batch_src_ptrs.data(),
                          sizes.data(), count, &attr, attrs_idxs, 1, nullptr, copy_stream.stream());

  hipError_t publish_status = hipSuccess;
  for (const auto& [slot, buffer] : slots_to_publish) {
    if (publish_status != hipSuccess) break;
    const uint64_t address = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(buffer));
    publish_status =
        hipStreamWriteValue64(publish_stream.stream(), slot, address, hipStreamWriteValueDefault);
  }
  const hipError_t publish_sync_status = hipStreamSynchronize(publish_stream.stream());

  // Signal memory is host accessible, so releasing from the host cannot fail and cannot throw. The
  // publishing writes have already landed because publish_stream has drained.
  __atomic_store_n(signal, kReleased, __ATOMIC_RELEASE);

  const hipError_t copy_sync_status = hipStreamSynchronize(copy_stream.stream());
  HIP_CHECK(hipFree(signal));

  HIP_CHECK(publish_status);
  HIP_CHECK(publish_sync_status);
  HIP_CHECK(copy_sync_status);
  HIP_CHECK_ERROR(batch_status, expected_return);

  if (expected_return == hipSuccess) {
    for (size_t i = 0; i < count; ++i) {
      requireBufferEquals(buffers.dst_ptrs[i], buffers.initial_values[i], alloc_type_dst);
    }
  }
}

/**
 * Test Description
 * ------------------------
 * Verifies one batch that carries plain copies, swaps and indirect-source copies, each kind under
 * its own attribute entry spanning several copies.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_MixedAttributes_DefaultSwapIndirect) {
  skipMemcpyBatchAsyncIfAnyGfx1250();

  constexpr size_t kSizeInBytes = 4096;
  constexpr size_t kCopiesPerAttr = 2;

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

  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> pattern(kSizeInBytes, static_cast<unsigned char>(10 + i));
    src_ptrs.push_back(addBuffer(allocations, pattern, LinearAllocs::hipMalloc));
    dst_ptrs.push_back(addBuffer(allocations, zeros, LinearAllocs::hipMalloc));
    expectations.push_back({dst_ptrs.back(), LinearAllocs::hipMalloc, pattern});
  }

  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> device_pattern(kSizeInBytes,
                                                    static_cast<unsigned char>(20 + i));
    const std::vector<unsigned char> host_pattern(kSizeInBytes, static_cast<unsigned char>(30 + i));
    dst_ptrs.push_back(addBuffer(allocations, device_pattern, LinearAllocs::hipMalloc));
    src_ptrs.push_back(addBuffer(allocations, host_pattern, LinearAllocs::hipHostMalloc));
    expectations.push_back({dst_ptrs.back(), LinearAllocs::hipMalloc, host_pattern});
    expectations.push_back({src_ptrs.back(), LinearAllocs::hipHostMalloc, device_pattern});
  }

  for (size_t i = 0; i < kCopiesPerAttr; ++i) {
    const std::vector<unsigned char> pattern(kSizeInBytes, static_cast<unsigned char>(40 + i));
    void* indirect_src = addBuffer(allocations, pattern, LinearAllocs::hipHostMalloc);
    src_ptrs.push_back(addPointerSlot(slots, indirect_src, LinearAllocs::hipHostMalloc));
    dst_ptrs.push_back(addBuffer(allocations, zeros, LinearAllocs::hipMalloc));
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

/**
 * Test Description
 * ------------------------
 * - Verifies a large D2D batch through the shader and SDMA copy paths.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_LargeBatch) {
  constexpr size_t copy_count = 1024 * 32;
  constexpr size_t copy_size = 64;
  constexpr size_t total_size = copy_count * copy_size;
  constexpr size_t copy_elements = copy_size / sizeof(int);
  const hipMemcpyFlags flag = GENERATE(hipMemcpyFlagDefault, hipMemcpyFlagExtPreferCE);

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, total_size);
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, total_size);
  std::vector<void*> src_ptrs(copy_count);
  std::vector<void*> dst_ptrs(copy_count);
  std::vector<size_t> sizes(copy_count, copy_size);
  std::vector<int> values(total_size / sizeof(int));
  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderAny, {}, {}, static_cast<unsigned int>(flag)};
  size_t attrs_idx = 0;

  for (size_t i = 0; i < copy_count; ++i) {
    src_ptrs[i] = reinterpret_cast<char*>(src.ptr()) + i * copy_size;
    dst_ptrs[i] = reinterpret_cast<char*>(dst.ptr()) + i * copy_size;
    std::fill_n(values.data() + i * copy_elements, copy_elements,
                kPatternValue + static_cast<int>(i));
  }
  HIP_CHECK(hipMemcpy(src.ptr(), values.data(), total_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), copy_count, &attr,
                                &attrs_idx, 1, nullptr, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  HIP_CHECK(hipMemcpy(values.data(), dst.ptr(), total_size, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < copy_count; ++i) {
    VerifyArrayFromBothEnds(values.data() + i * copy_elements, copy_elements,
                            kPatternValue + static_cast<int>(i), i);
  }
}
#endif

#if HT_AMD
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_cp) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  // Allocate device buffers
  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  // Fill with distinct values
  std::vector<int> hostA(kNumElements, kValA);
  std::vector<int> hostB(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeBytes, hipMemcpyHostToDevice));

  // Set up batch swap: dst=d_a, src=d_b with swap flag
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1,
                                       &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back and verify swap
  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verifies swap between pinned host memory and device memory (H<->D).
 *   Swap operands only need to be SDMA-accessible (device or pinned host), not
 *   strictly D2D, so this exercises the host<->device swap path.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_HD_Swap) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValHost = 42;
  constexpr int kValDev = 99;

  // One pinned-host operand, one device operand.
  int* h_pinned = nullptr;
  void* d_dev = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_pinned), kSizeBytes));
  HIP_CHECK(hipMalloc(&d_dev, kSizeBytes));

  for (size_t i = 0; i < kNumElements; i++) h_pinned[i] = kValHost;
  std::vector<int> hostDev(kNumElements, kValDev);
  HIP_CHECK(hipMemcpy(d_dev, hostDev.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_dev};
  void* srcs[] = {h_pinned};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1,
                                       &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_pinned));
    HIP_CHECK(hipFree(d_dev));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Device side now holds the host pattern; host side holds the device pattern.
  std::vector<int> resultDev(kNumElements);
  HIP_CHECK(hipMemcpy(resultDev.data(), d_dev, kSizeBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultDev[i] == kValHost);
    REQUIRE(h_pinned[i] == kValDev);
  }

  HIP_CHECK(hipHostFree(h_pinned));
  HIP_CHECK(hipFree(d_dev));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify asymmetric swap: size_a > size_b.
 *   Asymmetric swap decomposes on the CLR side into:
 *   1. Swap min(size_a, size_b) bytes between A and B
 *   2. Copy remaining (size_a - size_b) bytes from A to B
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric) {
  constexpr size_t kSizeA = 8192;  // 8 KB
  constexpr size_t kSizeB = 4096;  // 4 KB (smaller side)
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  // Allocate two device buffers, both large enough for kSizeA
  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  // Fill A entirely with kValA, B entirely with kValB
  std::vector<int> hostA(kSizeA / sizeof(int), kValA);
  std::vector<int> hostB(kSizeA / sizeof(int), kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeA};
  size_t sizesB[] = {kSizeB};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back both buffers
  std::vector<int> resultA(kSizeA / sizeof(int));
  std::vector<int> resultB(kSizeA / sizeof(int));
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeA, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  size_t swapElems = kSizeB / sizeof(int);  // Elements in the swapped region
  size_t totalElems = kSizeA / sizeof(int);

  // A[0..swapElems-1] = old B values (swapped)
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(resultA[i] == kValB);
  }
  // A[swapElems..end] = unchanged (still kValA)
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(resultA[i] == kValA);
  }

  // B[0..swapElems-1] = old A values (swapped)
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(resultB[i] == kValA);
  }
  // B[swapElems..end] = old A tail values (copied from A)
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Same asymmetric swap as Unit_hipMemcpyBatchAsync_swap_asymmetric, but the
 *   swap op is driven via ops[] (hipExtMemcpyOpSwap) instead of attrs[].flags.
 *   Confirms sizesB is honored when the swap comes from the ops[] path.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_swap_asymmetric_ops) {
  constexpr size_t kSizeA = 8192;  // 8 KB
  constexpr size_t kSizeB = 4096;  // 4 KB (smaller side)
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  std::vector<int> hostA(kSizeA / sizeof(int), kValA);
  std::vector<int> hostB(kSizeA / sizeof(int), kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeA};
  size_t sizesB[] = {kSizeB};
  size_t attrsIdxs[] = {0};

  // Swap requested via ops[], NOT attrs.flags.
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpSwap};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, ops,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kSizeA / sizeof(int));
  std::vector<int> resultB(kSizeA / sizeof(int));
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeA, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  const size_t swapElems = kSizeB / sizeof(int);
  const size_t totalElems = kSizeA / sizeof(int);
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(resultA[i] == kValB);  // head swapped
    REQUIRE(resultB[i] == kValA);
  }
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(resultA[i] == kValA);  // A tail unchanged
    REQUIRE(resultB[i] == kValA);  // B tail copied from A
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify asymmetric swap with multiple attribute ranges.
 *   Two swap pairs under separate attributes with different swapSizesA/B,
 *   testing that rangeIdx (idx - attrsIdxs[attrIdx]) indexes correctly.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric_multi_attr) {
  // Pair 0: attr[0], swap 8KB A with 4KB B
  // Pair 1: attr[1], swap 4KB A with 2KB B
  constexpr int kValA0 = 10, kValB0 = 20;
  constexpr int kValA1 = 30, kValB1 = 40;
  constexpr size_t kBufSize = 8192;  // all allocations are 8KB

  void *dA0, *dB0, *dA1, *dB1;
  HIP_CHECK(hipMalloc(&dA0, kBufSize));
  HIP_CHECK(hipMalloc(&dB0, kBufSize));
  HIP_CHECK(hipMalloc(&dA1, kBufSize));
  HIP_CHECK(hipMalloc(&dB1, kBufSize));

  // Fill each buffer with its value
  std::vector<int> hA0(kBufSize / sizeof(int), kValA0);
  std::vector<int> hB0(kBufSize / sizeof(int), kValB0);
  std::vector<int> hA1(kBufSize / sizeof(int), kValA1);
  std::vector<int> hB1(kBufSize / sizeof(int), kValB1);
  HIP_CHECK(hipMemcpy(dA0, hA0.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB0, hB0.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dA1, hA1.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB1, hB1.data(), kBufSize, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {dA0, dA1};
  void* srcs[] = {dB0, dB1};
  size_t sizesA[] = {8192, 4096};
  size_t sizesB[] = {4096, 2048};

  hipMemcpyAttributes attrs[2] = {};
  attrs[0].flags = hipMemcpyFlagExtOpSwap;
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = hipMemcpyFlagExtOpSwap;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t attrsIdxs[] = {0, 1};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, nullptr,
                                          2, attrs, attrsIdxs, 2,
                                          stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA0)); HIP_CHECK(hipFree(dB0));
    HIP_CHECK(hipFree(dA1)); HIP_CHECK(hipFree(dB1));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back all buffers
  std::vector<int> rA0(kBufSize / sizeof(int));
  std::vector<int> rB0(kBufSize / sizeof(int));
  std::vector<int> rA1(kBufSize / sizeof(int));
  std::vector<int> rB1(kBufSize / sizeof(int));
  HIP_CHECK(hipMemcpy(rA0.data(), dA0, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rB0.data(), dB0, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rA1.data(), dA1, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rB1.data(), dB1, kBufSize, hipMemcpyDeviceToHost));

  // Pair 0: swap 4KB (1024 ints), copy 4KB tail from A0 to B0
  size_t swap0 = 4096 / sizeof(int);
  size_t total = kBufSize / sizeof(int);
  for (size_t i = 0; i < swap0; i++) {
    REQUIRE(rA0[i] == kValB0);   // swapped from B0
    REQUIRE(rB0[i] == kValA0);   // swapped from A0
  }
  for (size_t i = swap0; i < total; i++) {
    REQUIRE(rA0[i] == kValA0);   // unchanged
    REQUIRE(rB0[i] == kValA0);   // tail copied from A0
  }

  // Pair 1: swap 2KB (512 ints), copy 2KB tail from A1 to B1
  size_t swap1 = 2048 / sizeof(int);
  for (size_t i = 0; i < swap1; i++) {
    REQUIRE(rA1[i] == kValB1);   // swapped from B1
    REQUIRE(rB1[i] == kValA1);   // swapped from A1
  }
  for (size_t i = swap1; i < 4096 / sizeof(int); i++) {
    REQUIRE(rA1[i] == kValA1);   // unchanged
    REQUIRE(rB1[i] == kValA1);   // tail copied from A1
  }
  // Beyond swapSizesA: untouched by the swap operation
  for (size_t i = 4096 / sizeof(int); i < total; i++) {
    REQUIRE(rA1[i] == kValA1);   // unchanged
    REQUIRE(rB1[i] == kValB1);   // unchanged
  }

  HIP_CHECK(hipFree(dA0)); HIP_CHECK(hipFree(dB0));
  HIP_CHECK(hipFree(dA1)); HIP_CHECK(hipFree(dB1));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipExtMemcpyBatchAsync with sizesB = nullptr performs
 *   a full symmetric swap (same as hipMemcpyBatchAsync).
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric_fallback) {
  constexpr size_t kSizeA = 8192;
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  std::vector<int> hostA(kSizeA / sizeof(int), kValA);
  std::vector<int> hostB(kSizeA / sizeof(int), kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeA};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  // Use hipExtMemcpyBatchAsync with sizesB = nullptr.
  // Full symmetric swap expected even though sizesA > sizesB would apply.
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kSizeA / sizeof(int));
  std::vector<int> resultB(kSizeA / sizeof(int));
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeA, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  size_t totalElems = kSizeA / sizeof(int);

  // Full symmetric swap: ALL of A should be kValB, ALL of B should be kValA
  for (size_t i = 0; i < totalElems; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify that per-entry ops can specify swap operation instead
 *   of using hipMemcpyAttributes.flags. The attrs have no swap flag set;
 *   only ops specifies the swap.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_entryFlags_swap) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  std::vector<int> hostA(kNumElements, kValA);
  std::vector<int> hostB(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};

  // Attrs have NO swap flag — swap is specified only via ops
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  size_t attrsIdxs[] = {0};

  hipExtMemcpyOp ops[] = {hipExtMemcpyOpSwap};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verifies hipExtMemcpyBatchAsync rejects unknown/reserved bits in ops[]
 *   with hipErrorInvalidValue.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_UnknownOpsBits_Negative) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  size_t attrsIdxs[] = {0};

  // 0x8 is outside the valid hipExtMemcpyOp bit set (Swap|IndirectSrc|IndirectDst).
  hipExtMemcpyOp ops[] = {static_cast<hipExtMemcpyOp>(0x8)};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  REQUIRE(err == hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - When ops[] is provided it is authoritative for the op type: an entry with
 *   hipExtMemcpyOpDefault performs a plain linear copy even if attrs[].flags
 *   also carries hipMemcpyFlagExtOpSwap. Verifies the copy runs (dst gets src)
 *   and no swap occurred (src is unchanged).
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_OpsDefault_OverridesAttrSwap) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValDst = 42;
  constexpr int kValSrc = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  std::vector<int> hostDst(kNumElements, kValDst);
  std::vector<int> hostSrc(kNumElements, kValSrc);
  HIP_CHECK(hipMemcpy(d_a, hostDst.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostSrc.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  // attrs asks for swap, but ops[] says Default -> ops[] wins: plain copy.
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpSwap;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpDefault};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValSrc);  // dst received src (copy happened)
    REQUIRE(resultB[i] == kValSrc);  // src unchanged (no swap)
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}


/**
 * Test Description
 * ------------------------
 * - Ops-only usage: attrs is NULL (numAttrs = 0) and the swap op is driven
 *   entirely via ops[]. Verifies the attrs-NULL path (default metadata, op type
 *   from ops[]) performs a correct D2D swap.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_OpsOnly_NullAttrs_Swap) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  std::vector<int> hostA(kNumElements, kValA);
  std::vector<int> hostB(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpSwap};

  // attrs / attrsIdxs NULL, numAttrs 0 -> op type comes solely from ops[].
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops, 1,
                                          nullptr, nullptr, 0, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - ops[] is authoritative: an ops[i]=hipExtMemcpyOpDefault entry must NOT be
 *   rejected by the swap/indirect support gates just because attrs[].flags
 *   carries an op bit. Uses an indirect flag in attrs with ops[i]=Default on a
 *   plain D2D copy; the call must succeed as a linear copy (not return
 *   hipErrorNotSupported), even on devices without indirect-copy support.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_OpsDefault_IgnoresAttrIndirectGate) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValDst = 7;
  constexpr int kValSrc = 88;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  std::vector<int> hostDst(kNumElements, kValDst);
  std::vector<int> hostSrc(kNumElements, kValSrc);
  HIP_CHECK(hipMemcpy(d_a, hostDst.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostSrc.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  // attrs requests IndirectSrc, but ops[] says Default -> ops wins: linear copy.
  // The indirect-support gate must not reject this even where indirect is
  // unsupported, because ops[] is authoritative.
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpIndirectSrc;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpDefault};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops, 1,
                                          &attr, attrsIdxs, 1, stream);
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValSrc);  // plain linear copy: dst received src
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - The GPU-side wait/signal parameters are reserved and must be NULL. Passing a
 *   non-NULL waits or signals array returns hipErrorNotSupported. Device-
 *   independent: the check runs before any device/op dispatch.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_WaitsSignals_Reserved_Negative) {
  constexpr size_t kBytes = 1024;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kBytes};
  size_t attrsIdxs[] = {0};
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipExtMemcpyWait waits[1] = {};
  hipExtMemcpySignal signals[1] = {};

  // Non-NULL waits -> reserved -> hipErrorNotSupported.
  hipError_t errW = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                           waits, nullptr, nullptr, 1,
                                           &attr, attrsIdxs, 1, stream);
  REQUIRE(errW == hipErrorNotSupported);

  // Non-NULL signals -> reserved -> hipErrorNotSupported.
  hipError_t errS = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                           nullptr, signals, nullptr, 1,
                                           &attr, attrsIdxs, 1, stream);
  REQUIRE(errS == hipErrorNotSupported);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - For a swap entry, sizesB[i] must be non-zero and <= sizesA[i]. Both a zero
 *   and an over-large sizesB return hipErrorInvalidValue. Skipped where swap
 *   is unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_SizesB_Constraint_Negative) {
  constexpr size_t kBytes = 4096;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kBytes};
  size_t attrsIdxs[] = {0};
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpSwap};

  // Case 1: sizesB[0] == 0 (invalid for a swap entry).
  size_t sizesB_zero[] = {0};
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB_zero,
                                          nullptr, nullptr, ops, 1,
                                          &attr, attrsIdxs, 1, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  REQUIRE(err == hipErrorInvalidValue);

  // Case 2: sizesB[0] > sizesA[0] (invalid for a swap entry).
  size_t sizesB_big[] = {kBytes * 2};
  err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB_big,
                               nullptr, nullptr, ops, 1,
                               &attr, attrsIdxs, 1, stream);
  REQUIRE(err == hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Asymmetric swap (sizesA > sizesB) between device and pinned host memory.
 *   The first sizesB bytes are swapped between the two buffers; the A-side tail
 *   (sizesA - sizesB) is copied into B. Verifies the exchanged head and copied
 *   tail on both operands.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_HD_Swap_Asymmetric) {
  constexpr size_t kSizeA = 8192;
  constexpr size_t kSizeB = 4096;
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  int* h_a = nullptr;             // pinned host (A side)
  void* d_b = nullptr;            // device (B side)
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_a), kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  const size_t totalElems = kSizeA / sizeof(int);
  for (size_t i = 0; i < totalElems; i++) h_a[i] = kValA;
  std::vector<int> hostB(totalElems, kValB);
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {h_a};   // A = pinned host
  void* srcs[] = {d_b};   // B = device
  size_t sizesA[] = {kSizeA};
  size_t sizesB[] = {kSizeB};
  size_t attrsIdxs[] = {0};
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpSwap;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, nullptr, 1,
                                          &attr, attrsIdxs, 1, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultB(totalElems);
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  const size_t swapElems = kSizeB / sizeof(int);
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(h_a[i] == kValB);      // A head: swapped from B
    REQUIRE(resultB[i] == kValA);  // B head: swapped from A
  }
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(h_a[i] == kValA);      // A tail: unchanged
    REQUIRE(resultB[i] == kValA);  // B tail: copied from A
  }

  HIP_CHECK(hipHostFree(h_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Indirect source specified via ops[] (hipExtMemcpyOpIndirectSrc): srcs[i]
 *   holds a pointer to the real source buffer, read when the copy executes.
 *   Skipped where indirect copy is unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_ops_IndirectSrc) {
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
  std::vector<size_t> sizesA{copy_size};
  size_t attrs_idx = 0;
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpIndirectSrc};

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizesA.data(),
                                             nullptr, nullptr, nullptr, ops, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDeviceBuffers(dst_ptrs, copy_size);
}

/**
 * Test Description
 * ------------------------
 * - Indirect destination specified via ops[] (hipExtMemcpyOpIndirectDst):
 *   dsts[i] holds a pointer to the real destination buffer, read when the copy
 *   executes. Skipped where indirect copy is unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_ops_IndirectDst) {
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
  std::vector<size_t> sizesA{copy_size};
  size_t attrs_idx = 0;
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  hipExtMemcpyOp ops[] = {hipExtMemcpyOpIndirectDst};

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizesA.data(),
                                             nullptr, nullptr, nullptr, ops, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyArrayFromBothEnds(dst.host_ptr(), copy_elements, kPatternValue, 0);
}

/**
 * Test Description
 * ------------------------
 * - Both source and destination indirect via ops[] (hipExtMemcpyOpIndirectSrc |
 *   hipExtMemcpyOpIndirectDst): srcs[i] and dsts[i] each hold a pointer to the
 *   real buffer, read when the copy executes. Skipped where indirect copy is
 *   unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_ops_IndirectSrcDst) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipHostMalloc, copy_size);  // real src (host)
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, copy_size);      // real dst (device)
  LinearAllocGuard<char> src_slot(LinearAllocs::hipHostMalloc, sizeof(void*));  // host slot
  LinearAllocGuard<char> dst_slot(LinearAllocs::hipMalloc, sizeof(void*));      // device slot

  const size_t copy_elements = copy_size / sizeof(int);
  std::fill_n(src.host_ptr(), copy_elements, kPatternValue);

  void* src_ptr = src.ptr();
  void* dst_ptr = dst.ptr();
  std::memcpy(src_slot.ptr(), &src_ptr, sizeof(void*));
  HIP_CHECK(hipMemcpy(dst_slot.ptr(), &dst_ptr, sizeof(void*), hipMemcpyHostToDevice));

  std::vector<void*> src_ptrs{src_slot.ptr()};
  std::vector<void*> dst_ptrs{dst_slot.ptr()};
  std::vector<size_t> sizesA{copy_size};
  size_t attrs_idx = 0;
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  hipExtMemcpyOp ops[] = {static_cast<hipExtMemcpyOp>(hipExtMemcpyOpIndirectSrc |
                                                      hipExtMemcpyOpIndirectDst)};

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizesA.data(),
                                             nullptr, nullptr, nullptr, ops, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDeviceBuffers(std::vector<void*>{dst.ptr()}, copy_size);
}
#endif
/**
 * Test Description
 * ------------------------
 * - Test case to verify the negative cases of hipMemcpyBatchAsync.
 * 1. Dst Array as nullptr.
 * 2. Src Array as nullptr.
 * 3. Operations Count as 0.
 * 4. Num of attributes as 0.
 * 5. Sizes Array as nullptr.
 * 6. Attr Array as nullptr.
 * 7. AttrsIdxs Array as nullptr.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_NegativeTsts) {
  const size_t count = 2;
  size_t numAttrs = 0;
  size_t sizes[2];
  size_t attrsIdxs[1];
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));
  void *srcPtr[count], *dstPtr[count];
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipMalloc(&srcPtr[i], size));
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
    sizes[i] = size;
  }

  attrsIdxs[0] = 0;
  size_t failIdx;
  SECTION("Dst Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(nullptr, srcPtr, sizes, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Src Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, nullptr, sizes, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Count as zero") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, 0, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("sizes Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, nullptr, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
#if 0 // Enable these tests when support for memcpy attributes is enabled.
  SECTION("Number of Attributes as zero") {
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, attrsIdxs, 0, &failIdx, stream),
        hipErrorInvalidValue);
  }
  SECTION("Attr Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                        &failIdx, stream),
                    hipErrorInvalidValue);
  }

  SECTION("attrsIdxs Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, nullptr, numAttrs,
                                        &failIdx, stream),
                    hipErrorInvalidValue);
  }
#endif
  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(srcPtr[i]));
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
