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

size_t CopyElements(size_t copy_size) {
  REQUIRE(copy_size % sizeof(int) == 0);
  return copy_size / sizeof(int);
}

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
  const size_t copy_elements = CopyElements(copy_size);
  std::vector<int> source(copy_elements);

  for (size_t i = 0; i < ptrs.size(); ++i) {
    std::fill(source.begin(), source.end(), value + static_cast<int>(i));
    HIP_CHECK(hipMemcpy(ptrs[i], source.data(), copy_size, hipMemcpyHostToDevice));
  }
}

void FillHostBuffers(std::vector<LinearAllocGuard<int>>& buffers, size_t copy_size,
                     int value = kPatternValue) {
  const size_t copy_elements = CopyElements(copy_size);

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
  const size_t copy_elements = CopyElements(copy_size);
  std::vector<int> result(copy_elements);

  for (size_t i = 0; i < ptrs.size(); ++i) {
    HIP_CHECK(hipMemcpy(result.data(), ptrs[i], copy_size, hipMemcpyDeviceToHost));
    const int value = expected + (add_index ? static_cast<int>(i) : 0);
    VerifyArrayFromBothEnds(result.data(), copy_elements, value, i);
  }
}

void VerifyHostBuffers(std::vector<LinearAllocGuard<int>>& buffers, size_t copy_size,
                       int expected = kPatternValue) {
  const size_t copy_elements = CopyElements(copy_size);

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
 * - Verifies segmented host-source access attributes and swap attributes for
 * hipMemcpyBatchAsync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_Attrs_Functional) {
  constexpr size_t kCopiesPerAttr = 3;
  StreamGuard stream_guard(Streams::created);

  SECTION("Regular attributes") {
#if HT_AMD
    std::array<hipMemcpyAttributes, 6> host_attrs{
        hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtPreferCE},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderDuringApiCall, {}, {}, hipMemcpyFlagDefault},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderDuringApiCall, {}, {}, hipMemcpyFlagExtPreferCE},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderAny, {}, {}, hipMemcpyFlagDefault},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderAny, {}, {}, hipMemcpyFlagExtPreferCE},
    };
#else
    std::array<hipMemcpyAttributes, 3> host_attrs{
        hipMemcpyAttributes{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagDefault},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderDuringApiCall, {}, {}, hipMemcpyFlagDefault},
        hipMemcpyAttributes{hipMemcpySrcAccessOrderAny, {}, {}, hipMemcpyFlagDefault},
    };
#endif
    BatchConfig host_config{host_attrs.size() * kCopiesPerAttr, kSmallCopySize};
    std::vector<LinearAllocGuard<int>> host_src =
        AllocateBatchBuffers(LinearAllocs::hipHostMalloc, host_config);
    std::vector<LinearAllocGuard<int>> dst =
        AllocateBatchBuffers(LinearAllocs::hipMalloc, host_config);
    std::vector<void*> host_src_ptrs = MakeBatchPtrs(host_src);
    std::vector<void*> dst_ptrs = MakeBatchPtrs(dst);
    std::vector<size_t> sizes(host_src_ptrs.size(), kSmallCopySize);
    std::vector<size_t> attrs_idxs;

    for (size_t attr_idx = 0; attr_idx < host_attrs.size(); ++attr_idx) {
      attrs_idxs.push_back(attr_idx * kCopiesPerAttr);
    }

    FillHostBuffers(host_src, kSmallCopySize);

    HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), host_src_ptrs.data(), sizes.data(), sizes.size(),
                                  host_attrs.data(), attrs_idxs.data(), attrs_idxs.size(), nullptr,
                                  stream_guard.stream()));
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    VerifyDeviceBuffers(dst_ptrs, kSmallCopySize);
  }

#if HT_AMD
  SECTION("Swap attribute") {
    BatchConfig d2d_config{kCopiesPerAttr, kSmallCopySize};
    std::vector<LinearAllocGuard<int>> d2d_src =
        AllocateBatchBuffers(LinearAllocs::hipMalloc, d2d_config);
    std::vector<LinearAllocGuard<int>> d2d_dst =
        AllocateBatchBuffers(LinearAllocs::hipMalloc, d2d_config);
    std::vector<void*> d2d_src_ptrs = MakeBatchPtrs(d2d_src);
    std::vector<void*> d2d_dst_ptrs = MakeBatchPtrs(d2d_dst);
    std::vector<size_t> sizes(d2d_src_ptrs.size(), kSmallCopySize);
    hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, hipMemcpyFlagExtOpSwap};
    size_t attrs_idxs[1] = {0};
    constexpr int kSwapSrcValue = 23;
    constexpr int kSwapDstValue = 47;
    FillDeviceBuffers(d2d_src_ptrs, kSmallCopySize, kSwapSrcValue);
    FillDeviceBuffers(d2d_dst_ptrs, kSmallCopySize, kSwapDstValue);

    hipError_t status = hipMemcpyBatchAsync(d2d_dst_ptrs.data(), d2d_src_ptrs.data(), sizes.data(),
                                            d2d_src_ptrs.size(), &attr, attrs_idxs, 1, nullptr,
                                            stream_guard.stream());
    if (status == hipErrorNotSupported) {
      SUCCEED("hipMemcpyFlagExtOpSwap is not supported on this device");
    } else {
      HIP_CHECK(status);
      HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
      VerifyDeviceBuffers(d2d_src_ptrs, kSmallCopySize, kSwapDstValue);
      VerifyDeviceBuffers(d2d_dst_ptrs, kSmallCopySize, kSwapSrcValue);
    }
  }
#endif
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
      hipMemcpySrcAccessOrderStream, {}, {}, static_cast<unsigned int>(flag)};

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
  const size_t copy_elements = CopyElements(copy_size);

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

/**
 * End doxygen group MemoryTest.
 * @}
 */
