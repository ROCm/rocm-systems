/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

#if HT_AMD && !defined(_WIN32)

namespace {

constexpr size_t kKiB = 1024;
constexpr size_t kExternalBytes = 64 * kKiB;

// Buffer views of one external allocation: the inner view sits inside the outer view.
constexpr size_t kOuterOffset = 0;
constexpr size_t kOuterSize = 32 * kKiB;
constexpr size_t kInnerOffset = 8 * kKiB;
constexpr size_t kInnerSize = 8 * kKiB;

uint8_t PatternByte(size_t i) { return static_cast<uint8_t>(i * 7 + 3); }

// Imports a dmabuf of a device allocation as external memory and creates the nested buffer views.
// The views are separate memory objects that overlap in the runtime's global memory-object map.
class NestedViews {
 public:
  NestedViews() = default;
  NestedViews(const NestedViews&) = delete;
  NestedViews& operator=(const NestedViews&) = delete;
  ~NestedViews() {
    if (ext_mem_ != nullptr) {
      static_cast<void>(hipDestroyExternalMemory(ext_mem_));
    }
  }

  // Returns false if the device cannot export or import the allocation.
  bool Create(void* device_alloc) {
    int fd = -1;
    hipError_t err = hipMemGetHandleForAddressRange(&fd, reinterpret_cast<hipDeviceptr_t>(device_alloc),
                                                    kExternalBytes, hipMemRangeHandleTypeDmaBufFd, 0);
    if (err == hipErrorNotSupported) {
      return false;
    }
    HIP_CHECK(err);

    hipExternalMemoryHandleDesc handle_desc{};
    handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    handle_desc.handle.fd = fd;
    handle_desc.size = kExternalBytes;
    err = hipImportExternalMemory(&ext_mem_, &handle_desc);
    if (err == hipErrorNotSupported) {
      return false;
    }
    HIP_CHECK(err);

    hipExternalMemoryBufferDesc outer_desc{};
    outer_desc.offset = kOuterOffset;
    outer_desc.size = kOuterSize;
    HIP_CHECK(hipExternalMemoryGetMappedBuffer(&outer_, ext_mem_, &outer_desc));
    hipExternalMemoryBufferDesc inner_desc{};
    inner_desc.offset = kInnerOffset;
    inner_desc.size = kInnerSize;
    HIP_CHECK(hipExternalMemoryGetMappedBuffer(&inner_, ext_mem_, &inner_desc));
    return true;
  }

  uint8_t* outer() const { return static_cast<uint8_t*>(outer_); }
  uint8_t* inner() const { return static_cast<uint8_t*>(inner_); }

 private:
  hipExternalMemory_t ext_mem_ = nullptr;
  void* outer_ = nullptr;
  void* inner_ = nullptr;
};

struct Copy {
  void* src;
  size_t dst_offset;
  size_t size;
};

}  // namespace

/**
 * Test Description
 * ------------------------
 * - The global memory-object map can hold nested entries: buffer views created by
 * hipExternalMemoryGetMappedBuffer of one external allocation. A pointer inside the inner view
 * belongs to the inner view, so a copy that starts there and runs past the inner view's end is out
 * of bounds, even though it still lies inside the outer view.
 * - Verifies that hipMemcpyBatchAsync reports that error whichever order the entries are in, and
 * that valid copies from both views succeed and move the right data. The result must not depend on
 * whether an earlier entry of the batch resolved to the outer view.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync_NestedBufferViews.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 *  - Linux, with dmabuf export and external memory import support
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_NestedBufferViews) {
  HIP_CHECK(hipSetDevice(0));

  LinearAllocGuard<uint8_t> external(LinearAllocs::hipMalloc, kExternalBytes);
  std::vector<uint8_t> pattern(kExternalBytes);
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = PatternByte(i);
  }
  HIP_CHECK(hipMemcpy(external.ptr(), pattern.data(), kExternalBytes, hipMemcpyHostToDevice));

  NestedViews views;
  if (!views.Create(external.ptr())) {
    HIP_SKIP_TEST("Skipping because dmabuf export or external memory import is not supported");
  }
  REQUIRE(views.inner() == views.outer() + (kInnerOffset - kOuterOffset));

  LinearAllocGuard<uint8_t> dst(LinearAllocs::hipMalloc, kExternalBytes);
  HIP_CHECK(hipMemset(dst.ptr(), 0, kExternalBytes));
  StreamGuard stream_guard(Streams::created);

  // Entirely inside the outer view, before the inner view starts.
  const Copy in_outer{views.outer() + kOuterOffset, 0, 4 * kKiB};
  // Entirely inside the inner view.
  const Copy in_inner{views.inner() + 1 * kKiB, 8 * kKiB, 2 * kKiB};
  // Starts inside the inner view and runs 2 KiB past its end, still inside the outer view.
  const Copy past_inner{views.inner() + 6 * kKiB, 16 * kKiB, 4 * kKiB};

  const auto run = [&](const std::vector<Copy>& copies, size_t* fail_idx) {
    std::vector<void*> dsts;
    std::vector<void*> srcs;
    std::vector<size_t> sizes;
    for (const Copy& c : copies) {
      srcs.push_back(c.src);
      dsts.push_back(dst.ptr() + c.dst_offset);
      sizes.push_back(c.size);
    }
    *fail_idx = ~size_t{0};
    return hipMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), copies.size(), nullptr,
                               nullptr, 0, fail_idx, stream_guard.stream());
  };

  // The expected content of the destination for a copy: the bytes of the external allocation that
  // the source pointer addresses.
  const auto expect_copied = [&](const Copy& c) {
    const size_t ext_offset = static_cast<size_t>(static_cast<uint8_t*>(c.src) - views.outer()) +
                              kOuterOffset;
    std::vector<uint8_t> got(c.size);
    HIP_CHECK(hipMemcpy(got.data(), dst.ptr() + c.dst_offset, c.size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < c.size; ++i) {
      INFO("dst offset " << c.dst_offset << " byte " << i);
      REQUIRE(got[i] == PatternByte(ext_offset + i));
    }
  };

  // The rejected batches below set the runtime's last error on purpose; consume it so the test leaves none behind.
  const auto consume_expected_error = [] {
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
  };

  SECTION("Copy past the inner view alone is rejected") {
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(run({past_inner}, &fail_idx), hipErrorInvalidValue);
    consume_expected_error();
    REQUIRE(fail_idx == 0);
  }

  SECTION("Copies inside each view succeed, in either order") {
    const bool outer_first = GENERATE(true, false);
    const std::vector<Copy> copies = outer_first ? std::vector<Copy>{in_outer, in_inner}
                                                 : std::vector<Copy>{in_inner, in_outer};
    size_t fail_idx = 0;
    HIP_CHECK(run(copies, &fail_idx));
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
    expect_copied(in_outer);
    expect_copied(in_inner);
  }

  SECTION("Copy past the inner view is rejected after an entry in the outer view") {
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(run({in_outer, past_inner}, &fail_idx), hipErrorInvalidValue);
    consume_expected_error();
    REQUIRE(fail_idx == 1);
  }

  SECTION("Copy past the inner view is rejected before an entry in the outer view") {
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(run({past_inner, in_outer}, &fail_idx), hipErrorInvalidValue);
    consume_expected_error();
    REQUIRE(fail_idx == 0);
  }

  SECTION("Copy past the inner view is rejected after an entry in the inner view") {
    size_t fail_idx = 0;
    HIP_CHECK_ERROR(run({in_inner, past_inner}, &fail_idx), hipErrorInvalidValue);
    consume_expected_error();
    REQUIRE(fail_idx == 1);
  }
}

#endif  // HT_AMD && !defined(_WIN32)
