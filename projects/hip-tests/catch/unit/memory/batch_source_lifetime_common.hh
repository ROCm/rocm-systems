/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__) && defined(__HIP_PLATFORM_AMD__)
#include <dlfcn.h>
#include <filesystem>
#endif

#include <hip_test_common.hh>
#if defined(__linux__) && defined(__HIP_PLATFORM_AMD__)
#include <hip_test_process.hh>
#endif
#include <resource_guards.hh>
#include "stream_copy_gate.hh"

namespace HipTest {
inline void checkBatchSourceLifetime(LinearAllocs allocation, LinearAllocs destination_allocation,
                                     size_t size, bool extended_api, unsigned int engine_flags = 0,
                                     bool three_dimensional = false) {
  static_cast<void>(engine_flags);
  LinearAllocGuard<uint8_t> source(allocation, size);
  LinearAllocGuard<uint8_t> destination(destination_allocation, size);
  std::vector<uint8_t> expected(size);
  for (size_t i = 0; i < size; ++i) {
    expected[i] = static_cast<uint8_t>(i * 131 + 17);
  }
  std::copy(expected.begin(), expected.end(), source.host_ptr());
  StreamGuard stream(Streams::created);
  HIP_CHECK(hipDeviceSynchronize());
  void* src = source.host_ptr();
  void* dst = destination.ptr();
  hipMemcpyAttributes attributes{};
  attributes.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
#if defined(__HIP_PLATFORM_AMD__)
  hipExtMemcpyAttributes extended{};
  extended.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  extended.flags = engine_flags;
#else
  REQUIRE_FALSE(extended_api);
#endif
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  StreamCopyGate gate(stream.stream());
  hipError_t result = hipSuccess;
  if (three_dimensional) {
    hipMemcpy3DBatchOp op{};
    op.src.type = hipMemcpyOperandTypePointer;
    op.src.op.ptr.ptr = src;
    op.src.op.ptr.rowLength = size / 8;
    op.src.op.ptr.layerHeight = 4;
    op.dst.type = hipMemcpyOperandTypePointer;
    op.dst.op.ptr.ptr = dst;
    op.dst.op.ptr.rowLength = size / 8;
    op.dst.op.ptr.layerHeight = 4;
    op.extent = make_hipExtent(size / 8, 4, 2);
    op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
    result = hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream());
#if defined(__HIP_PLATFORM_AMD__)
  } else if (extended_api) {
    result = hipExtMemcpyBatchAsync(&dst, &src, &size, nullptr, nullptr, nullptr, 1, &extended,
                                    &attrs_idx, 1, stream.stream());
#endif
  } else {
    result = hipMemcpyBatchAsync(&dst, &src, &size, 1, &attributes, &attrs_idx, 1, &fail_idx,
                                 stream.stream());
  }
  std::fill_n(source.host_ptr(), size, uint8_t{0xAB});
  gate.release();
  HIP_CHECK(result);
  // Pageable host-to-host entries may finish synchronously.
  if (allocation != LinearAllocs::malloc || destination_allocation == LinearAllocs::hipMalloc) {
    REQUIRE_FALSE(gate.timed_out());
  }
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  std::vector<uint8_t> actual(size);
  HIP_CHECK(hipMemcpy(actual.data(), destination.ptr(), size, hipMemcpyDeviceToHost));
  REQUIRE(actual == expected);
}

// Interleave retained host reads with stream-ordered device producers. Odd
// sizes and distinct data detect overlapping offsets in a shared snapshot.
inline void checkBatchMixedSourceOrders(bool extended_api, unsigned int engine_flags = 0) {
  static_cast<void>(engine_flags);
  constexpr size_t kCount = 5;
  constexpr size_t kStride = 128;
  constexpr size_t kBytes = kCount * kStride;
  std::array<size_t, kCount> sizes{13, 29, 47, 3, 65};
  LinearAllocGuard<uint8_t> host_source(LinearAllocs::hipHostMalloc, kBytes);
  LinearAllocGuard<uint8_t> stream_source(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kBytes);
  std::vector<uint8_t> expected(kBytes, 0xF8);
  std::array<void*, kCount> src_ptrs{};
  std::array<void*, kCount> dst_ptrs{};
  std::array<hipMemcpyAttributes, kCount> attributes{};
#if defined(__HIP_PLATFORM_AMD__)
  std::array<hipExtMemcpyAttributes, kCount> extended{};
#else
  REQUIRE_FALSE(extended_api);
#endif
  std::array<size_t, kCount> indices{};
  for (size_t i = 0; i < kCount; ++i) {
    const size_t offset = i * kStride + 1;
    const bool during_call = i % 2 == 0;
    src_ptrs[i] = (during_call ? host_source.host_ptr() : stream_source.ptr()) + offset;
    dst_ptrs[i] = destination.ptr() + offset;
    indices[i] = i;
    attributes[i].srcAccessOrder =
        during_call ? hipMemcpySrcAccessOrderDuringApiCall : hipMemcpySrcAccessOrderStream;
#if defined(__HIP_PLATFORM_AMD__)
    extended[i].srcAccessOrder = attributes[i].srcAccessOrder;
    extended[i].flags = engine_flags;
#endif
    for (size_t j = 0; j < sizes[i]; ++j) {
      const uint8_t value =
          during_call ? static_cast<uint8_t>(17 + 31 * i + j) : static_cast<uint8_t>(0x60 + i);
      host_source.host_ptr()[offset + j] = value;
      expected[offset + j] = value;
    }
  }
  StreamGuard stream(Streams::created);
  HIP_CHECK(hipMemset(destination.ptr(), 0xF8, kBytes));
  HIP_CHECK(hipDeviceSynchronize());
  StreamCopyGate gate(stream.stream());
  for (size_t i : {size_t{1}, size_t{3}}) {
    HIP_CHECK(hipMemsetAsync(src_ptrs[i], 0x60 + i, sizes[i], stream.stream()));
  }
  size_t fail_idx = 0;
  hipError_t result;
#if defined(__HIP_PLATFORM_AMD__)
  if (extended_api) {
    result = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), nullptr,
                                    nullptr, nullptr, kCount, extended.data(), indices.data(),
                                    kCount, stream.stream());
  } else
#endif
  {
    result =
        hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                            attributes.data(), indices.data(), kCount, &fail_idx, stream.stream());
  }
  std::fill_n(host_source.host_ptr(), kBytes, uint8_t{0xEE});
  gate.release();
  HIP_CHECK(result);
  REQUIRE_FALSE(gate.timed_out());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  std::vector<uint8_t> actual(kBytes);
  HIP_CHECK(hipMemcpy(actual.data(), destination.ptr(), kBytes, hipMemcpyDeviceToHost));
  REQUIRE(actual == expected);
}

class ScopedBatchDevice {
 public:
  explicit ScopedBatchDevice(int device) {
    HIP_CHECK(hipGetDevice(&original_));
    HIP_CHECK(hipSetDevice(device));
  }
  ~ScopedBatchDevice() { static_cast<void>(hipSetDevice(original_)); }
  ScopedBatchDevice(const ScopedBatchDevice&) = delete;
  ScopedBatchDevice& operator=(const ScopedBatchDevice&) = delete;
  ScopedBatchDevice(ScopedBatchDevice&&) = delete;
  ScopedBatchDevice& operator=(ScopedBatchDevice&&) = delete;

 private:
  int original_ = 0;
};


// Repeated read-only sources need not consume one full extra pinned copy per
// logical entry. The destinations are disjoint, and the API must finish its
// source reads or retain their data before returning even for a large batch.
inline void checkBatchLargeSharedSource(bool extended_api, bool three_dimensional,
                                        bool shared_source = true) {
  constexpr size_t kCount = 5;
  constexpr size_t kBytes = 64 * 1024 * 1024;
  constexpr size_t kWidth = 4096;
  constexpr size_t kHeight = 256;
  constexpr size_t kDepth = 64;
  const size_t source_bytes = (shared_source ? 1 : kCount) * kBytes;
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipHostMalloc, source_bytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kCount * kBytes);
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    expected[i] = static_cast<uint8_t>(i * 131 + 17);
  }
  for (size_t i = 0; i < (shared_source ? 1 : kCount); ++i) {
    for (size_t byte = 0; byte < kBytes; ++byte) {
      source.host_ptr()[i * kBytes + byte] = expected[byte] ^ static_cast<uint8_t>(i * 29);
    }
  }
  std::array<void*, kCount> src_ptrs{};
  std::array<void*, kCount> dst_ptrs{};
  std::array<size_t, kCount> sizes{};
  std::array<hipMemcpy3DBatchOp, kCount> operations{};
  for (size_t i = 0; i < kCount; ++i) {
    src_ptrs[i] = source.host_ptr() + (shared_source ? 0 : i * kBytes);
    dst_ptrs[i] = destination.ptr() + i * kBytes;
    sizes[i] = kBytes;
    operations[i].src.type = hipMemcpyOperandTypePointer;
    operations[i].src.op.ptr.ptr = src_ptrs[i];
    operations[i].src.op.ptr.rowLength = kWidth;
    operations[i].src.op.ptr.layerHeight = kHeight;
    operations[i].dst.type = hipMemcpyOperandTypePointer;
    operations[i].dst.op.ptr.ptr = dst_ptrs[i];
    operations[i].dst.op.ptr.rowLength = kWidth;
    operations[i].dst.op.ptr.layerHeight = kHeight;
    operations[i].extent = make_hipExtent(kWidth, kHeight, kDepth);
    operations[i].srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  }
  StreamGuard stream(Streams::created);
  HIP_CHECK(hipDeviceSynchronize());
  hipMemcpyAttributes attributes{};
  attributes.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
#if defined(__HIP_PLATFORM_AMD__)
  hipExtMemcpyAttributes extended{};
  extended.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
#else
  REQUIRE_FALSE(extended_api);
#endif
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  // Leave headroom for the required CPU capture of up to 320 MiB.
  StreamCopyGate gate(stream.stream(), std::chrono::seconds(15));
  hipError_t result = hipSuccess;
  if (three_dimensional) {
    result = hipMemcpy3DBatchAsync(kCount, operations.data(), &fail_idx, 0, stream.stream());
#if defined(__HIP_PLATFORM_AMD__)
  } else if (extended_api) {
    result =
        hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), nullptr, nullptr,
                               nullptr, kCount, &extended, &attrs_idx, 1, stream.stream());
#endif
  } else {
    result = hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), kCount,
                                 &attributes, &attrs_idx, 1, &fail_idx, stream.stream());
  }
  std::fill_n(source.host_ptr(), source_bytes, uint8_t{0xAB});
  gate.release();
  HIP_CHECK(result);
  REQUIRE_FALSE(gate.timed_out());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  std::vector<uint8_t> actual(kBytes);
  for (size_t i = 0; i < kCount; ++i) {
    HIP_CHECK(hipMemcpy(actual.data(), dst_ptrs[i], kBytes, hipMemcpyDeviceToHost));
    if (!shared_source) {
      for (size_t byte = 0; byte < kBytes; ++byte) {
        expected[byte] = static_cast<uint8_t>(byte * 131 + 17) ^ static_cast<uint8_t>(i * 29);
      }
    }
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    INFO("entry " << i << ", first mismatch byte "
                  << std::distance(actual.begin(), mismatch.first));
    REQUIRE(mismatch.first == actual.end());
  }
}

// Each call owns a different version of the same source until its GPU reads
// finish, even when many commands wait behind prior work on the same stream.
inline void checkQueuedBatchSources(bool extended_api, bool three_dimensional) {
  constexpr size_t kCount = 8;
  constexpr size_t kWidth = 64;
  constexpr size_t kHeight = 8;
  constexpr size_t kDepth = 8;
  constexpr size_t kBytes = kWidth * kHeight * kDepth;
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipHostMalloc, kBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kCount * kBytes);
  StreamGuard stream(Streams::created);
  HIP_CHECK(hipDeviceSynchronize());
  StreamCopyGate gate(stream.stream());
  hipMemcpyAttributes attributes{};
  attributes.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
#if defined(__HIP_PLATFORM_AMD__)
  hipExtMemcpyAttributes extended{};
  extended.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
#else
  REQUIRE_FALSE(extended_api);
#endif
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  for (size_t i = 0; i < kCount; ++i) {
    std::fill_n(source.host_ptr(), kBytes, static_cast<uint8_t>(i + 17));
    void* src = source.host_ptr();
    void* dst = destination.ptr() + i * kBytes;
    size_t bytes = kBytes;
    if (three_dimensional) {
      hipMemcpy3DBatchOp operation{};
      operation.src.type = hipMemcpyOperandTypePointer;
      operation.src.op.ptr.ptr = src;
      operation.src.op.ptr.rowLength = kWidth;
      operation.src.op.ptr.layerHeight = kHeight;
      operation.dst.type = hipMemcpyOperandTypePointer;
      operation.dst.op.ptr.ptr = dst;
      operation.dst.op.ptr.rowLength = kWidth;
      operation.dst.op.ptr.layerHeight = kHeight;
      operation.extent = make_hipExtent(kWidth, kHeight, kDepth);
      operation.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
      HIP_CHECK(hipMemcpy3DBatchAsync(1, &operation, &fail_idx, 0, stream.stream()));
#if defined(__HIP_PLATFORM_AMD__)
    } else if (extended_api) {
      HIP_CHECK(hipExtMemcpyBatchAsync(&dst, &src, &bytes, nullptr, nullptr, nullptr, 1, &extended,
                                       &attrs_idx, 1, stream.stream()));
#endif
    } else {
      HIP_CHECK(hipMemcpyBatchAsync(&dst, &src, &bytes, 1, &attributes, &attrs_idx, 1, &fail_idx,
                                    stream.stream()));
    }
  }
  std::fill_n(source.host_ptr(), kBytes, uint8_t{0xAB});
  gate.release();
  REQUIRE_FALSE(gate.timed_out());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  std::vector<uint8_t> actual(kCount * kBytes);
  HIP_CHECK(hipMemcpy(actual.data(), destination.ptr(), actual.size(), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < actual.size(); ++i) {
    REQUIRE(actual[i] == static_cast<uint8_t>(i / kBytes + 17));
  }
}


#if defined(__linux__) && defined(__HIP_PLATFORM_AMD__)
// Re-exec with a test-only allocator shim so runtime initialization is unaffected.
inline void checkBatchAllocationFailure(const char* test_name, bool extended_api,
                                        bool three_dimensional) {
  using Arm = void (*)(size_t);
  using Count = unsigned (*)();
  auto arm = reinterpret_cast<Arm>(dlsym(RTLD_DEFAULT, "hip_test_fail_pool_allocation"));
  auto count = reinterpret_cast<Count>(dlsym(RTLD_DEFAULT, "hip_test_pool_allocation_failures"));
  if (arm == nullptr || count == nullptr) {
    constexpr const char* child_marker = "HIP_TEST_BATCH_ALLOC_FAILURE_CHILD";
    INFO("allocator failure helper must load in the isolated test child");
    REQUIRE(std::getenv(child_marker) == nullptr);
    const auto library =
        std::filesystem::path(getSelfExePath()).parent_path() / "libhip_batch_alloc_failure.so";
    REQUIRE(std::filesystem::exists(library));
    hip::SpawnProc child(getSelfExePath(), true, true);
    const char* preload = std::getenv("LD_PRELOAD");
    child.setEnv("LD_PRELOAD", library.string() + (preload ? ":" + std::string(preload) : ""));
    child.setEnv(child_marker, "1");
    const int result = child.run(test_name);
    INFO(child.getOutput());
    REQUIRE(result == 0);
    return;
  }
  ScopedBatchDevice device(0);
  constexpr size_t kWidth = 1024 + 48;
  constexpr size_t kHeight = 1024 - 1;
  const size_t bytes = three_dimensional ? kWidth * kHeight : (1u << 20) + 48;
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipHostMalloc, bytes + 4096);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, bytes + 4096);
  LinearAllocGuard<uint8_t> device_source(LinearAllocs::hipMalloc, 4096);
  LinearAllocGuard<uint8_t> device_destination(LinearAllocs::hipMalloc, 4096);
  for (size_t i = 0; i < bytes; ++i) source.host_ptr()[i] = static_cast<uint8_t>(i * 31 + 5);
  HIP_CHECK(hipMemset(device_source.ptr(), 0x5A, 4096));
  HIP_CHECK(hipMemset(destination.ptr(), 0xC7, bytes));
  HIP_CHECK(hipMemset(device_destination.ptr(), 0xC7, 4096));
  LinearAllocGuard<uint8_t> host_source(LinearAllocs::malloc, 64);
  LinearAllocGuard<uint8_t> host_destination(LinearAllocs::malloc, 64);
  std::fill_n(host_source.host_ptr(), 64, uint8_t{0x11});
  std::fill_n(host_destination.host_ptr(), 64, uint8_t{0xC7});
  StreamGuard stream(Streams::created);
  HIP_CHECK(hipDeviceSynchronize());
  void* sources[3] = {host_source.host_ptr(), source.host_ptr(), device_source.ptr()};
  void* destinations[3] = {host_destination.host_ptr(), destination.ptr(),
                           device_destination.ptr()};
  size_t sizes[3] = {size_t{64}, bytes, size_t{4096}};
  size_t indices[3] = {0, 1, 2};
  size_t fail_index = 0;
  hipMemcpyAttributes attributes[3]{};
  hipExtMemcpyAttributes extended[3]{};
  for (size_t i = 0; i < 3; ++i) {
    attributes[i].srcAccessOrder =
        (i == 1) ? hipMemcpySrcAccessOrderDuringApiCall : hipMemcpySrcAccessOrderStream;
    attributes[i].flags = hipMemcpyFlagPreferOverlapWithCompute;
    extended[i].srcAccessOrder = attributes[i].srcAccessOrder;
    extended[i].flags = attributes[i].flags;
  }
  StreamCopyGate gate(stream.stream());
  arm(bytes);
  hipError_t result = hipSuccess;
  if (three_dimensional) {
    hipMemcpy3DBatchOp operations[3]{};
    for (size_t i = 0; i < 3; ++i) {
      operations[i].src.type = hipMemcpyOperandTypePointer;
      operations[i].src.op.ptr.ptr = sources[i];
      operations[i].src.op.ptr.rowLength = i == 1 ? kWidth : 64;
      operations[i].src.op.ptr.layerHeight = i == 0 ? 1 : (i == 1 ? kHeight : 8);
      operations[i].dst.type = hipMemcpyOperandTypePointer;
      operations[i].dst.op.ptr.ptr = destinations[i];
      operations[i].dst.op.ptr.rowLength = operations[i].src.op.ptr.rowLength;
      operations[i].dst.op.ptr.layerHeight = operations[i].src.op.ptr.layerHeight;
      operations[i].extent =
          i == 0 ? make_hipExtent(64, 1, 1)
                 : (i == 1 ? make_hipExtent(kWidth, kHeight, 1) : make_hipExtent(64, 8, 8));
      operations[i].srcAccessOrder = attributes[i].srcAccessOrder;
    }
    result = hipMemcpy3DBatchAsync(3, operations, &fail_index, 0, stream.stream());
#if defined(__HIP_PLATFORM_AMD__)
  } else if (extended_api) {
    result = hipExtMemcpyBatchAsync(destinations, sources, sizes, nullptr, nullptr, nullptr, 3,
                                    extended, indices, 3, stream.stream());
#endif
  } else {
    result = hipMemcpyBatchAsync(destinations, sources, sizes, 3, attributes, indices, 3,
                                 &fail_index, stream.stream());
  }
  arm(0);
  std::fill_n(source.host_ptr(), bytes, uint8_t{0xAB});
  gate.release();
  HIP_CHECK_ERROR(result, hipErrorOutOfMemory);
  static_cast<void>(hipGetLastError());
  REQUIRE_FALSE(gate.timed_out());
  REQUIRE(count() > 0);
  REQUIRE(std::all_of(host_destination.host_ptr(), host_destination.host_ptr() + 64,
                      [](uint8_t x) { return x == 0xC7; }));
  if (!extended_api) REQUIRE(fail_index == 1);
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  std::vector<uint8_t> actual(bytes);
  HIP_CHECK(hipMemcpy(actual.data(), destination.ptr(), bytes, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint8_t x) { return x == 0xC7; }));
  actual.resize(4096);
  HIP_CHECK(
      hipMemcpy(actual.data(), device_destination.ptr(), actual.size(), hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint8_t x) { return x == 0xC7; }));
}
#endif

}  // namespace HipTest
