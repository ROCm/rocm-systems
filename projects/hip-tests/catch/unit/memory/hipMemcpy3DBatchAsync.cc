/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

namespace {
hipMemcpy3DBatchOp MakePointerToPointerOp(void* source, void* destination, hipExtent extent) {
  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypePointer;
  op.src.op.ptr.ptr = source;
  op.dst.type = hipMemcpyOperandTypePointer;
  op.dst.op.ptr.ptr = destination;
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;
  return op;
}

hipMemcpy3DBatchOp MakeArrayToArrayOp(hipArray_t source, hipArray_t destination, hipExtent extent) {
  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypeArray;
  op.src.op.array.array = source;
  op.dst.type = hipMemcpyOperandTypeArray;
  op.dst.op.array.array = destination;
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;
  return op;
}

struct SnapshotGate {
  std::atomic<bool> open{false};
  std::atomic<bool> timed_out{false};
};

void HoldSnapshotCopy(void* data) {
  auto& gate = *static_cast<SnapshotGate*>(data);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!gate.open.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      gate.timed_out.store(true, std::memory_order_release);
      return;
    }
    std::this_thread::yield();
  }
}

struct OpenSnapshotGateOnExit {
  SnapshotGate& gate;
  ~OpenSnapshotGateOnExit() { gate.open.store(true, std::memory_order_release); }
};
}  // namespace

/**
 * @addtogroup hipMemcpy3DBatchAsync hipMemcpy3DBatchAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy3DBatchAsync(size_t numOps, hipMemcpy3DBatchOp* opList,
 * size_t* failIdx, unsigned long long flags, hipStream_t stream)`
 */

/**
 * Test Description
 * ------------------------
 * - A pointer-to-pointer copy of a 4x3x2 uint32 volume whose extent width counts bytes, with
 *   tightly packed layers or with source and destination layer heights of 4 and 5 rows: after
 *   stream sync, the destination layers match the source layers and the destination layer padding
 *   is unchanged.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_PointerToPointer_MatchesSourceAfterSync) {
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  const size_t layer_padding = GENERATE(0, 1);
  INFO("layer_padding: " << layer_padding);
  const size_t source_layer_height = kHeight + layer_padding;
  const size_t destination_layer_height = kHeight + 2 * layer_padding;
  const size_t source_count = kWidth * source_layer_height * kDepth;
  const size_t destination_count = kWidth * destination_layer_height * kDepth;
  const size_t source_bytes = source_count * sizeof(uint32_t);
  const size_t destination_bytes = destination_count * sizeof(uint32_t);
  std::vector<uint32_t> pattern(source_count);
  for (uint32_t i = 0; i < source_count; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  std::vector<uint32_t> expected(destination_count, 0xFFFFFFFFu);
  for (size_t z = 0; z < kDepth; ++z) {
    std::copy_n(pattern.begin() + z * kWidth * source_layer_height, kWidth * kHeight,
                expected.begin() + z * kWidth * destination_layer_height);
  }
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint32_t> source(LinearAllocs::hipMalloc, source_bytes);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, destination_bytes);
  HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), source_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, destination_bytes));

  hipMemcpy3DBatchOp op = MakePointerToPointerOp(
      source.ptr(), destination.ptr(), make_hipExtent(kWidth * sizeof(uint32_t), kHeight, kDepth));
  op.src.op.ptr.layerHeight = source_layer_height;
  op.dst.op.ptr.layerHeight = destination_layer_height;

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint32_t> copied(destination_count);
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), destination_bytes, hipMemcpyDeviceToHost));
  REQUIRE(copied == expected);
}

/**
 * Test Description
 * ------------------------
 * - DuringApiCall also finishes accessing device, managed and array sources before the caller
 *   submits an overwrite on another stream. A gate holds any queued source read until the
 *   overwrite finishes; an independent releaser lets synchronous copies make progress.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_DuringApiCall_NonHostSources) {
  enum class SourceKind { Device, Managed, Array };
  const auto source_kind = GENERATE(SourceKind::Device, SourceKind::Managed, SourceKind::Array);
  const bool single_operation = GENERATE(false, true);
  INFO("source kind: " << static_cast<int>(source_kind)
                       << ", single operation: " << single_operation);
  if (source_kind == SourceKind::Array) {
    CHECK_IMAGE_SUPPORT
  }
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kCount = kWidth * kHeight * kDepth;
  constexpr size_t kBytes = kCount * sizeof(uint32_t);
  std::vector<uint32_t> pattern(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  SnapshotGate gate;
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard overwrite_stream(Streams::withFlags, hipStreamNonBlocking);
  LinearAllocGuard<uint32_t> source(
      source_kind == SourceKind::Managed ? LinearAllocs::hipMallocManaged : LinearAllocs::hipMalloc,
      kBytes);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, kBytes);
  std::unique_ptr<ArrayAllocGuard<uint32_t>> array;
  OpenSnapshotGateOnExit release{gate};
  hipMemcpy3DParms fill{};
  hipMemcpy3DBatchOp op = MakePointerToPointerOp(
      source.ptr(), destination.ptr(), make_hipExtent(kWidth * sizeof(uint32_t), kHeight, kDepth));
  if (source_kind == SourceKind::Array) {
    array = std::make_unique<ArrayAllocGuard<uint32_t>>(extent);
    fill.srcPtr = make_hipPitchedPtr(pattern.data(), kWidth * sizeof(uint32_t), kWidth, kHeight);
    fill.dstArray = array->ptr();
    fill.extent = extent;
    fill.kind = hipMemcpyHostToDevice;
    HIP_CHECK(hipMemcpy3D(&fill));
    op.src.type = hipMemcpyOperandTypeArray;
    op.src.op.array.array = array->ptr();
    op.src.op.array.offset = {0, 0, 0};
    op.extent = extent;
  } else {
    HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), kBytes, hipMemcpyHostToDevice));
  }
  std::vector<uint32_t> overwrite(kCount, 0xABABABABu);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipLaunchHostFunc(stream.stream(), HoldSnapshotCopy, &gate));
  std::atomic<bool> api_returned{false};
  std::atomic<bool> overwrite_finished{false};
  bool overwrite_timed_out = false;
  std::thread releaser([&] {
    // If the API returns early, keep its stream blocked until the source overwrite completes.
    // Otherwise release it after a bounded wait so a synchronous implementation can return.
    const auto return_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!api_returned.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < return_deadline) {
      std::this_thread::yield();
    }
    if (api_returned.load(std::memory_order_acquire)) {
      const auto overwrite_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!overwrite_finished.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < overwrite_deadline) {
        std::this_thread::yield();
      }
      overwrite_timed_out = !overwrite_finished.load(std::memory_order_acquire);
    }
    gate.open.store(true, std::memory_order_release);
  });
  op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  // Defer assertions until the releaser is joined, including on an API error.
  const hipError_t copy_status = single_operation
                                     ? hipMemcpy3DWithAttributesAsync(&op, 0, stream.stream())
                                     : hipMemcpy3DBatchAsync(1, &op, nullptr, 0, stream.stream());
  api_returned.store(true, std::memory_order_release);
  hipError_t overwrite_status = hipSuccess;
  if (source_kind == SourceKind::Array) {
    fill.srcPtr.ptr = overwrite.data();
    overwrite_status = hipMemcpy3DAsync(&fill, overwrite_stream.stream());
  } else if (source_kind == SourceKind::Managed) {
    std::copy(overwrite.begin(), overwrite.end(), source.host_ptr());
  } else {
    overwrite_status = hipMemcpyAsync(source.ptr(), overwrite.data(), kBytes, hipMemcpyHostToDevice,
                                      overwrite_stream.stream());
  }
  const hipError_t overwrite_sync_status = hipStreamSynchronize(overwrite_stream.stream());
  overwrite_finished.store(true, std::memory_order_release);
  releaser.join();
  HIP_CHECK(copy_status);
  HIP_CHECK(overwrite_status);
  HIP_CHECK(overwrite_sync_status);
  REQUIRE_FALSE(overwrite_timed_out);
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  REQUIRE_FALSE(gate.timed_out.load(std::memory_order_acquire));
  std::vector<uint32_t> copied(kCount);
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), kBytes, hipMemcpyDeviceToHost));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - DuringApiCall rejects and invalidates capture before accessing the source or waiting on the
 *   captured stream.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_DuringApiCall_CaptureIsUnsupported) {
  const bool single_operation = GENERATE(false, true);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, 8);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, 8);
  hipMemcpy3DBatchOp op =
      MakePointerToPointerOp(source.ptr(), destination.ptr(), make_hipExtent(4, 2, 1));
  op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  HIP_CHECK(hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeRelaxed));
  for (hipError_t expected : {hipErrorStreamCaptureUnsupported, hipErrorStreamCaptureInvalidated}) {
    if (single_operation) {
      HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(&op, 0, stream.stream()), expected);
    } else {
      size_t fail_idx = 0;
      HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()), expected);
      REQUIRE(fail_idx == SIZE_MAX);
    }
  }
  hipGraph_t graph = nullptr;
  HIP_CHECK_ERROR(hipStreamEndCapture(stream.stream(), &graph), hipErrorStreamCaptureInvalidated);
  REQUIRE(graph == nullptr);
}

/**
 * Test Description
 * ------------------------
 * - A pointer-to-array copy of a 4x3x2 uint32 volume whose extent width counts array elements:
 *   after stream sync, the array matches the source.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_PointerToArray_MatchesSourceAfterSync) {
  CHECK_IMAGE_SUPPORT

  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kElementCount = kWidth * kHeight * kDepth;
  constexpr size_t kRowBytes = kWidth * sizeof(uint32_t);
  constexpr size_t kVolumeBytes = kElementCount * sizeof(uint32_t);
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  std::vector<uint32_t> pattern(kElementCount);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  const std::vector<uint32_t> untouched(kElementCount, 0xFFFFFFFFu);
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint32_t> source(LinearAllocs::hipMalloc, kVolumeBytes);
  ArrayAllocGuard<uint32_t> destination(extent);
  HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), kVolumeBytes, hipMemcpyHostToDevice));
  hipMemcpy3DParms fill{};
  fill.srcPtr =
      make_hipPitchedPtr(const_cast<uint32_t*>(untouched.data()), kRowBytes, kRowBytes, kHeight);
  fill.dstArray = destination.ptr();
  fill.extent = extent;
  fill.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fill));

  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypePointer;
  op.src.op.ptr.ptr = source.ptr();
  op.dst.type = hipMemcpyOperandTypeArray;
  op.dst.op.array.array = destination.ptr();
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint32_t> copied(kElementCount);
  hipMemcpy3DParms read{};
  read.srcArray = destination.ptr();
  read.dstPtr = make_hipPitchedPtr(copied.data(), kRowBytes, kRowBytes, kHeight);
  read.extent = extent;
  read.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&read));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - An array-to-pointer copy of a 4x3x2 uint32 volume whose extent width counts array elements:
 *   after stream sync, the destination matches the array.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_ArrayToPointer_MatchesSourceAfterSync) {
  CHECK_IMAGE_SUPPORT

  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kElementCount = kWidth * kHeight * kDepth;
  constexpr size_t kRowBytes = kWidth * sizeof(uint32_t);
  constexpr size_t kVolumeBytes = kElementCount * sizeof(uint32_t);
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  std::vector<uint32_t> pattern(kElementCount);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  StreamGuard stream(Streams::created);
  ArrayAllocGuard<uint32_t> source(extent);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, kVolumeBytes);
  hipMemcpy3DParms fill{};
  fill.srcPtr = make_hipPitchedPtr(pattern.data(), kRowBytes, kRowBytes, kHeight);
  fill.dstArray = source.ptr();
  fill.extent = extent;
  fill.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fill));
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, kVolumeBytes));

  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypeArray;
  op.src.op.array.array = source.ptr();
  op.dst.type = hipMemcpyOperandTypePointer;
  op.dst.op.ptr.ptr = destination.ptr();
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint32_t> copied(kElementCount);
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), kVolumeBytes, hipMemcpyDeviceToHost));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - An array-to-array copy of a 4x3x2 uint32 volume whose extent width counts array elements:
 *   after stream sync, the destination array matches the source array.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_ArrayToArray_MatchesSourceAfterSync) {
  CHECK_IMAGE_SUPPORT

  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kElementCount = kWidth * kHeight * kDepth;
  constexpr size_t kRowBytes = kWidth * sizeof(uint32_t);
  const hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);
  std::vector<uint32_t> pattern(kElementCount);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  const std::vector<uint32_t> untouched(kElementCount, 0xFFFFFFFFu);
  StreamGuard stream(Streams::created);
  ArrayAllocGuard<uint32_t> source(extent);
  ArrayAllocGuard<uint32_t> destination(extent);
  hipMemcpy3DParms fill_source{};
  fill_source.srcPtr = make_hipPitchedPtr(pattern.data(), kRowBytes, kRowBytes, kHeight);
  fill_source.dstArray = source.ptr();
  fill_source.extent = extent;
  fill_source.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fill_source));
  hipMemcpy3DParms fill_destination{};
  fill_destination.srcPtr =
      make_hipPitchedPtr(const_cast<uint32_t*>(untouched.data()), kRowBytes, kRowBytes, kHeight);
  fill_destination.dstArray = destination.ptr();
  fill_destination.extent = extent;
  fill_destination.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fill_destination));

  hipMemcpy3DBatchOp op = MakeArrayToArrayOp(source.ptr(), destination.ptr(), extent);

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint32_t> copied(kElementCount);
  hipMemcpy3DParms read{};
  read.srcArray = destination.ptr();
  read.dstPtr = make_hipPitchedPtr(copied.data(), kRowBytes, kRowBytes, kHeight);
  read.extent = extent;
  read.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&read));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - Two copies in one batch, from a buffer of each linear allocation kind to device memory and from
 *   device memory to a buffer of the same kind: after stream sync, both destinations match the
 *   source pattern.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_AllocationKinds_MatchesSourceAfterSync) {
  const auto allocation =
      GENERATE(LinearAllocs::malloc, LinearAllocs::mallocAndRegister, LinearAllocs::hipHostMalloc,
               LinearAllocs::hipMallocManaged, LinearAllocs::hipMalloc);
  INFO("allocation: " << to_string(allocation));
  if (allocation == LinearAllocs::mallocAndRegister) {
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    int host_register = 0;
    HIP_CHECK(
        hipDeviceGetAttribute(&host_register, hipDeviceAttributeHostRegisterSupported, device));
    if (host_register == 0) {
      HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported);
    }
  }
  if (allocation == LinearAllocs::hipMallocManaged) {
    CHECK_MANAGED_MEMORY_SUPPORT
  }

  const std::vector<uint8_t> pattern = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
                                        0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x0F};
  const std::vector<uint8_t> untouched(pattern.size(), 0xFF);
  const hipExtent extent = make_hipExtent(4, 2, 2);
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(allocation, pattern.size());
  LinearAllocGuard<uint8_t> device_destination(LinearAllocs::hipMalloc, pattern.size());
  LinearAllocGuard<uint8_t> device_source(LinearAllocs::hipMalloc, pattern.size());
  LinearAllocGuard<uint8_t> destination(allocation, pattern.size());
  uint8_t* const source_ptr =
      allocation == LinearAllocs::hipMalloc ? source.ptr() : source.host_ptr();
  uint8_t* const destination_ptr =
      allocation == LinearAllocs::hipMalloc ? destination.ptr() : destination.host_ptr();
  HIP_CHECK(hipMemcpy(source_ptr, pattern.data(), pattern.size(), hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(destination_ptr, untouched.data(), pattern.size(), hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(device_source.ptr(), pattern.data(), pattern.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_destination.ptr(), 0xFF, pattern.size()));

  hipMemcpy3DBatchOp ops[] = {
      MakePointerToPointerOp(source_ptr, device_destination.ptr(), extent),
      MakePointerToPointerOp(device_source.ptr(), destination_ptr, extent)};

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(2, ops, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint8_t> device_copied(pattern.size());
  std::vector<uint8_t> copied(pattern.size());
  HIP_CHECK(hipMemcpy(device_copied.data(), device_destination.ptr(), pattern.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(copied.data(), destination_ptr, pattern.size(), hipMemcpyDefault));
  REQUIRE(device_copied == pattern);
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - A tight copy from device 1 memory to device 0 memory on a device 0 stream, with device 0 peer
 *   access to device 1 enabled, for a 7-byte width and a DWORD-aligned 16-byte width: after stream
 *   sync, the destination matches the source.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 * - Multi-device
 * - Peer access supported
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_PeerDeviceCopy_MatchesSourceAfterSync) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  constexpr int kDestinationDevice = 0;
  constexpr int kSourceDevice = 1;
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, kDestinationDevice, kSourceDevice));
  if (can_access_peer == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  const size_t width = GENERATE(7, 16);
  INFO("width: " << width);
  constexpr size_t kHeight = 5;
  constexpr size_t kDepth = 3;
  const size_t kVolumeBytes = width * kHeight * kDepth;
  constexpr int kUntouched = 0xFF;

  int original_device = 0;
  HIP_CHECK(hipGetDevice(&original_device));
  HIP_CHECK(hipSetDevice(kDestinationDevice));
  const hipError_t enable_status = hipDeviceEnablePeerAccess(kSourceDevice, 0);
  if (enable_status != hipErrorPeerAccessAlreadyEnabled) {
    HIP_CHECK(enable_status);
  }
  static_cast<void>(hipGetLastError());

  std::vector<uint8_t> pattern(kVolumeBytes);
  for (size_t byte = 0; byte < kVolumeBytes; ++byte) {
    pattern[byte] = static_cast<uint8_t>(0x21 + byte);
  }

  std::vector<uint8_t> copied(kVolumeBytes);
  {
    HIP_CHECK(hipSetDevice(kSourceDevice));
    LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, kVolumeBytes);
    HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), kVolumeBytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(kDestinationDevice));
    LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kVolumeBytes);
    HIP_CHECK(hipMemset(destination.ptr(), kUntouched, kVolumeBytes));
    StreamGuard stream(Streams::created);

    hipMemcpy3DBatchOp op = MakePointerToPointerOp(source.ptr(), destination.ptr(),
                                                   make_hipExtent(width, kHeight, kDepth));

    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipGetLastError());
    size_t fail_idx = 0;
    HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), kVolumeBytes, hipMemcpyDeviceToHost));
  }
  if (enable_status == hipSuccess) {
    HIP_CHECK(hipDeviceDisablePeerAccess(kSourceDevice));
  }
  HIP_CHECK(hipSetDevice(original_device));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - On the null stream and on a created stream, with the stream blocked, copies that write the
 *   batch sources are queued before the batch and copies that read the batch destinations are
 *   queued after it. After unblocking and stream sync, the later copies hold the values the earlier
 *   copies wrote.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_Stream_OrdersBatchBetweenPriorAndLaterWork) {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::created);
  const std::vector<uint8_t> first_pattern = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
  const std::vector<uint8_t> second_pattern = {0x90, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07};
  const hipExtent extent = make_hipExtent(4, 2, 1);
  StreamGuard stream(stream_type);
  LinearAllocGuard<uint8_t> first_produced(LinearAllocs::hipMalloc, first_pattern.size());
  LinearAllocGuard<uint8_t> second_produced(LinearAllocs::hipMalloc, second_pattern.size());
  LinearAllocGuard<uint8_t> first_source(LinearAllocs::hipMalloc, first_pattern.size());
  LinearAllocGuard<uint8_t> second_source(LinearAllocs::hipMalloc, second_pattern.size());
  LinearAllocGuard<uint8_t> first_destination(LinearAllocs::hipMalloc, first_pattern.size());
  LinearAllocGuard<uint8_t> second_destination(LinearAllocs::hipMalloc, second_pattern.size());
  LinearAllocGuard<uint8_t> first_witness(LinearAllocs::hipMalloc, first_pattern.size());
  LinearAllocGuard<uint8_t> second_witness(LinearAllocs::hipMalloc, second_pattern.size());
  HIP_CHECK(hipMemcpy(first_produced.ptr(), first_pattern.data(), first_pattern.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(second_produced.ptr(), second_pattern.data(), second_pattern.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(first_source.ptr(), 0x11, first_pattern.size()));
  HIP_CHECK(hipMemset(second_source.ptr(), 0x11, second_pattern.size()));
  HIP_CHECK(hipMemset(first_witness.ptr(), 0x33, first_pattern.size()));
  HIP_CHECK(hipMemset(second_witness.ptr(), 0x33, second_pattern.size()));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  // Holding the stream makes a batch that skips stream order read the sources before the
  // producing copies run, instead of relying on those copies finishing first by chance.
  HipTest::BlockingContext blocker{stream.stream()};
  blocker.block_stream();
  HIP_CHECK(hipMemcpyAsync(first_source.ptr(), first_produced.ptr(), first_pattern.size(),
                           hipMemcpyDeviceToDevice, stream.stream()));
  HIP_CHECK(hipMemcpyAsync(second_source.ptr(), second_produced.ptr(), second_pattern.size(),
                           hipMemcpyDeviceToDevice, stream.stream()));

  hipMemcpy3DBatchOp ops[] = {
      MakePointerToPointerOp(first_source.ptr(), first_destination.ptr(), extent),
      MakePointerToPointerOp(second_source.ptr(), second_destination.ptr(), extent)};
  size_t fail_idx = 0;
  const hipError_t batch_status = hipMemcpy3DBatchAsync(2, ops, &fail_idx, 0, stream.stream());
  HIP_CHECK(hipMemcpyAsync(first_witness.ptr(), first_destination.ptr(), first_pattern.size(),
                           hipMemcpyDeviceToDevice, stream.stream()));
  HIP_CHECK(hipMemcpyAsync(second_witness.ptr(), second_destination.ptr(), second_pattern.size(),
                           hipMemcpyDeviceToDevice, stream.stream()));
  blocker.unblock_stream();
  HIP_CHECK(batch_status);
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint8_t> first_copied(first_pattern.size());
  std::vector<uint8_t> second_copied(second_pattern.size());
  HIP_CHECK(hipMemcpy(first_copied.data(), first_witness.ptr(), first_pattern.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(second_copied.data(), second_witness.ptr(), second_pattern.size(),
                      hipMemcpyDeviceToHost));
  REQUIRE(first_copied == first_pattern);
  REQUIRE(second_copied == second_pattern);
}

/**
 * Test Description
 * ------------------------
 * - A copy between two pinned host buffers or two pageable host buffers: the destination matches
 *   the source when hipMemcpy3DBatchAsync returns, before any stream sync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_HostToHost_IsVisibleOnReturn) {
  const auto host_allocation = GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::malloc);
  const std::vector<uint8_t> pattern = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
  const hipExtent extent = make_hipExtent(4, 2, 1);
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(host_allocation, pattern.size());
  LinearAllocGuard<uint8_t> destination(host_allocation, pattern.size());
  std::copy(pattern.begin(), pattern.end(), source.host_ptr());
  std::fill_n(destination.host_ptr(), pattern.size(), static_cast<uint8_t>(0xFF));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  hipMemcpy3DBatchOp op = MakePointerToPointerOp(source.host_ptr(), destination.host_ptr(), extent);
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  const std::vector<uint8_t> copied(destination.host_ptr(),
                                    destination.host_ptr() + pattern.size());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - A pageable host to device copy with srcAccessOrder Any, with the source unchanged until after
 *   stream sync: after stream sync, the destination matches the source.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_Any_MatchesSourceAfterSync) {
  const std::vector<uint8_t> pattern = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
  const hipExtent extent = make_hipExtent(4, 2, 1);
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::malloc, pattern.size());
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, pattern.size());
  std::copy(pattern.begin(), pattern.end(), source.host_ptr());
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, pattern.size()));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  hipMemcpy3DBatchOp op = MakePointerToPointerOp(source.host_ptr(), destination.ptr(), extent);
  op.srcAccessOrder = hipMemcpySrcAccessOrderAny;
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint8_t> copied(pattern.size());
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), pattern.size(), hipMemcpyDeviceToHost));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 * - DuringApiCall captures pageable and mapped host sources before return while the destination
 *   waits on its stream. Padded rows, padded layers, pointer offsets and array offsets are
 * preserved by both the batch API and the single-operation API, despite a subsequent source
 * overwrite. Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_DuringApiCall_IgnoresSourceWritesAfterReturn) {
  const auto source_kind =
      GENERATE(LinearAllocs::malloc, LinearAllocs::mallocAndRegister, LinearAllocs::hipHostMalloc);
  const bool array_destination = GENERATE(false, true);
  const bool single_operation = GENERATE(false, true);
  INFO("source: " << to_string(source_kind) << ", array destination: " << array_destination
                  << ", single operation: " << single_operation);
  if (array_destination) {
    CHECK_IMAGE_SUPPORT
  }

  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kSourceRow = 7;
  constexpr size_t kSourceLayer = 5;
  constexpr size_t kSourceOffset = 3;
  constexpr size_t kSourceCount = kSourceOffset + kSourceRow * kSourceLayer * kDepth;
  constexpr size_t kDestinationRow = 8;
  constexpr size_t kDestinationLayer = 5;
  constexpr size_t kDestinationDepth = 3;
  constexpr size_t kDestinationOffset = kDestinationRow * (kDestinationLayer + 1) + 2;
  constexpr size_t kDestinationCount = kDestinationRow * kDestinationLayer * kDestinationDepth;
  const hipExtent array_extent =
      make_hipExtent(kDestinationRow, kDestinationLayer, kDestinationDepth);
  SnapshotGate gate;
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  LinearAllocGuard<uint32_t> source(source_kind, kSourceCount * sizeof(uint32_t));
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc,
                                         kDestinationCount * sizeof(uint32_t));
  std::unique_ptr<ArrayAllocGuard<uint32_t>> array;
  if (array_destination) {
    array = std::make_unique<ArrayAllocGuard<uint32_t>>(array_extent);
  }
  OpenSnapshotGateOnExit release{gate};
  for (size_t i = 0; i < kSourceCount; ++i) {
    source.host_ptr()[i] = 0x11000000u + i;
  }
  std::vector<uint32_t> expected(kDestinationCount, 0xFFFFFFFFu);
  if (array_destination) {
    hipMemcpy3DParms fill{};
    fill.srcPtr = make_hipPitchedPtr(expected.data(), kDestinationRow * sizeof(uint32_t),
                                     kDestinationRow, kDestinationLayer);
    fill.dstArray = array->ptr();
    fill.extent = array_extent;
    fill.kind = hipMemcpyHostToDevice;
    HIP_CHECK(hipMemcpy3D(&fill));
  } else {
    HIP_CHECK(hipMemset(destination.ptr(), 0xFF, kDestinationCount * sizeof(uint32_t)));
  }
  for (size_t z = 0; z < kDepth; ++z) {
    for (size_t y = 0; y < kHeight; ++y) {
      std::copy_n(
          source.host_ptr() + kSourceOffset + (z * kSourceLayer + y) * kSourceRow, kWidth,
          expected.data() + kDestinationOffset + (z * kDestinationLayer + y) * kDestinationRow);
    }
  }
  const size_t element_bytes = array_destination ? 1 : sizeof(uint32_t);
  hipMemcpy3DBatchOp op =
      MakePointerToPointerOp(source.ptr() + kSourceOffset, destination.ptr() + kDestinationOffset,
                             make_hipExtent(kWidth * element_bytes, kHeight, kDepth));
  op.src.op.ptr.rowLength = kSourceRow * element_bytes;
  op.src.op.ptr.layerHeight = kSourceLayer;
  if (array_destination) {
    op.dst.type = hipMemcpyOperandTypeArray;
    op.dst.op.array.array = array->ptr();
    op.dst.op.array.offset = {2, 1, 1};
  } else {
    op.dst.op.ptr.rowLength = kDestinationRow * sizeof(uint32_t);
    op.dst.op.ptr.layerHeight = kDestinationLayer;
  }
  op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipLaunchHostFunc(stream.stream(), HoldSnapshotCopy, &gate));
  size_t fail_idx = 0;
  if (single_operation) {
    HIP_CHECK(hipMemcpy3DWithAttributesAsync(&op, 0, stream.stream()));
  } else {
    HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
    REQUIRE(fail_idx == SIZE_MAX);
  }
  std::fill_n(source.host_ptr(), kSourceCount, 0xABABABABu);
  gate.open.store(true, std::memory_order_release);
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  REQUIRE_FALSE(gate.timed_out.load(std::memory_order_acquire));

  std::vector<uint32_t> copied(kDestinationCount);
  if (array_destination) {
    hipMemcpy3DParms read{};
    read.srcArray = array->ptr();
    read.dstPtr = make_hipPitchedPtr(copied.data(), kDestinationRow * sizeof(uint32_t),
                                     kDestinationRow, kDestinationLayer);
    read.extent = array_extent;
    read.kind = hipMemcpyDeviceToHost;
    HIP_CHECK(hipMemcpy3D(&read));
  } else {
    HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), kDestinationCount * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
  }
  REQUIRE(copied == expected);
}

/**
 * Test Description
 * ------------------------
 * - Both DuringApiCall entry points preserve multiple layered-1D array layers from pageable,
 *   registered and pinned host sources with default or padded row and layer strides.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_DuringApiCall_Layered1D) {
  CHECK_IMAGE_SUPPORT
  const auto source_kind =
      GENERATE(LinearAllocs::malloc, LinearAllocs::mallocAndRegister, LinearAllocs::hipHostMalloc);
  const bool padded = GENERATE(false, true);
  const bool single_operation = GENERATE(false, true);
  INFO("source: " << to_string(source_kind) << ", padded: " << padded
                  << ", single operation: " << single_operation);
  constexpr size_t kWidth = 4;
  constexpr size_t kLayers = 3;
  constexpr size_t kSourceOffset = 3;
  constexpr size_t kDestinationWidth = 6;
  constexpr size_t kDestinationLayers = 5;
  const size_t source_row = padded ? 7 : kWidth;
  const size_t source_layer_height = padded ? 2 : 1;
  const size_t source_count = kSourceOffset + source_row * source_layer_height * kLayers;
  const hipExtent destination_extent = make_hipExtent(kDestinationWidth, 1, kDestinationLayers);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  LinearAllocGuard<uint32_t> source(source_kind, source_count * sizeof(uint32_t));
  ArrayAllocGuard<uint32_t> destination(make_hipExtent(kDestinationWidth, 0, kDestinationLayers),
                                        hipArrayLayered);
  for (size_t i = 0; i < source_count; ++i) {
    source.host_ptr()[i] = 0x11000000u + i;
  }
  std::vector<uint32_t> expected(kDestinationWidth * kDestinationLayers, 0xFFFFFFFFu);
  hipMemcpy3DParms fill{};
  fill.srcPtr = make_hipPitchedPtr(expected.data(), kDestinationWidth * sizeof(uint32_t),
                                   kDestinationWidth, 1);
  fill.dstArray = destination.ptr();
  fill.extent = destination_extent;
  fill.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fill));
  for (size_t layer = 0; layer < kLayers; ++layer) {
    std::copy_n(source.host_ptr() + kSourceOffset + layer * source_row * source_layer_height,
                kWidth, expected.data() + (layer + 1) * kDestinationWidth + 1);
  }

  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypePointer;
  op.src.op.ptr.ptr = source.ptr() + kSourceOffset;
  op.src.op.ptr.rowLength = padded ? source_row : 0;
  op.src.op.ptr.layerHeight = padded ? source_layer_height : 0;
  op.dst.type = hipMemcpyOperandTypeArray;
  op.dst.op.array.array = destination.ptr();
  op.dst.op.array.offset = {1, 0, 1};
  op.extent = make_hipExtent(kWidth, 1, kLayers);
  op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  if (single_operation) {
    HIP_CHECK(hipMemcpy3DWithAttributesAsync(&op, 0, stream.stream()));
  } else {
    size_t fail_idx = 0;
    HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
    REQUIRE(fail_idx == SIZE_MAX);
  }
  std::fill_n(source.host_ptr(), source_count, 0xABABABABu);
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint32_t> copied(expected.size());
  hipMemcpy3DParms read{};
  read.srcArray = destination.ptr();
  read.dstPtr =
      make_hipPitchedPtr(copied.data(), kDestinationWidth * sizeof(uint32_t), kDestinationWidth, 1);
  read.extent = destination_extent;
  read.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&read));
  REQUIRE(copied == expected);
}

/**
 * Test Description
 * ------------------------
 * - A batch whose flags argument is 1 instead of 0 returns hipErrorInvalidValue and sets failIdx to
 *   SIZE_MAX.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_NonZeroFlags_SetsFailIdxToSizeMax) {
  constexpr size_t kCopyBytes = 256;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, kCopyBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kCopyBytes);

  hipMemcpy3DBatchOp op =
      MakePointerToPointerOp(source.ptr(), destination.ptr(), make_hipExtent(kCopyBytes, 1, 1));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 1, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == SIZE_MAX);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A batch with a count of 1 and a null operation list returns hipErrorInvalidValue and sets
 *   failIdx to SIZE_MAX.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_NullOpList_SetsFailIdxToSizeMax) {
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, nullptr, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == SIZE_MAX);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A two-operation batch whose second operation has srcAccessOrder Invalid returns
 *   hipErrorInvalidValue and sets failIdx to 1.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_InvalidSrcAccessOrder_ReportsThatOp) {
  constexpr size_t kCopyBytes = 256;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source_0(LinearAllocs::hipMalloc, kCopyBytes);
  LinearAllocGuard<uint8_t> destination_0(LinearAllocs::hipMalloc, kCopyBytes);
  LinearAllocGuard<uint8_t> source_1(LinearAllocs::hipMalloc, kCopyBytes);
  LinearAllocGuard<uint8_t> destination_1(LinearAllocs::hipMalloc, kCopyBytes);
  const hipExtent extent = make_hipExtent(kCopyBytes, 1, 1);

  hipMemcpy3DBatchOp ops[] = {MakePointerToPointerOp(source_0.ptr(), destination_0.ptr(), extent),
                              MakePointerToPointerOp(source_1.ptr(), destination_1.ptr(), extent)};
  ops[1].srcAccessOrder = hipMemcpySrcAccessOrderInvalid;

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(2, ops, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == 1);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A two-operation batch whose second operation has a zero width, height, or depth returns
 *   hipErrorInvalidValue and sets failIdx to 1.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_ZeroExtent_IsRejected) {
  const int zero_axis = GENERATE(0, 1, 2);
  constexpr size_t kBytes = 16;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kBytes);

  hipMemcpy3DBatchOp ops[2]{};
  ops[0] = MakePointerToPointerOp(source.ptr(), destination.ptr(), make_hipExtent(4, 2, 2));
  ops[1] = ops[0];
  if (zero_axis == 0) {
    ops[1].extent.width = 0;
  } else if (zero_axis == 1) {
    ops[1].extent.height = 0;
  } else {
    ops[1].extent.depth = 0;
  }

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(2, ops, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == 1);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A copy of width 4 with a source or destination rowLength of 3 returns hipErrorInvalidValue and
 *   sets failIdx to 0.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_RowLengthShorterThanWidth_IsRejected) {
  constexpr size_t kBytes = 256;
  constexpr size_t kShortRowLength = 3;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kBytes);

  hipMemcpy3DBatchOp op =
      MakePointerToPointerOp(source.ptr(), destination.ptr(), make_hipExtent(4, 2, 1));
  SECTION("Source rowLength") { op.src.op.ptr.rowLength = kShortRowLength; }
  SECTION("Destination rowLength") { op.dst.op.ptr.rowLength = kShortRowLength; }

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 7;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == 0);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A copy of height 3 with a source or destination layerHeight of 2 returns hipErrorInvalidValue
 *   and sets failIdx to 0.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_LayerHeightShorterThanHeight_IsRejected) {
  constexpr size_t kBytes = 256;
  constexpr size_t kShortLayerHeight = 2;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kBytes);

  hipMemcpy3DBatchOp op =
      MakePointerToPointerOp(source.ptr(), destination.ptr(), make_hipExtent(4, 3, 2));
  SECTION("Source layerHeight") { op.src.op.ptr.layerHeight = kShortLayerHeight; }
  SECTION("Destination layerHeight") { op.dst.op.ptr.layerHeight = kShortLayerHeight; }

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 7;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == 0);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A copy from a uint8 array to a uint32 array returns hipErrorInvalidValue and sets failIdx to 0.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_MismatchedArrayElementSize_ReportsThatOp) {
  CHECK_IMAGE_SUPPORT

  const hipExtent extent = make_hipExtent(4, 2, 1);
  StreamGuard stream(Streams::created);
  ArrayAllocGuard<uint8_t> source(extent);
  ArrayAllocGuard<uint32_t> destination(extent);

  hipMemcpy3DBatchOp op = MakeArrayToArrayOp(source.ptr(), destination.ptr(), extent);

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 7;
  HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()),
                  hipErrorInvalidValue);
  REQUIRE(fail_idx == 0);
  static_cast<void>(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 * - A copy from an 8192-byte malloc buffer whose first 4096 bytes are registered returns
 *   hipErrorInvalidValue.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_PartiallyRegisteredHostSource_IsRejected) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  int host_register = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&host_register, hipDeviceAttributeHostRegisterSupported, device));
  if (host_register == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported);
  }

  constexpr size_t kRegisteredBytes = 4096;
  constexpr size_t kSourceBytes = 2 * kRegisteredBytes;
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> source(LinearAllocs::malloc, kSourceBytes);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, kSourceBytes);
  std::fill_n(source.host_ptr(), kSourceBytes, static_cast<uint8_t>(0x5A));
  HIP_CHECK(hipHostRegister(source.host_ptr(), kRegisteredBytes, hipHostRegisterDefault));

  hipMemcpy3DBatchOp op = MakePointerToPointerOp(source.host_ptr(), destination.ptr(),
                                                 make_hipExtent(kSourceBytes, 1, 1));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  size_t fail_idx = 0;
  const hipError_t status = hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipHostUnregister(source.host_ptr()));
  static_cast<void>(hipGetLastError());
  REQUIRE(status == hipErrorInvalidValue);
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
