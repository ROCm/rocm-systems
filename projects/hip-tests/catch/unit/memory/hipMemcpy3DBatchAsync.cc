/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
 * - A pointer-to-pointer copy of a 4x3x2 uint32 volume whose extent width counts bytes: after
 *   stream sync, the destination matches the source.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_PointerToPointer_MatchesSourceAfterSync) {
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;
  constexpr size_t kDepth = 2;
  constexpr size_t kElementCount = kWidth * kHeight * kDepth;
  constexpr size_t kVolumeBytes = kElementCount * sizeof(uint32_t);
  std::vector<uint32_t> pattern(kElementCount);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    pattern[i] = 0x11000000u + i;
  }
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint32_t> source(LinearAllocs::hipMalloc, kVolumeBytes);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, kVolumeBytes);
  HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), kVolumeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, kVolumeBytes));

  hipMemcpy3DBatchOp op = MakePointerToPointerOp(
      source.ptr(), destination.ptr(), make_hipExtent(kWidth * sizeof(uint32_t), kHeight, kDepth));

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
 *   access to device 1 enabled: after stream sync, the destination matches the source.
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

  constexpr size_t kWidth = 7;
  constexpr size_t kHeight = 5;
  constexpr size_t kDepth = 3;
  constexpr size_t kVolumeBytes = kWidth * kHeight * kDepth;
  constexpr int kUntouched = 0xFF;

  int original_device = 0;
  HIP_CHECK(hipGetDevice(&original_device));
  HIP_CHECK(hipSetDevice(kDestinationDevice));
  HIP_CHECK(hipDeviceEnablePeerAccess(kSourceDevice, 0));

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
                                                   make_hipExtent(kWidth, kHeight, kDepth));

    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipGetLastError());
    size_t fail_idx = 0;
    HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), kVolumeBytes, hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipDeviceDisablePeerAccess(kSourceDevice));
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
 * - A copy from a stack array to device memory with srcAccessOrder DuringApiCall, followed by an
 *   overwrite of the stack array right after the call returns: after stream sync, the destination
 *   holds the values from before the overwrite.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpy3DBatchAsync_DuringApiCall_IgnoresSourceWritesAfterReturn) {
  const std::vector<uint8_t> pattern = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
  uint8_t source[8];
  const hipExtent extent = make_hipExtent(4, 2, 1);
  StreamGuard stream(Streams::created);
  LinearAllocGuard<uint8_t> destination(LinearAllocs::hipMalloc, pattern.size());
  std::copy(pattern.begin(), pattern.end(), source);
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, pattern.size()));

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGetLastError());
  hipMemcpy3DBatchOp op = MakePointerToPointerOp(source, destination.ptr(), extent);
  op.srcAccessOrder = hipMemcpySrcAccessOrderDuringApiCall;
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(1, &op, &fail_idx, 0, stream.stream()));
  std::fill_n(source, pattern.size(), static_cast<uint8_t>(0xAB));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::vector<uint8_t> copied(pattern.size());
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), pattern.size(), hipMemcpyDeviceToHost));
  REQUIRE(copied == pattern);
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
