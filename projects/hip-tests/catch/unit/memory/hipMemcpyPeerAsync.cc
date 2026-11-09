/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <vector>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpyPeerAsync hipMemcpyPeerAsync
 * @{
 * @ingroup PeerToPeerTest
 * `hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src,
 * int srcDevice, size_t sizeBytes, hipStream_t stream __dparm(0))` -
 * Copies memory from one device to memory on another device.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs basic peer to peer async memcpy functionality between each pair of devices.
 *  - Launches computation kernel.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_Positive_Default) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);

  int can_access_peer = 0;

  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  HIP_CHECK(hipSetDevice(src_device));

  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, allocation_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size,
                                 hipHostMallocPortable);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

    const auto element_count = allocation_size / sizeof(*src_alloc.ptr());
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int expected_value = 22;
    HIP_CHECK(hipSetDevice(src_device));
    VectorSet<<<block_count, thread_count, 0, stream>>>(src_alloc.ptr(), expected_value,
                                                        element_count);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device,
                                 allocation_size, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(
        hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

    ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks synchronization behavior for peer async memcpy.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    HIP_CHECK(hipSetDevice(src_device));
    LaunchDelayKernel(std::chrono::milliseconds{100}, nullptr);

    HIP_CHECK(hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device,
                                 kPageSize, stream));
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks that no data is copied if size is set to 0.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_Positive_ZeroSize) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  const auto allocation_size = kPageSize;

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, allocation_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size,
                                 hipHostMallocPortable);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

    const auto element_count = allocation_size / sizeof(*src_alloc.ptr());
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int set_value_s = 22;
    HIP_CHECK(hipSetDevice(src_device));
    VectorSet<<<block_count, thread_count, 0, stream>>>(src_alloc.ptr(), set_value_s,
                                                        element_count);
    HIP_CHECK(hipGetLastError());

    constexpr int expected_value = 20;
    VectorSet<<<block_count, thread_count, 0, stream>>>(dst_alloc.ptr(), expected_value,
                                                        element_count);
    HIP_CHECK(hipGetLastError());

    constexpr int set_value_h = 21;
    std::fill_n(result.host_ptr(), element_count, set_value_h);

    HIP_CHECK(
        hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, 0, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(
        hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

    ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When copying more than allocated
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When destination device ID is not valid (out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When source device ID is not valid (out of bounds)
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When stream is not valid
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_Negative_Parameters) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    HIP_CHECK(hipSetDevice(src_device));

    SECTION("Nullptr to Destination Pointer") {
      HIP_CHECK_ERROR(
          hipMemcpyPeerAsync(nullptr, dst_device, src_alloc.ptr(), src_device, kPageSize, stream),
          hipErrorInvalidValue);
    }

    SECTION("Nullptr to Source Pointer") {
      HIP_CHECK_ERROR(
          hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, nullptr, src_device, kPageSize, stream),
          hipErrorInvalidValue);
    }

    SECTION("Passing more than allocated size") {
      HIP_CHECK_ERROR(hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device,
                                         kPageSize + 1, stream),
                      hipErrorInvalidValue);
    }

    SECTION("Passing invalid Destination device ID") {
      HIP_CHECK_ERROR(hipMemcpyPeerAsync(dst_alloc.ptr(), device_count, src_alloc.ptr(), src_device,
                                         kPageSize, stream),
                      hipErrorInvalidDevice);
    }

    SECTION("Passing invalid Source device ID") {
      HIP_CHECK_ERROR(hipMemcpyPeerAsync(dst_alloc.ptr(), dst_device, src_alloc.ptr(), device_count,
                                         kPageSize, stream),
                      hipErrorInvalidDevice);
    }
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}


/**
 * Test Description
 * ------------------------
 *  - Allocates VMM (expandable-segment style) memory on two devices, enables
 *    peer access, and grants hipMemSetAccess to the owning device only.
 *  - hipMemcpyPeerAsync and hipMemcpyPeer must return an explicit error instead
 *    of reporting success while copying zero bytes.
 *  - Positive control: after hipMemSetAccess grants the peer device access to
 *    both mappings, the copies must succeed and transfer every byte.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - VMM supported
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_VmmPeerAccessValidation) {
  const int device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  int vmm_src = 0;
  int vmm_dst = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&vmm_src, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  HIP_CHECK(
      hipDeviceGetAttribute(&vmm_dst, hipDeviceAttributeVirtualMemoryManagementSupported, 1));
  if (vmm_src == 0 || vmm_dst == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
  }

  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (can_access_peer == 0) {
    INFO("Peer access cannot be enabled between devices 0 and 1");
    return;
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityRecommended));
  const size_t size = granularity * 4;
  const unsigned char pattern = 0x5a;

  void* src = nullptr;
  void* dst = nullptr;
  hipMemGenericAllocationHandle_t src_handle{nullptr};
  hipMemGenericAllocationHandle_t dst_handle{nullptr};

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemAddressReserve(&src, size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&src_handle, size, &prop, 0));
  HIP_CHECK(hipMemMap(src, size, 0, src_handle, 0));

  prop.location.id = 1;
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemAddressReserve(&dst, size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&dst_handle, size, &prop, 0));
  HIP_CHECK(hipMemMap(dst, size, 0, dst_handle, 0));

  // Owner-only access on both mappings.
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  desc.location.id = 0;
  HIP_CHECK(hipMemSetAccess(src, size, &desc, 1));
  desc.location.id = 1;
  HIP_CHECK(hipMemSetAccess(dst, size, &desc, 1));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemset(src, pattern, size));
  HIP_CHECK(hipDeviceSynchronize());

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<unsigned char> host(size, 0);

  // Negative: peer access is enabled, but hipMemSetAccess never granted the
  // current device access to the peer-owned mapping. The copy must fail
  // explicitly; returning success with an unchanged destination is the bug.
  REQUIRE(hipMemcpyPeerAsync(dst, 1, src, 0, size, stream) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipMemcpyPeer(dst, 1, src, 0, size) == hipErrorInvalidValue);
  (void)hipGetLastError();

  // Positive control: grant the peer device access to both mappings.
  desc.location.id = 1;
  HIP_CHECK(hipMemSetAccess(src, size, &desc, 1));
  desc.location.id = 0;
  HIP_CHECK(hipMemSetAccess(dst, size, &desc, 1));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemset(src, pattern, size));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpyPeerAsync(dst, 1, src, 0, size, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(host.data(), dst, size, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(host.begin(), host.end(),
                      [pattern](unsigned char value) { return value == pattern; }));

  // Sync peer copy with the same explicit grant.
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemset(src, pattern, size));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpyPeer(dst, 1, src, 0, size));
  HIP_CHECK(hipSetDevice(1));
  std::fill(host.begin(), host.end(), 0);
  HIP_CHECK(hipMemcpy(host.data(), dst, size, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(host.begin(), host.end(),
                      [pattern](unsigned char value) { return value == pattern; }));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemUnmap(src, size));
  HIP_CHECK(hipMemRelease(src_handle));
  HIP_CHECK(hipMemAddressFree(src, size));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemUnmap(dst, size));
  HIP_CHECK(hipMemRelease(dst_handle));
  HIP_CHECK(hipMemAddressFree(dst, size));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceDisablePeerAccess(1));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipDeviceDisablePeerAccess(0));
}

HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_Capture) {
  const int device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipSetDevice(0));
  LinearAllocGuard<int> src_device_alloc(LinearAllocs::hipMalloc, kPageSize);
  HIP_CHECK(hipSetDevice(1));
  LinearAllocGuard<int> dst_device_alloc(LinearAllocs::hipMalloc, kPageSize);

  HIP_CHECK(hipSetDevice(0));
  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(
      hipMemcpyPeerAsync(src_device_alloc.ptr(), 0, dst_device_alloc.ptr(), 1, kPageSize, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group PeerToPeerTest.
 * @}
 */
