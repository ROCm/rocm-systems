/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <numeric>
#include <random>
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


namespace {
// Clean up even when a fatal assertion aborts a generated test case. Preserve
// pre-existing peer access and the caller's current device.
struct VmmPeerResources {
  int original_device = 0;
  void* ptr[2]{};
  hipMemGenericAllocationHandle_t handle[2][2]{};
  bool mapped[2][2]{};
  size_t mapped_size[2][2]{};
  bool enabled_peer[2]{};
  size_t size = 0;
  int stream_device = 0;
  hipStream_t stream = nullptr;

  ~VmmPeerResources() {
    (void)hipSetDevice(stream_device);
    if (stream) (void)hipStreamDestroy(stream);
    for (int device = 0; device < 2; ++device) {
      (void)hipSetDevice(device);
      for (int segment = 0; segment < 2; ++segment) {
        void* va = ptr[device] ? static_cast<char*>(ptr[device]) + segment * mapped_size[device][0]
                               : nullptr;
        if (mapped[device][segment]) (void)hipMemUnmap(va, mapped_size[device][segment]);
        if (handle[device][segment]) (void)hipMemRelease(handle[device][segment]);
      }
      if (ptr[device]) (void)hipMemAddressFree(ptr[device], size);
      if (enabled_peer[device]) (void)hipDeviceDisablePeerAccess(1 - device);
    }
    (void)hipSetDevice(original_device);
  }
};
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Without enabled peer access, owner-only VMM mappings must reject copies.
 *  - With enabled peer access, copies must grant access and transfer random
 *    bytes exactly, while respecting explicit VMM restrictions.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device, peer access supported, VMM supported
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_VmmPeerAccessValidation) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  for (int device = 0; device < 2; ++device) {
    int supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device));
    if (!supported) HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
    HIP_CHECK(hipDeviceCanAccessPeer(&supported, device, 1 - device));
    if (!supported) HIP_SKIP_TEST("Bidirectional peer access is required");
  }

  const int src_device = GENERATE(0, 1);
  const int dst_device = 1 - src_device;
  const bool async = GENERATE(false, true);
  const bool on_source = GENERATE(true, false);
  const size_t bytes = 4 * 1024 * 1024;
  // 0: no peer access; 1: peer enabled, owner-only VMM access;
  // 2: explicit RW; 3: explicit denial; 4: read-only destination;
  // 5: read-only source.
  const int permissions = GENERATE(0, 1, 2, 3, 4, 5);
  CAPTURE(src_device, async, on_source, bytes, permissions);

  VmmPeerResources resources;
  HIP_CHECK(hipGetDevice(&resources.original_device));
  resources.stream_device = on_source ? src_device : dst_device;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  size_t granularity[2]{};
  for (int device = 0; device < 2; ++device) {
    prop.location.id = device;
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity[device], &prop,
                                             hipMemAllocationGranularityMinimum));
  }
  const size_t alignment = std::lcm(granularity[0], granularity[1]);
  resources.size = ((bytes + alignment - 1) / alignment) * alignment;
  auto set_access = [&](int allocation, int device, hipMemAccessFlags flags) {
    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = device;
    desc.flags = flags;
    HIP_CHECK(hipMemSetAccess(resources.ptr[allocation], resources.size, &desc, 1));
  };
  if (permissions != 0) {
    for (int device = 0; device < 2; ++device) {
      HIP_CHECK(hipSetDevice(device));
      const auto status = hipDeviceEnablePeerAccess(1 - device, 0);
      resources.enabled_peer[device] = status == hipSuccess;
      REQUIRE((status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled));
      (void)hipGetLastError();
    }
  }
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    prop.location.id = device;
    HIP_CHECK(hipMemAddressReserve(&resources.ptr[device], resources.size, 0, nullptr, 0));
    HIP_CHECK(hipMemCreate(&resources.handle[device][0], resources.size, &prop, 0));
    HIP_CHECK(hipMemMap(resources.ptr[device], resources.size, 0, resources.handle[device][0], 0));
    resources.mapped[device][0] = true;
    resources.mapped_size[device][0] = resources.size;
    set_access(device, device, hipMemAccessFlagsProtReadWrite);
  }

  std::mt19937 random(0x15331);
  std::vector<unsigned char> source(resources.size), expected(resources.size);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<unsigned char>(random());
    expected[i] = source[i] ^ 0xff;  // Every destination byte starts different.
  }
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(
      hipMemcpy(resources.ptr[src_device], source.data(), source.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipSetDevice(dst_device));
  HIP_CHECK(hipMemcpy(resources.ptr[dst_device], expected.data(), expected.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned char> actual(resources.size);
  HIP_CHECK(
      hipMemcpy(actual.data(), resources.ptr[dst_device], actual.size(), hipMemcpyDeviceToHost));
  REQUIRE(std::equal(actual.begin(), actual.end(), expected.begin()));
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(
      hipMemcpy(actual.data(), resources.ptr[src_device], actual.size(), hipMemcpyDeviceToHost));
  REQUIRE(std::equal(actual.begin(), actual.end(), source.begin()));

  HIP_CHECK(hipSetDevice(resources.stream_device));
  HIP_CHECK(hipStreamCreate(&resources.stream));
  if (permissions == 2) {
    set_access(src_device, resources.stream_device, hipMemAccessFlagsProtReadWrite);
    set_access(dst_device, resources.stream_device, hipMemAccessFlagsProtReadWrite);
  } else if (permissions == 3) {
    set_access(1 - resources.stream_device, resources.stream_device, hipMemAccessFlagsProtNone);
  } else if (permissions == 4) {
    set_access(dst_device, resources.stream_device, hipMemAccessFlagsProtRead);
  } else if (permissions == 5) {
    set_access(src_device, resources.stream_device, hipMemAccessFlagsProtRead);
  }
  auto* src = resources.ptr[src_device];
  auto* dst = resources.ptr[dst_device];
  hipMemLocation location{hipMemLocationTypeDevice, resources.stream_device};
  unsigned long long src_access = 0, dst_access = 0;
  HIP_CHECK(hipMemGetAccess(&src_access, &location, src));
  HIP_CHECK(hipMemGetAccess(&dst_access, &location, dst));
  REQUIRE(src_access == (permissions == 5                                            ? 1
                         : permissions == 2 || resources.stream_device == src_device ? 3
                                                                                     : 0));
  REQUIRE(dst_access == (permissions == 4                                            ? 1
                         : permissions == 2 || resources.stream_device == dst_device ? 3
                                                                                     : 0));
  const auto status =
      async ? hipMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, resources.stream)
            : hipMemcpyPeer(dst, dst_device, src, src_device, bytes);
  const bool allowed = permissions == 1 || permissions == 2 || permissions == 5;
  REQUIRE(status == (allowed ? hipSuccess : hipErrorInvalidValue));
  (void)hipGetLastError();
  HIP_CHECK(hipStreamSynchronize(resources.stream));
  HIP_CHECK(hipDeviceSynchronize());

  // Restore owner access for readback. The destination must be unchanged on
  // error and match the entire random payload on success.
  set_access(src_device, src_device, hipMemAccessFlagsProtReadWrite);
  set_access(dst_device, dst_device, hipMemAccessFlagsProtReadWrite);
  if (allowed) std::copy_n(source.begin(), bytes, expected.begin());
  HIP_CHECK(hipSetDevice(dst_device));
  HIP_CHECK(
      hipMemcpy(actual.data(), resources.ptr[dst_device], actual.size(), hipMemcpyDeviceToHost));
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  INFO("First destination mismatch at byte " << std::distance(actual.begin(), mismatch.first));
  REQUIRE(std::equal(actual.begin(), actual.end(), expected.begin()));
}

/**
 * Test Description
 * ------------------------
 *  - Copy random bytes across two separately mapped VMM physical segments.
 *  - Peer access is enabled, but each mapping initially grants owner access only.
 *  - Both copy APIs must grant each segment and preserve the exact payload.
 *    An explicit denial on the second segment must reject the entire copy.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeerAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device, peer access supported, VMM supported
 */
HIP_TEST_CASE(Unit_hipMemcpyPeerAsync_VmmSegmentedPeerAccess) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  for (int device = 0; device < 2; ++device) {
    int supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device));
    if (!supported) HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
    HIP_CHECK(hipDeviceCanAccessPeer(&supported, device, 1 - device));
    if (!supported) HIP_SKIP_TEST("Bidirectional peer access is required");
  }

  const int src_device = GENERATE(0, 1);
  const int dst_device = 1 - src_device;
  const bool async = GENERATE(false, true);
  const bool on_source = GENERATE(true, false);
  const bool deny_second = GENERATE(false, true);
  CAPTURE(src_device, async, on_source, deny_second);

  VmmPeerResources resources;
  HIP_CHECK(hipGetDevice(&resources.original_device));
  resources.stream_device = on_source ? src_device : dst_device;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  size_t granularity[2]{};
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    const auto status = hipDeviceEnablePeerAccess(1 - device, 0);
    resources.enabled_peer[device] = status == hipSuccess;
    REQUIRE((status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled));
    (void)hipGetLastError();
    prop.location.id = device;
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity[device], &prop,
                                             hipMemAllocationGranularityMinimum));
  }
  const size_t alignment = std::lcm(granularity[0], granularity[1]);
  const size_t segment_size = ((2 * 1024 * 1024 + alignment - 1) / alignment) * alignment;
  resources.size = 2 * segment_size;

  std::mt19937 random(0x15333);
  std::vector<unsigned char> source(resources.size), poison(resources.size), actual(resources.size);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<unsigned char>(random());
    poison[i] = source[i] ^ 0xff;
  }
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    prop.location.id = device;
    HIP_CHECK(hipMemAddressReserve(&resources.ptr[device], resources.size, 0, nullptr, 0));
    for (int segment = 0; segment < 2; ++segment) {
      auto* mapped = static_cast<char*>(resources.ptr[device]) + segment * segment_size;
      HIP_CHECK(hipMemCreate(&resources.handle[device][segment], segment_size, &prop, 0));
      HIP_CHECK(hipMemMap(mapped, segment_size, 0, resources.handle[device][segment], 0));
      resources.mapped[device][segment] = true;
      resources.mapped_size[device][segment] = segment_size;
      hipMemAccessDesc desc{};
      desc.location = prop.location;
      desc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemSetAccess(mapped, segment_size, &desc, 1));
      auto& payload = device == src_device ? source : poison;
      HIP_CHECK(hipMemcpy(mapped, payload.data() + segment * segment_size, segment_size,
                          hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipSetDevice(resources.stream_device));
  HIP_CHECK(hipStreamCreate(&resources.stream));
  hipMemLocation location{hipMemLocationTypeDevice, resources.stream_device};
  for (int device = 0; device < 2; ++device) {
    for (int segment = 0; segment < 2; ++segment) {
      auto* mapped = static_cast<char*>(resources.ptr[device]) + segment * segment_size;
      unsigned long long access = 0;
      HIP_CHECK(hipMemGetAccess(&access, &location, mapped));
      REQUIRE(access == (device == resources.stream_device ? 3 : 0));
    }
  }
  if (deny_second) {
    const int remote = 1 - resources.stream_device;
    auto* second = static_cast<char*>(resources.ptr[remote]) + segment_size;
    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = resources.stream_device;
    desc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemSetAccess(second, segment_size, &desc, 1));
  }

  constexpr size_t offset = 17;
  const size_t bytes = resources.size - 2 * offset;
  auto* src = static_cast<char*>(resources.ptr[src_device]) + offset;
  auto* dst = static_cast<char*>(resources.ptr[dst_device]) + offset;
  const auto status =
      async ? hipMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, resources.stream)
            : hipMemcpyPeer(dst, dst_device, src, src_device, bytes);
  REQUIRE(status == (deny_second ? hipErrorInvalidValue : hipSuccess));
  (void)hipGetLastError();
  HIP_CHECK(hipStreamSynchronize(resources.stream));
  HIP_CHECK(hipSetDevice(dst_device));
  for (int segment = 0; segment < 2; ++segment) {
    auto* mapped = static_cast<char*>(resources.ptr[dst_device]) + segment * segment_size;
    HIP_CHECK(hipMemcpy(actual.data() + segment * segment_size, mapped, segment_size,
                        hipMemcpyDeviceToHost));
  }
  if (!deny_second) std::copy_n(source.begin() + offset, bytes, poison.begin() + offset);
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), poison.begin());
  INFO("First destination mismatch at byte " << std::distance(actual.begin(), mismatch.first));
  REQUIRE(mismatch.first == actual.end());
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
