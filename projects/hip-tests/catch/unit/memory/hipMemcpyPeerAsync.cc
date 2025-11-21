/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime_api.h>
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
TEST_CASE("Unit_hipMemcpyPeerAsync_Positive_Default") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
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
TEST_CASE("Unit_hipMemcpyPeerAsync_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
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
TEST_CASE("Unit_hipMemcpyPeerAsync_Positive_ZeroSize") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
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
TEST_CASE("Unit_hipMemcpyPeerAsync_Negative_Parameters") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
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

TEST_CASE("Unit_hipMemcpyPeerAsync_Capture") {
  const int device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
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
