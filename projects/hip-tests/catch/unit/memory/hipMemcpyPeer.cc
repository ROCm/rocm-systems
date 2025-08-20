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
 * @addtogroup hipMemcpyPeer hipMemcpyPeer
 * @{
 * @ingroup PeerToPeerTest
 * `hipMemcpyPeer(void* dst, int dstDeviceId, const void* src,
 * int srcDeviceId, size_t sizeBytes)` -
 * Copies memory from one device to memory on another device.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs basic peer to peer memcpy functionality between each pair of devices.
 *  - Launches computation kernel.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeer.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyPeer_Positive_Default") {
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
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, allocation_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

    const auto element_count = allocation_size / sizeof(*src_alloc.ptr());
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int expected_value = 22;
    HIP_CHECK(hipSetDevice(src_device));
    VectorSet<<<block_count, thread_count, 0>>>(src_alloc.ptr(), expected_value, element_count);
    HIP_CHECK(hipGetLastError());

    hipError_t memcpy_err = hipSuccess;
    BEGIN_CAPTURE_SYNC(memcpy_err, false);
    HIP_CHECK_ERROR(
        hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, allocation_size),
        memcpy_err);
    END_CAPTURE_SYNC(memcpy_err);

    if (memcpy_err == hipSuccess) {
      HIP_CHECK(
          hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

      HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

      ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
    }
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks synchronization behavior for peer memcpy.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeer.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyPeer_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

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
    LaunchDelayKernel(std::chrono::milliseconds{100}, nullptr);

    HIP_CHECK(hipSetDevice(src_device));
    LaunchDelayKernel(std::chrono::milliseconds{100}, nullptr);

    HIP_CHECK(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, kPageSize));
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
 *  - unit/memory/hipMemcpyPeer.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyPeer_Positive_ZeroSize") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

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
    VectorSet<<<block_count, thread_count, 0>>>(src_alloc.ptr(), set_value_s, element_count);
    HIP_CHECK(hipGetLastError());

    constexpr int expected_value = 20;
    HIP_CHECK(hipSetDevice(dst_device));
    VectorSet<<<block_count, thread_count, 0>>>(dst_alloc.ptr(), expected_value, element_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipSetDevice(src_device));

    constexpr int set_value_h = 21;
    std::fill_n(result.host_ptr(), element_count, set_value_h);

    hipError_t memcpy_err = hipSuccess;
    BEGIN_CAPTURE_SYNC(memcpy_err, false);
    HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, 0),
                    memcpy_err);
    END_CAPTURE_SYNC(memcpy_err);

    if (memcpy_err == hipSuccess) {
      HIP_CHECK(
          hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

      HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

      ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
    }
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
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyPeer.cc
 * Test requirements
 * ------------------------
 *  - Peer access supported
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyPeer_Negative_Parameters") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

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
      HIP_CHECK_ERROR(hipMemcpyPeer(nullptr, dst_device, src_alloc.ptr(), src_device, kPageSize),
                      hipErrorInvalidValue);
    }

    SECTION("Nullptr to Source Pointer") {
      HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), dst_device, nullptr, src_device, kPageSize),
                      hipErrorInvalidValue);
    }

    SECTION("Passing more than allocated size") {
      HIP_CHECK_ERROR(
          hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, kPageSize + 1),
          hipErrorInvalidValue);
    }

    SECTION("Passing invalid Destination device ID") {
      HIP_CHECK_ERROR(
          hipMemcpyPeer(dst_alloc.ptr(), device_count, src_alloc.ptr(), src_device, kPageSize),
          hipErrorInvalidDevice);
    }

    SECTION("Passing invalid Source device ID") {
      HIP_CHECK_ERROR(
          hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), device_count, kPageSize),
          hipErrorInvalidDevice);
    }

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

/**
 * End doxygen group PeerToPeerTest.
 * @}
 */
