/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipMemcpy2DToArray_Positive_Default - Test basic memcpy between host/device
and 2D array with hipMemcpy2DToArray api
Unit_hipMemcpy2DToArray_Positive_Synchronization_Behavior - Test synchronization
behavior for hipMemcpy2DToArray api
Unit_hipMemcpy2DToArray_Positive_ZeroWidthHeight - Test that no data is copied
when width/height is set to 0 Unit_hipMemcpy2DToArray_Negative_Parameters - Test
unsuccessful execution of hipMemcpy2DToArray api when parameters are invalid
*/
#include "array_memcpy_tests_common.hh"

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipMemcpy2DToArray_Positive_Default) {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  const auto width = GENERATE(16, 32, 48);
  const auto height = GENERATE(1, 16, 32, 48);

  SECTION("Host to Array") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3,
                                               width * sizeof(int), height, hipMemcpyHostToDevice),
                                     width, height);
  }

  SECTION("Host to Array with default kind") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3,
                                               width * sizeof(int), height, hipMemcpyDefault),
                                     width, height);
  }
#if HT_NVIDIA  // EXSWHTEC-120
  SECTION("Device to Array") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell<false, false, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell<false, true, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
  }

  SECTION("Device to Array with default kind") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell<false, false, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDefault),
          width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell<false, true, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDefault),
          width, height);
    }
  }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpy2DToArray_Positive_Synchronization_Behavior) {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyHtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, width * sizeof(int),
                                     width * sizeof(int), height, hipMemcpyHostToDevice),
                           width, height, true);
  }
#if HT_NVIDIA  // EXSWHTEC-214
  SECTION("Device to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyDtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int),
                                     height, hipMemcpyDeviceToDevice),
                           width, height, false);
  }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpy2DToArray_Positive_ZeroWidthHeight) {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;
  const auto width = 16;
  const auto height = 16;

  SECTION("Array to host") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0,
                    hipMemcpyHostToDevice),
          width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyHostToDevice), width,
          height);
    }
  }
  SECTION("Array to device") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyDeviceToDevice),
          width, height);
    }
  }
}

HIP_TEST_CASE(Unit_hipMemcpy2DToArray_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  const auto width = 32;
  const auto height = 32;
  const auto allocation_size = 2 * width * height * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  SECTION("Host to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                                         width * sizeof(int), height, hipMemcpyHostToDevice),
                      hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, 2 * width * sizeof(int),
                                         width * sizeof(int), height, hipMemcpyHostToDevice),
                      hipErrorInvalidValue);
    }
#if HT_NVIDIA  // EXSWHTEC-119
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), width * sizeof(int) - 10,
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int) + 1, height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height + 1, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  }
  SECTION("Device to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                                         width * sizeof(int), height, hipMemcpyDeviceToDevice),
                      hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, device_alloc.pitch(),
                                         width * sizeof(int), height, hipMemcpyDeviceToDevice),
                      hipErrorInvalidValue);
    }
#if HT_NVIDIA  // EXSWHTEC-119
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), width * sizeof(int) - 10,
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int) + 1, height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height + 1, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  }
}

HIP_TEST_CASE(Unit_hipMemcpy2DToArray_Capture) {
  CHECK_IMAGE_SUPPORT

  const auto width = 16;
  const auto height = 16;
  const auto size = width * height * sizeof(int);

  ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), hipArrayDefault);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, size);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), width * sizeof(int),
                                     width * sizeof(int), height, hipMemcpyHostToDevice),
                  memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}

/*
 * A copy between a hipMalloc buffer on device 1 and a hipArray on device 0, in
 * either direction, must succeed without hipDeviceEnablePeerAccess: the runtime
 * grants the buffer to its peers for the copy, as it does for linear copies.
 */
HIP_TEST_CASE(Unit_hipMemcpy2DToArray_PeerBufferFromOtherDevice) {
  CHECK_IMAGE_SUPPORT

  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  // Ensure peer access 0->1 is off so a prior test cannot make this one pass.
  HIP_CHECK(hipSetDevice(0));
  { hipError_t e = hipDeviceDisablePeerAccess(1);
    if (e != hipSuccess && e != hipErrorPeerAccessNotEnabled) HIP_CHECK(e);
    static_cast<void>(hipGetLastError()); }
  // Restore device 0 on scope exit (also fires if a REQUIRE inside a SECTION throws).
  struct DevGuard { ~DevGuard() { static_cast<void>(hipSetDevice(0)); } } dg;

  constexpr size_t kWidth = 16;   // elements
  constexpr size_t kHeight = 16;
  constexpr size_t kRowBytes = kWidth * sizeof(uint32_t);
  constexpr size_t kTotal = kWidth * kHeight * sizeof(uint32_t);

  // Allocate a 2D hipArray on device 0.
  HIP_CHECK(hipSetDevice(0));
  hipChannelFormatDesc channel_desc = hipCreateChannelDesc<uint32_t>();
  hipArray_t arr0 = nullptr;
  HIP_CHECK(hipMallocArray(&arr0, &channel_desc, kWidth, kHeight));

  // Allocate a linear buffer on device 1 without enabling peer access.
  HIP_CHECK(hipSetDevice(1));
  LinearAllocGuard<uint32_t> dev1_alloc(LinearAllocs::hipMalloc, kTotal);
  uint32_t* dev1_buf = dev1_alloc.ptr();

  HIP_CHECK(hipSetDevice(0));

  SECTION("GPU1 buffer -> GPU0 array (hipMemcpy2DToArray, no peer enable)") {
    // Fill device-1 buffer with a known pattern from host, then copy it into
    // the device-0 array across the peer link.
    std::vector<uint32_t> pattern(kWidth * kHeight);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<uint32_t>(i + 1);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(dev1_buf, pattern.data(), kTotal, hipMemcpyHostToDevice));
    HIP_CHECK(hipSetDevice(0));

    HIP_CHECK(hipMemcpy2DToArray(arr0, 0, 0, dev1_buf, kRowBytes, kRowBytes, kHeight,
                                 hipMemcpyDeviceToDevice));

    // Read back from device-0 array to host to verify data arrived correctly.
    std::vector<uint32_t> host(kWidth * kHeight, 0);
    HIP_CHECK(hipMemcpy2DFromArray(host.data(), kRowBytes, arr0, 0, 0, kRowBytes, kHeight,
                                   hipMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
      INFO("element " << i);
      REQUIRE(host[i] == pattern[i]);
    }
  }

  SECTION("GPU0 array -> GPU1 buffer (hipMemcpy2DFromArray, no peer enable)") {
    // Fill the device-0 array from host, then copy it out into the peer buffer.
    std::vector<uint32_t> pattern(kWidth * kHeight);
    for (size_t i = 0; i < pattern.size(); ++i)
      pattern[i] = static_cast<uint32_t>(0xABC00000u + i);
    HIP_CHECK(hipMemcpy2DToArray(arr0, 0, 0, pattern.data(), kRowBytes, kRowBytes, kHeight,
                                 hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy2DFromArray(dev1_buf, kRowBytes, arr0, 0, 0, kRowBytes, kHeight,
                                   hipMemcpyDeviceToDevice));

    // Read back via device 1 to verify every element.
    std::vector<uint32_t> host(kWidth * kHeight, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kTotal, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < host.size(); ++i) {
      INFO("element " << i);
      REQUIRE(host[i] == pattern[i]);
    }
  }

  HIP_CHECK(hipFreeArray(arr0));
  HIP_CHECK(hipSetDevice(0));
}
