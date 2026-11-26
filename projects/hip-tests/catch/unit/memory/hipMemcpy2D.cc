/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <vector>

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>


HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Basic) {
  constexpr bool async = false;

  SECTION("Device to Host") { Memcpy2DDeviceToHostShell<async>(hipMemcpy2D); }

  SECTION("Device to Device") {
    SECTION("Peer access enabled") { Memcpy2DDeviceToDeviceShell<async, true>(hipMemcpy2D); }
  }

  SECTION("Host to Device") { Memcpy2DHostToDeviceShell<async>(hipMemcpy2D); }

  SECTION("Host to Host") { Memcpy2DHostToHostShell<async>(hipMemcpy2D); }
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy2DHtoDSyncBehavior(hipMemcpy2D, true); }

  SECTION("Device to Host") {
    Memcpy2DDtoHPageableSyncBehavior(hipMemcpy2D, true);
    Memcpy2DDtoHPinnedSyncBehavior(hipMemcpy2D, true);
  }

  SECTION("Device to Device") {
    Memcpy2DDtoDSyncBehavior(hipMemcpy2D, false);
  }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-232
  SECTION("Host to Host") { Memcpy2DHtoHSyncBehavior(hipMemcpy2D, true); }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Parameters) {
  constexpr bool async = false;
  Memcpy2DZeroWidthHeight<async>(hipMemcpy2D);
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Negative_Parameters) {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(nullptr, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, nullptr, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, width - 1, src, spitch, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, src, width - 1, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("dpitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, static_cast<size_t>(attr) + 1, src, spitch, width, height, kind),
          hipErrorInvalidValue);
    }

    SECTION("spitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, static_cast<size_t>(attr) + 1, width, height, kind),
          hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, spitch, width, height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  };

  SECTION("Host to Device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(host_alloc.ptr(), device_alloc.pitch(), device_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}

/*
 * Device-to-device copies whose source and destination start at byte offsets with no common
 * alignment, with pitches that aren't multiples of 16, sized from 1 KiB per compute unit up, where
 * the runtime copies 16-byte chunks with unaligned loads. Checks every byte of the destination
 * allocation against a host reference, including the padding between rows.
 */
HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_UnalignedDeviceToDevice) {
  int computeUnits = 0;
  HIP_CHECK(hipDeviceGetAttribute(&computeUnits, hipDeviceAttributeMultiprocessorCount, 0));
  const size_t width = GENERATE(17, 1000, 4097);
  const size_t scale = GENERATE(1, 4);
  const auto offsets = GENERATE(table<size_t, size_t>({{7, 11}, {0, 13}, {5, 0}}));
  const size_t srcOffset = std::get<0>(offsets);
  const size_t dstOffset = std::get<1>(offsets);
  const size_t height = scale * computeUnits * 1024 / width + 1;
  const size_t srcPitch = width + 3;
  const size_t dstPitch = width + 21;
  const size_t srcSize = srcOffset + srcPitch * height + 32;
  const size_t dstSize = dstOffset + dstPitch * height + 32;
  constexpr uint8_t paddingVal = 0x3C;

  std::vector<uint8_t> src_h(srcSize);
  for (size_t i = 0; i < srcSize; i++) {
    src_h[i] = static_cast<uint8_t>(i * 131 + (i >> 9));
  }
  LinearAllocGuard<uint8_t> src_d(LinearAllocs::hipMalloc, srcSize);
  LinearAllocGuard<uint8_t> dst_d(LinearAllocs::hipMalloc, dstSize);
  HIP_CHECK(hipMemcpy(src_d.ptr(), src_h.data(), srcSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dst_d.ptr(), paddingVal, dstSize));

  HIP_CHECK(hipMemcpy2D(dst_d.ptr() + dstOffset, dstPitch, src_d.ptr() + srcOffset, srcPitch, width,
                        height, hipMemcpyDeviceToDevice));

  std::vector<uint8_t> dst_h(dstSize);
  HIP_CHECK(hipMemcpy(dst_h.data(), dst_d.ptr(), dstSize, hipMemcpyDeviceToHost));
  std::vector<uint8_t> expected(dstSize, paddingVal);
  for (size_t row = 0; row < height; row++) {
    std::copy_n(src_h.begin() + srcOffset + row * srcPitch, width,
                expected.begin() + dstOffset + row * dstPitch);
  }
  const auto mismatch = std::mismatch(dst_h.begin(), dst_h.end(), expected.begin());
  INFO("width " << width << " height " << height << " offsets " << srcOffset << "/" << dstOffset
                << ": first mismatch at byte " << (mismatch.first - dst_h.begin()));
  REQUIRE(mismatch.first == dst_h.end());
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Capture) {

  constexpr size_t width = 16;
  constexpr size_t height = 16;

  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * width);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(
      hipMemcpy2D(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice),
      memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}

/**
 * Test Description
 * ------------------------
 *  - A tall, one-byte-wide host<->device rect has pitches that aren't dword aligned. The SDMA
 *    rect copy can't take those, and copying them one SDMA submission per row took seconds for a
 *    million rows; the copy must be done in one pass. Checks the data both ways and bounds the
 *    time generously (well under the per-row cost).
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_NarrowTallHostRect) {
  constexpr size_t kHeight = 1 << 20;
  constexpr size_t kHostPitch = 1;
  constexpr size_t kDevPitch = 3;
  const hipMemcpyKind kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDeviceToHost);

  LinearAllocGuard<uint8_t> host_src(LinearAllocs::hipHostMalloc, kHeight * kHostPitch);
  LinearAllocGuard<uint8_t> host_dst(LinearAllocs::hipHostMalloc, kHeight * kHostPitch);
  LinearAllocGuard<uint8_t> dev(LinearAllocs::hipMalloc, kHeight * kDevPitch);
  for (size_t i = 0; i < kHeight; i++) {
    host_src.host_ptr()[i] = static_cast<uint8_t>(i * 13 + 1);
    host_dst.host_ptr()[i] = 0;
  }

  if (kind == hipMemcpyDeviceToHost) {
    HIP_CHECK(hipMemcpy2D(dev.ptr(), kDevPitch, host_src.host_ptr(), kHostPitch, 1, kHeight,
                          hipMemcpyHostToDevice));
  }
  const auto start = std::chrono::steady_clock::now();
  if (kind == hipMemcpyHostToDevice) {
    HIP_CHECK(hipMemcpy2D(dev.ptr(), kDevPitch, host_src.host_ptr(), kHostPitch, 1, kHeight,
                          hipMemcpyHostToDevice));
  } else {
    HIP_CHECK(hipMemcpy2D(host_dst.host_ptr(), kHostPitch, dev.ptr(), kDevPitch, 1, kHeight,
                          hipMemcpyDeviceToHost));
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  if (kind == hipMemcpyHostToDevice) {
    HIP_CHECK(hipMemcpy2D(host_dst.host_ptr(), kHostPitch, dev.ptr(), kDevPitch, 1, kHeight,
                          hipMemcpyDeviceToHost));
  }

  for (size_t i = 0; i < kHeight; i++) {
    if (host_dst.host_ptr()[i] != host_src.host_ptr()[i]) {
      INFO("row " << i);
      REQUIRE(host_dst.host_ptr()[i] == host_src.host_ptr()[i]);
    }
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
  INFO("copy took " << elapsed_ms << " ms");
  REQUIRE(elapsed < std::chrono::seconds(1));
}

// Kernel used by Unit_hipMemcpy2D_SameDeviceCopyGrantsPeers below.
static __global__ void AddOnePitchedKernel(int* buf, size_t pitch_in_ints, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < height && col < width) {
    buf[row * pitch_in_ints + col] += 1;
  }
}

/*
 * A same-device D-to-D hipMemcpy2D (src and dst both on device 0) must grant
 * both buffers to all XGMI/PCIe peer GPUs for the duration of the copy, so
 * that a kernel launched on device 1 afterwards can read and write them
 * without hipDeviceEnablePeerAccess ever being called.  On CRUX master the
 * grant is absent and the device-1 kernel faults (hipErrorIllegalAddress /
 * error 700), which is caught by the REQUIRE on hipDeviceSynchronize.
 * Note: a plain linear hipMemcpy D2D on the same device does NOT exhibit this
 * behavior on the reference runtime; this test is specific to hipMemcpy2D.
 */
HIP_TEST_CASE(Unit_hipMemcpy2D_SameDeviceCopyGrantsPeers) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  // Disable peer access in both directions so residual state from a prior test
  // cannot mask the absence of the per-copy grant we are testing.
  HIP_CHECK(hipSetDevice(0));
  { hipError_t e = hipDeviceDisablePeerAccess(1);
    if (e != hipSuccess && e != hipErrorPeerAccessNotEnabled) HIP_CHECK(e);
    static_cast<void>(hipGetLastError()); }
  HIP_CHECK(hipSetDevice(1));
  { hipError_t e = hipDeviceDisablePeerAccess(0);
    if (e != hipSuccess && e != hipErrorPeerAccessNotEnabled) HIP_CHECK(e);
    static_cast<void>(hipGetLastError()); }

  // Restore device 0 on scope exit (fires even when a REQUIRE throws).
  struct DevGuard { ~DevGuard() { static_cast<void>(hipSetDevice(0)); } } dg;

  constexpr size_t kWidth      = 16;                       // elements per row
  constexpr size_t kHeight     = 16;                       // rows
  constexpr size_t kWidthBytes = kWidth * sizeof(int);     // bytes of data per row

  // Allocate src and dst as 2D pitched buffers on device 0.
  HIP_CHECK(hipSetDevice(0));
  LinearAllocGuard2D<int> src_alloc(kWidth, kHeight);
  LinearAllocGuard2D<int> dst_alloc(kWidth, kHeight);

  // Fill src from host with a simple sequential pattern.
  std::vector<int> host_src(kWidth * kHeight);
  for (size_t i = 0; i < host_src.size(); ++i) host_src[i] = static_cast<int>(i + 1);
  HIP_CHECK(hipMemcpy2D(src_alloc.ptr(), src_alloc.pitch(),
                         host_src.data(), kWidthBytes,
                         kWidthBytes, kHeight, hipMemcpyHostToDevice));

  // The copy under test: same-device D2D on device 0.
  // The reference runtime grants both buffers to peer GPUs during this copy.
  HIP_CHECK(hipMemcpy2D(dst_alloc.ptr(), dst_alloc.pitch(),
                         src_alloc.ptr(), src_alloc.pitch(),
                         kWidthBytes, kHeight, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  // Launch a kernel on device 1 that increments every element of dst_alloc.
  // No hipDeviceEnablePeerAccess is called: the grant from the copy above is
  // the only thing that makes this access legal.  On CRUX master (no grant)
  // the kernel triggers a page fault and hipDeviceSynchronize returns
  // hipErrorIllegalAddress (error 700) or the process aborts.
  HIP_CHECK(hipSetDevice(1));
  const size_t pitch_in_ints = dst_alloc.pitch() / sizeof(int);
  AddOnePitchedKernel<<<dim3(1, 1), dim3(static_cast<unsigned>(kWidth),
                                         static_cast<unsigned>(kHeight))>>>(
      dst_alloc.ptr(), pitch_in_ints,
      static_cast<int>(kWidth), static_cast<int>(kHeight));
  hipError_t sync_err = hipDeviceSynchronize();
  REQUIRE(sync_err == hipSuccess);  // fails (error 700) on CRUX master

  // Read dst back to host and verify every element was incremented by the kernel.
  HIP_CHECK(hipSetDevice(0));
  std::vector<int> host_dst(kWidth * kHeight, 0);
  HIP_CHECK(hipMemcpy2D(host_dst.data(), kWidthBytes,
                         dst_alloc.ptr(), dst_alloc.pitch(),
                         kWidthBytes, kHeight, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < host_dst.size(); ++i) {
    INFO("element " << i);
    REQUIRE(host_dst[i] == host_src[i] + 1);
  }
}
