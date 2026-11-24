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
