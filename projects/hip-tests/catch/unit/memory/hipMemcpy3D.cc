/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <vector>

#include <memcpy1d_tests_common.hh>
#include <memcpy3d_tests_common.hh>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

#pragma clang diagnostic ignored "-Wunused-variable"

HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_Basic) {
  constexpr bool async = false;

  SECTION("Device to Host") { Memcpy3DDeviceToHostShell<async>(Memcpy3DWrapper<>); }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(Memcpy3DWrapper<>);
    }
    SECTION("Peer access enabled") { Memcpy3DDeviceToDeviceShell<async, true>(Memcpy3DWrapper<>); }
  }

  SECTION("Host to Device") { Memcpy3DHostToDeviceShell<async>(Memcpy3DWrapper<>); }

  SECTION("Host to Host") { Memcpy3DHostToHostShell<async>(Memcpy3DWrapper<>); }
}

HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy3DHtoDSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Device to Pageable Host") { Memcpy3DDtoHPageableSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Device to Pinned Host") { Memcpy3DDtoHPinnedSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Host to Host") { Memcpy3DHtoHSyncBehavior(Memcpy3DWrapper<>, true); }
}

// Proves a device-to-device hipMemcpy3D does not synchronize on the host, which
// is its documented behavior: the runtime performs no host-side synchronization
// for device-to-device transfers, so the call returns before the copy executes.
// The stream is blocked first, so a host-synchronous copy could never return
// (the copy cannot run until the stream is unblocked). The call therefore
// returning while hipStreamQuery still reports hipErrorNotReady is only possible
// if it was asynchronous, which is what makes this assertion valid.
HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_DeviceToDevice_Synchronization_Behavior) {
  LinearAllocGuard3D<int> src_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  LinearAllocGuard3D<int> dst_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  HipTest::BlockingContext b_context{nullptr};
  hipStream_t kernel_stream{nullptr};

  auto parms =
      GetMemcpy3DParms(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                       make_hipPos(0, 0, 0), dst_alloc.extent(), hipMemcpyDeviceToDevice);

  b_context.block_stream();
  REQUIRE(b_context.is_blocked());

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemcpy3D(&parms), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);

  HIP_CHECK_ERROR(hipStreamQuery(kernel_stream), hipErrorNotReady);
  b_context.unblock_stream();
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(hipStreamQuery(kernel_stream) == hipSuccess);
}

HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_Parameters) {
  constexpr bool async = false;
  Memcpy3DZeroWidthHeightDepth<async>(Memcpy3DWrapper<>);
}

HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_Array) {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;
  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(Memcpy3DWrapper<>); }
#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-238
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper<>); }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpy3D_Negative_Parameters) {
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-239
    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }
#endif

    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-237
    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }
#endif

#if HT_NVIDIA  // Disable on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(Memcpy3DWrapper(dst_ptr, dst_pos, src_ptr, src_pos, extent,
                                      static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }
#endif
  };

  SECTION("Host to Device") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    NegativeTests(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard3D<int> src_alloc(extent);
    LinearAllocGuard3D<int> dst_alloc(extent);
    NegativeTests(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                  make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice);
  }
}

/*
 * Device-to-device volume copies whose source and destination start at byte offsets with no
 * common alignment, with row and slice pitches that aren't multiples of 16, sized from 1 KiB per
 * compute unit up, where the runtime copies 16-byte chunks with unaligned loads. Checks every byte
 * of the destination allocation against a host reference, including the row and slice padding.
 */
HIP_TEST_CASE(Unit_hipMemcpy3D_Positive_UnalignedDeviceToDevice) {
  int computeUnits = 0;
  HIP_CHECK(hipDeviceGetAttribute(&computeUnits, hipDeviceAttributeMultiprocessorCount, 0));
  const size_t width = GENERATE(17, 1000, 4097);
  const size_t scale = GENERATE(1, 4);
  const auto offsets = GENERATE(table<size_t, size_t>({{7, 11}, {0, 13}, {5, 0}}));
  const size_t srcOffset = std::get<0>(offsets);
  const size_t dstOffset = std::get<1>(offsets);
  constexpr size_t depth = 3;
  const size_t height = scale * computeUnits * 1024 / (width * depth) + 1;
  const size_t rowsPerSlice = height + 1;
  const size_t srcPitch = width + 3;
  const size_t dstPitch = width + 21;
  const size_t srcSlice = srcPitch * rowsPerSlice;
  const size_t dstSlice = dstPitch * rowsPerSlice;
  const size_t srcSize = srcOffset + srcSlice * depth + 32;
  const size_t dstSize = dstOffset + dstSlice * depth + 32;
  constexpr uint8_t paddingVal = 0x3C;

  std::vector<uint8_t> src_h(srcSize);
  for (size_t i = 0; i < srcSize; i++) {
    src_h[i] = static_cast<uint8_t>(i * 131 + (i >> 9));
  }
  LinearAllocGuard<uint8_t> src_d(LinearAllocs::hipMalloc, srcSize);
  LinearAllocGuard<uint8_t> dst_d(LinearAllocs::hipMalloc, dstSize);
  HIP_CHECK(hipMemcpy(src_d.ptr(), src_h.data(), srcSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dst_d.ptr(), paddingVal, dstSize));

  auto params = GetMemcpy3DParms(
      make_hipPitchedPtr(dst_d.ptr() + dstOffset, dstPitch, width, rowsPerSlice),
      make_hipPos(0, 0, 0),
      make_hipPitchedPtr(src_d.ptr() + srcOffset, srcPitch, width, rowsPerSlice),
      make_hipPos(0, 0, 0), make_hipExtent(width, height, depth), hipMemcpyDeviceToDevice);
  HIP_CHECK(hipMemcpy3D(&params));

  std::vector<uint8_t> dst_h(dstSize);
  HIP_CHECK(hipMemcpy(dst_h.data(), dst_d.ptr(), dstSize, hipMemcpyDeviceToHost));
  std::vector<uint8_t> expected(dstSize, paddingVal);
  for (size_t slice = 0; slice < depth; slice++) {
    for (size_t row = 0; row < height; row++) {
      std::copy_n(src_h.begin() + srcOffset + slice * srcSlice + row * srcPitch, width,
                  expected.begin() + dstOffset + slice * dstSlice + row * dstPitch);
    }
  }
  const auto mismatch = std::mismatch(dst_h.begin(), dst_h.end(), expected.begin());
  INFO("width " << width << " height " << height << " offsets " << srcOffset << "/" << dstOffset
                << ": first mismatch at byte " << (mismatch.first - dst_h.begin()));
  REQUIRE(mismatch.first == dst_h.end());
}

HIP_TEST_CASE(Unit_hipMemcpy3D_Capture) {
  constexpr hipExtent extent{16 * sizeof(int), 16, 16};
  LinearAllocGuard3D<int> dev_alloc(extent);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   dev_alloc.pitch() * dev_alloc.height() * dev_alloc.depth());

  auto params = GetMemcpy3DParms(dev_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                                 make_hipPitchedPtr(host_alloc.ptr(), dev_alloc.pitch(),
                                                    dev_alloc.width(), dev_alloc.height()),
                                 make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemcpy3D(&params), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}
