/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Positive_Basic) {
  constexpr bool async = false;

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-236
  SECTION("Device to Host") { Memcpy2DDeviceToHostShell<async>(MemcpyParam2DAdapter<async>()); }
#endif

  SECTION("Device to Device") {
    SECTION("Peer access enabled") {
      Memcpy2DDeviceToDeviceShell<async, true>(MemcpyParam2DAdapter<async>());
    }
  }

  SECTION("Host to Device") { Memcpy2DHostToDeviceShell<async>(MemcpyParam2DAdapter<async>()); }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-236
  SECTION("Host to Host") { Memcpy2DHostToHostShell<async>(MemcpyParam2DAdapter<async>()); }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy2DHtoDSyncBehavior(MemcpyParam2DAdapter<>(), true); }

  SECTION("Device to Pageable Host") {
    Memcpy2DDtoHPageableSyncBehavior(MemcpyParam2DAdapter<>(), true);
  }

  SECTION("Device to Pinned Host") {
    Memcpy2DDtoHPinnedSyncBehavior(MemcpyParam2DAdapter<>(), true);
  }

  SECTION("Device to Device") { Memcpy2DDtoDSyncBehavior(MemcpyParam2DAdapter<>(), false); }

  SECTION("Host to Host") { Memcpy2DHtoHSyncBehavior(MemcpyParam2DAdapter<>(), true); }
}

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Positive_Parameters) {
  constexpr bool async = false;
  Memcpy2DZeroWidthHeight<async>(MemcpyParam2DAdapter<async>());
}

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Positive_Array) {
  CHECK_IMAGE_SUPPORT
  constexpr bool async = false;
  SECTION("Array from/to Host") {
    MemcpyParam2DArrayHostShell<async>(MemcpyParam2DAdapter<async>());
  }
  SECTION("Array from/to Device") {
    MemcpyParam2DArrayDeviceShell<async>(MemcpyParam2DAdapter<async>());
  }
}

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Negative_Parameters) {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(static_cast<void*>(nullptr), dpitch, src, spitch,
                                               width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(dst, dpitch, static_cast<void*>(nullptr), spitch,
                                               width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dstPitch < WithInBytes") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(dst, width - 1, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("srcPitch < WidthInBytes") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(dst, dpitch, src, width - 1, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dstPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(dst, static_cast<size_t>(attr) + 1, src, spitch,
                                               width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("srcPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>()(dst, dpitch, src, static_cast<size_t>(attr) + 1,
                                               width, height, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-237
    SECTION("WidthInBytes + srcXInBytes > srcPitch") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>(make_hipExtent(spitch - width + 1, 0, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("WidthInBytes + dstXInBytes > dstPitch") {
      HIP_CHECK_ERROR(
          MemcpyParam2DAdapter<>(make_hipExtent(0, 0, 0), make_hipExtent(dpitch - width + 1, 0, 0))(
              dst, dpitch, src, spitch, width, height, kind),
          hipErrorInvalidValue);
    }

    SECTION("srcY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>(make_hipExtent(0, 1, 0))(dst, dpitch, src, spitch,
                                                                      width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dstY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<>(make_hipExtent(0, 0, 0), make_hipExtent(0, 1, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
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

HIP_TEST_CASE(Unit_hipMemcpyParam2D_Capture) {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  LinearAllocGuard2D<int> device_alloc(cols, rows);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);

  hip_Memcpy2D params = {};
  memset(&params, 0x0, sizeof(hip_Memcpy2D));

  InitializeMemcpy2DParams(&params, device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(),
                           device_alloc.pitch(), device_alloc.width(), device_alloc.height(),
                           hipMemcpyHostToDevice);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemcpyParam2D(&params), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}
