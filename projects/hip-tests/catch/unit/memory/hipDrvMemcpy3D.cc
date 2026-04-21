/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <memcpy1d_tests_common.hh>
#include <memcpy3d_tests_common.hh>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Positive_Basic) {
  constexpr bool async = false;

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-236
  SECTION("Device to Host") { Memcpy3DDeviceToHostShell<async>(DrvMemcpy3DWrapper<>); }
#endif

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(DrvMemcpy3DWrapper<>);
    }
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(DrvMemcpy3DWrapper<>);
    }
  }

  SECTION("Host to Device") { Memcpy3DHostToDeviceShell<async>(DrvMemcpy3DWrapper<>); }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-236
  SECTION("Host to Host") { Memcpy3DHostToHostShell<async>(DrvMemcpy3DWrapper<>); }
#endif
}

HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy3DHtoDSyncBehavior(DrvMemcpy3DWrapper<>, true); }

  SECTION("Device to Pageable Host") {
    Memcpy3DDtoHPageableSyncBehavior(DrvMemcpy3DWrapper<>, true);
  }

  SECTION("Device to Pinned Host") { Memcpy3DDtoHPinnedSyncBehavior(DrvMemcpy3DWrapper<>, true); }

  SECTION("Device to Device") { Memcpy3DDtoDSyncBehavior(DrvMemcpy3DWrapper<>, false); }

  SECTION("Host to Host") { Memcpy3DHtoHSyncBehavior(DrvMemcpy3DWrapper<>, true); }
}

HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Positive_Parameters) {
  constexpr bool async = false;
  Memcpy3DZeroWidthHeightDepth<async>(DrvMemcpy3DWrapper<>);
}

// Disabled on AMD due to defect - EXSWHTEC-238
HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Positive_Array) {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;
  SECTION("Array from/to Host") { DrvMemcpy3DArrayHostShell<async>(DrvMemcpy3DWrapper<>); }
  SECTION("Array from/to Device") { DrvMemcpy3DArrayDeviceShell<async>(DrvMemcpy3DWrapper<>); }
}

HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Negative_Parameters) {
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-237
    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(DrvMemcpy3DWrapper(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
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

HIP_TEST_CASE(Unit_hipDrvMemcpy3D_Capture) {
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};
  LinearAllocGuard3D<int> device_alloc(extent);
  LinearAllocGuard<int> host_alloc(
      LinearAllocs::hipHostMalloc,
      device_alloc.pitch() * device_alloc.height() * device_alloc.depth());

  auto params = GetDrvMemcpy3DParms(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                                    make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(),
                                                       device_alloc.width(), device_alloc.height()),
                                    make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipDrvMemcpy3D(&params), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}
