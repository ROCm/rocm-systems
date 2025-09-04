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

#include <memcpy1d_tests_common.hh>
#include <memcpy3d_tests_common.hh>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

#pragma clang diagnostic ignored "-Wunused-variable"

TEST_CASE("Unit_hipMemcpy3D_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

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

TEST_CASE("Unit_hipMemcpy3D_Positive_Synchronization_Behavior") {
  CHECK_IMAGE_SUPPORT

  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy3DHtoDSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Device to Pageable Host") { Memcpy3DDtoHPageableSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Device to Pinned Host") { Memcpy3DDtoHPinnedSyncBehavior(Memcpy3DWrapper<>, true); }

  SECTION("Host to Host") { Memcpy3DHtoHSyncBehavior(Memcpy3DWrapper<>, true); }
}

TEST_CASE("Unit_hipMemcpy3D_Positive_DeviceToDevice_Synchronization_Behavior") {
  CHECK_IMAGE_SUPPORT

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

TEST_CASE("Unit_hipMemcpy3D_Positive_Parameters") {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;
  Memcpy3DZeroWidthHeightDepth<async>(Memcpy3DWrapper<>);
}

TEST_CASE("Unit_hipMemcpy3D_Positive_Array") {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;
  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(Memcpy3DWrapper<>); }
#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-238
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper<>); }
#endif
}

TEST_CASE("Unit_hipMemcpy3D_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

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

TEST_CASE("Unit_hipMemcpy3D_Capture") {
  CHECK_IMAGE_SUPPORT

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
