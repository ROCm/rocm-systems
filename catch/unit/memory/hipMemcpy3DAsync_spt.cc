/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "memcpy3d_spt_tests_common.hh"
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy3d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>
/**
 * @addtogroup hipMemcpy3DAsync_spt hipMemcpy3DAsync_spt
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy3DAsync_spt(const struct hipMemcpy3DParms* p,
 * hipStream_t stream __dparm(0))` - Copies data between host and device
 * asynchronously.
 */
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the negative parameters of the hipMemcpy3DAsync_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy3DAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_spt_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  constexpr bool async = true;
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};
  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidPitchValue);
    }
    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
          hipErrorInvalidPitchValue);
    }
    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(
          Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
          hipErrorInvalidValue);
    }
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent,
                                                 static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }
    SECTION("Invalid stream") {
      StreamGuard stream_guard(Streams::created);
      HIP_CHECK(hipStreamDestroy(stream_guard.stream()));
      HIP_CHECK_ERROR(Memcpy3DWrapper_spt<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind,
                                                 stream_guard.stream()),
                      hipErrorContextIsDestroyed);
    }
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
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the positive parameters of the hipMemcpy3DAsync_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy3DAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_spt_Positive_Parameters") {
  CHECK_IMAGE_SUPPORT
  constexpr bool async = true;
  Memcpy3DZeroWidthHeightDepth_spt<async>(Memcpy3DWrapper_spt<async>);
}
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the data copy between Array with hipMemcpy3DAsync_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy3DAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_spt_Positive_Array") {
  CHECK_IMAGE_SUPPORT
  constexpr bool async = true;
  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(Memcpy3DWrapper_spt<async>); }
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper_spt<async>); }
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
