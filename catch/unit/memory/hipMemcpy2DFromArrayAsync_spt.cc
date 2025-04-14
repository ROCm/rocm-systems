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
#include "array_memcpy_tests_common.hh"
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>
#include <utils.hh>
/**
 * @addtogroup hipMemcpy2DFromArrayAsync_spt hipMemcpy2DFromArrayAsync_spt
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy2DFromArrayAsync( void* dst, size_t dpitch,
 * hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t
 * height, hipMemcpyKind kind, hipStream_t stream __dparm(0));` - Copies data
 * between host and device.
 */
/**
 * Test Description
 * ------------------------
 * - Test basic memcpy between host/device and 2D array with
 * hipMemcpy2DFromArrayAsync_spt api
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy2DFromArrayAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy2DFromArrayAsync_spt_Positive_Default") {
  CHECK_IMAGE_SUPPORT
  using namespace std::placeholders;
  const auto stream_type =
      GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();
  const auto width = GENERATE(16, 32, 48);
  const auto height = GENERATE(1, 16, 32, 48);
  SECTION("Array to host") {
    Memcpy2DHostFromAShell<true, int>(
        std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0,
                  width * sizeof(int), height, hipMemcpyDeviceToHost, stream),
        width, height, stream);
  }
  SECTION("Array to host with default kind") {
    Memcpy2DHostFromAShell<true, int>(
        std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0,
                  width * sizeof(int), height, hipMemcpyDefault, stream),
        width, height, stream);
  }
}
/**
 * Test Description
 * ------------------------
 * - Test synchronization behavior for hipMemcpy2DFromArrayAsync_spt api
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy2DFromArrayAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE(
    "Unit_hipMemcpy2DFromArrayAsync_spt_Positive_Synchronization_Behavior") {
  CHECK_IMAGE_SUPPORT
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());
  SECTION("Array to host") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);
    MemcpyAtoHPageableSyncBehavior(std::bind(hipMemcpy2DFromArrayAsync_spt, _1,
                                             width * sizeof(int), _2, 0, 0,
                                             width * sizeof(int), height,
                                             hipMemcpyDeviceToHost, nullptr),
                                   width, height, false);
    MemcpyAtoHPinnedSyncBehavior(std::bind(hipMemcpy2DFromArrayAsync_spt, _1,
                                           width * sizeof(int), _2, 0, 0,
                                           width * sizeof(int), height,
                                           hipMemcpyDeviceToHost, nullptr),
                                 width, height, false);
  }
  SECTION("Array to device") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);
    MemcpyAtoDSyncBehavior(std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3,
                                     0, 0, width * sizeof(int), height,
                                     hipMemcpyDeviceToDevice, nullptr),
                           width, height, false);
  }
}
/**
 * Test Description
 * ------------------------
 * - Test that no data is copied when width/height is set to 0
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy2DFromArrayAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy2DFromArrayAsync_spt_Positive_ZeroWidthHeight") {
  CHECK_IMAGE_SUPPORT
  using namespace std::placeholders;
  const auto stream_type =
      GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();
  const auto width = 16;
  const auto height = 16;
  SECTION("Array to host") {
    SECTION("Height is 0") {
      Memcpy2DFromArrayZeroWidthHeight<true>(
          std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0,
                    width * sizeof(int), 0, hipMemcpyDeviceToHost, stream),
          width, height, stream);
    }
    SECTION("Width is 0") {
      Memcpy2DFromArrayZeroWidthHeight<true>(
          std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0, 0, height,
                    hipMemcpyDeviceToHost, stream),
          width, height, stream);
    }
  }
  SECTION("Array to device") {
    SECTION("Height is 0") {
      Memcpy2DFromArrayZeroWidthHeight<true>(
          std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0,
                    width * sizeof(int), 0, hipMemcpyDeviceToDevice, stream),
          width, height, stream);
    }
    SECTION("Width is 0") {
      Memcpy2DFromArrayZeroWidthHeight<true>(
          std::bind(hipMemcpy2DFromArrayAsync_spt, _1, _2, _3, 0, 0, 0, height,
                    hipMemcpyDeviceToDevice, stream),
          width, height, stream);
    }
  }
}
/**
 * Test Description
 * ------------------------
 * - Test unsuccessful execution of hipMemcpy2DFromArrayAsync_spt api when
 * parameters are invalid. Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpy2DFromArrayAsync_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpy2DFromArrayAsync_spt_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  using namespace std::placeholders;
  const auto width = 32;
  const auto height = 32;
  const auto allocation_size = 2 * width * height * sizeof(int);
  const unsigned int flag = hipArrayDefault;
  ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   allocation_size);
  SECTION("Array to host") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          nullptr, 2 * width * sizeof(int), array_alloc.ptr(),
                          0, 0, width * sizeof(int), height,
                          hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int), nullptr, 0,
                          0, width * sizeof(int), height, hipMemcpyDeviceToHost,
                          nullptr),
                      hipErrorInvalidHandle);
    }
    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), width * sizeof(int) - 10,
                          array_alloc.ptr(), 0, 0, width * sizeof(int), height,
                          hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int),
                          array_alloc.ptr(), 1, 0, width * sizeof(int), height,
                          hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int),
                          array_alloc.ptr(), 0, 1, width * sizeof(int), height,
                          hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int),
                          array_alloc.ptr(), 0, 0, width * sizeof(int) + 1,
                          height, hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int),
                          array_alloc.ptr(), 0, 0, width * sizeof(int),
                          height + 1, hipMemcpyDeviceToHost, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          host_alloc.ptr(), 2 * width * sizeof(int),
                          array_alloc.ptr(), 0, 0, width * sizeof(int), height,
                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }
  SECTION("Array to device") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          nullptr, device_alloc.pitch(), array_alloc.ptr(), 0,
                          0, width * sizeof(int), height,
                          hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(), nullptr, 0,
                          0, width * sizeof(int), height,
                          hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidHandle);
    }
    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), width * sizeof(int) - 10,
                          array_alloc.ptr(), 0, 0, width * sizeof(int), height,
                          hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(),
                          array_alloc.ptr(), 1, 0, width * sizeof(int), height,
                          hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(),
                          array_alloc.ptr(), 0, 1, width * sizeof(int), height,
                          hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(),
                          array_alloc.ptr(), 0, 0, width * sizeof(int) + 1,
                          height, hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(),
                          array_alloc.ptr(), 0, 0, width * sizeof(int),
                          height + 1, hipMemcpyDeviceToDevice, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync_spt(
                          device_alloc.ptr(), device_alloc.pitch(),
                          array_alloc.ptr(), 0, 0, width * sizeof(int), height,
                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
  }
}
/**
 * End doxygen group MemoryTest.
 * @}
 */

