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
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>
#include <variant>
using PtrVariant = std::variant<hipPitchedPtr, hipArray_t>;
static inline hipMemcpy3DParms GetMemcpy3DParms_spt(PtrVariant dst_ptr, hipPos dst_pos,
                                                    PtrVariant src_ptr, hipPos src_pos,
                                                    hipExtent extent, hipMemcpyKind kind) {
  hipMemcpy3DParms parms = {0};
  if (std::holds_alternative<hipArray_t>(dst_ptr)) {
    parms.dstArray = std::get<hipArray_t>(dst_ptr);
  } else {
    parms.dstPtr = std::get<hipPitchedPtr>(dst_ptr);
  }
  parms.dstPos = dst_pos;
  if (std::holds_alternative<hipArray_t>(src_ptr)) {
    parms.srcArray = std::get<hipArray_t>(src_ptr);
  } else {
    parms.srcPtr = std::get<hipPitchedPtr>(src_ptr);
  }
  parms.srcPos = src_pos;
  parms.extent = extent;
  parms.kind = kind;
  return parms;
}
template <bool async = false>
hipError_t Memcpy3DWrapper_spt(PtrVariant dst_ptr, hipPos dst_pos, PtrVariant src_ptr,
                               hipPos src_pos, hipExtent extent, hipMemcpyKind kind,
                               hipStream_t stream = nullptr) {
  auto parms = GetMemcpy3DParms_spt(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);
  if constexpr (async) {
    return hipMemcpy3DAsync_spt(&parms, stream);
  } else {
    return hipMemcpy3D_spt(&parms);
  }
}
template <bool should_synchronize, typename F>
void Memcpy3DZeroWidthHeightDepth_spt(F memcpy_func, const hipStream_t stream = nullptr) {
  constexpr hipExtent extent{127 * sizeof(int), 128, 8};
  const auto [width_mult, height_mult, depth_mult] =
      GENERATE(std::make_tuple(0, 1, 1), std::make_tuple(1, 0, 1), std::make_tuple(1, 1, 0));
  SECTION("Device to Host") {
    LinearAllocGuard3D<uint8_t> device_alloc(extent);
    LinearAllocGuard<uint8_t> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.width() * device_alloc.height() * device_alloc.depth());
    std::fill_n(host_alloc.ptr(),
                device_alloc.width_logical() * device_alloc.height() * device_alloc.depth(), 42);
    HIP_CHECK(hipMemset3D(device_alloc.pitched_ptr(), 1, device_alloc.extent()));
    HIP_CHECK(memcpy_func(
        make_hipPitchedPtr(host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                           device_alloc.height()),
        make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
        make_hipExtent(device_alloc.width() * width_mult, device_alloc.height() * height_mult,
                       device_alloc.depth() * depth_mult),
        hipMemcpyDeviceToHost, stream));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    ArrayFindIfNot(host_alloc.ptr(), static_cast<uint8_t>(42),
                   device_alloc.width_logical() * device_alloc.height() * device_alloc.depth());
  }
  SECTION("Device to Device") {
    LinearAllocGuard3D<uint8_t> src_alloc(extent);
    LinearAllocGuard3D<uint8_t> dst_alloc(extent);
    LinearAllocGuard<uint8_t> host_alloc(
        LinearAllocs::hipHostMalloc, dst_alloc.width() * dst_alloc.height() * dst_alloc.depth());
    HIP_CHECK(hipMemset3D(src_alloc.pitched_ptr(), 1, src_alloc.extent()));
    HIP_CHECK(hipMemset3D(dst_alloc.pitched_ptr(), 42, dst_alloc.extent()));
    HIP_CHECK(
        memcpy_func(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                    make_hipPos(0, 0, 0),
                    make_hipExtent(dst_alloc.width() * width_mult, dst_alloc.height() * height_mult,
                                   dst_alloc.depth() * depth_mult),
                    hipMemcpyDeviceToDevice, stream));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(Memcpy3DWrapper_spt(make_hipPitchedPtr(host_alloc.ptr(), dst_alloc.width(),
                                                     dst_alloc.width(), dst_alloc.height()),
                                  make_hipPos(0, 0, 0), dst_alloc.pitched_ptr(),
                                  make_hipPos(0, 0, 0), dst_alloc.extent(), hipMemcpyDeviceToHost));
    ArrayFindIfNot(host_alloc.ptr(), static_cast<uint8_t>(42),
                   dst_alloc.width_logical() * dst_alloc.height());
  }
  SECTION("Host to Device") {
    LinearAllocGuard3D<uint8_t> device_alloc(extent);
    LinearAllocGuard<uint8_t> src_host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.width() * device_alloc.height() * device_alloc.depth());
    LinearAllocGuard<uint8_t> dst_host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.width() * device_alloc.height() * device_alloc.depth());
    std::fill_n(src_host_alloc.ptr(),
                device_alloc.width_logical() * device_alloc.height() * device_alloc.depth(), 1);
    HIP_CHECK(hipMemset3D(device_alloc.pitched_ptr(), 42, device_alloc.extent()));
    HIP_CHECK(memcpy_func(
        device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
        make_hipPitchedPtr(src_host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                           device_alloc.height()),
        make_hipPos(0, 0, 0),
        make_hipExtent(device_alloc.width() * width_mult, device_alloc.height() * height_mult,
                       device_alloc.depth() * depth_mult),
        hipMemcpyHostToDevice, stream));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(Memcpy3DWrapper_spt(make_hipPitchedPtr(dst_host_alloc.ptr(), device_alloc.width(),
                                                     device_alloc.width(), device_alloc.height()),
                                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(),
                                  make_hipPos(0, 0, 0), device_alloc.extent(),
                                  hipMemcpyDeviceToHost));
    ArrayFindIfNot(dst_host_alloc.ptr(), static_cast<uint8_t>(42),
                   device_alloc.width_logical() * device_alloc.height());
  }
  SECTION("Host to Host") {
    const auto alloc_size = extent.width * extent.height * extent.depth;
    LinearAllocGuard<uint8_t> src_alloc(LinearAllocs::hipHostMalloc, alloc_size);
    LinearAllocGuard<uint8_t> dst_alloc(LinearAllocs::hipHostMalloc, alloc_size);
    std::fill_n(src_alloc.ptr(), alloc_size, 1);
    std::fill_n(dst_alloc.ptr(), alloc_size, 42);
    HIP_CHECK(
        memcpy_func(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                    make_hipPos(0, 0, 0),
                    make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                    make_hipPos(0, 0, 0),
                    make_hipExtent(extent.width * width_mult, extent.height * height_mult,
                                   extent.depth * depth_mult),
                    hipMemcpyHostToHost, stream));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    ArrayFindIfNot(dst_alloc.ptr(), static_cast<uint8_t>(42), alloc_size);
  }
}
