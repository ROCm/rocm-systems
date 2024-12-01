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

#pragma once

#include <performance_common.hh>

static hip_Memcpy2D CreateMemcpy2DParam(void* dst, size_t dpitch, void* src, size_t spitch,
                                        size_t width, size_t height, hipMemcpyKind kind) {
  hip_Memcpy2D params = {};
  memset(&params, 0, sizeof(hip_Memcpy2D));

  const hipExtent src_offset = {0, 0, 0};
  const hipExtent dst_offset = {0, 0, 0};

  params.dstPitch = dpitch;
  switch (kind) {
    case hipMemcpyDeviceToHost:
    case hipMemcpyHostToHost:
        params.dstMemoryType = hipMemoryTypeHost;
      params.dstHost = dst;
      break;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyHostToDevice:
        params.dstMemoryType = hipMemoryTypeDevice;
      params.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
      break;
    default:
      REQUIRE(false);
  }

  params.srcPitch = dpitch;
  switch (kind) {
    case hipMemcpyDeviceToHost:
    case hipMemcpyHostToHost:
        params.srcMemoryType = hipMemoryTypeHost;
      params.srcHost = src;
      break;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyHostToDevice:
        params.srcMemoryType = hipMemoryTypeDevice;
      params.srcDevice = reinterpret_cast<hipDeviceptr_t>(src);
      break;
    default:
      REQUIRE(false);
  }

  params.WidthInBytes = width;
  params.Height = height;
  params.srcXInBytes = src_offset.width;
  params.srcY = src_offset.height;
  params.dstXInBytes = dst_offset.width;
  params.dstY = dst_offset.height;

  return params;
}

static hipMemcpy3DParms CreateMemcpy3DParam(hipPitchedPtr dst_ptr, hipPos dst_pos,
                                            hipPitchedPtr src_ptr, hipPos src_pos,
                                            hipExtent extent, hipMemcpyKind kind) {
  hipMemcpy3DParms params = {};
  memset(&params, 0, sizeof(hipMemcpy3DParms));
  params.dstPtr = dst_ptr;
  params.dstPos = dst_pos;
  params.srcPtr = src_ptr;
  params.srcPos = src_pos;
  params.extent = extent;
  params.kind = kind;
  return params;
}

static std::tuple<int, int> GetDeviceIds(bool enable_peer_access) {
  int src_device = 0;
  int dst_device = 1;

  if (enable_peer_access) {
    int can_access_peer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
    if (!can_access_peer) {
      std::string msg = "Skipped as peer access cannot be enabled between devices " +
          std::to_string(src_device) + " " + std::to_string(dst_device);
      HipTest::HIP_SKIP_TEST(msg.c_str());
      return {-1, -1};
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));
  } else {
    dst_device = 0;
  }

  return {src_device, dst_device};
}
