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
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <variant>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

using PtrVariant = std::variant<hipPitchedPtr, hipArray_t>;

static inline hipMemcpyKind ReverseMemcpyDirection(const hipMemcpyKind direction) {
  switch (direction) {
    case hipMemcpyHostToDevice:
      return hipMemcpyDeviceToHost;
    case hipMemcpyDeviceToHost:
      return hipMemcpyHostToDevice;
    default:
      return direction;
  }
};

static inline hipMemcpy3DParms GetMemcpy3DParms(PtrVariant dst_ptr, hipPos dst_pos,
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

static bool operator==(const hipPitchedPtr& lhs, const hipPitchedPtr& rhs) {
  // not checking for xsize currently as hipGraphMemcpyNodeGetParams returns incorrect value
  return lhs.ptr == rhs.ptr && lhs.pitch == rhs.pitch && lhs.ysize == rhs.ysize;
}

static bool operator==(const hipPos& lhs, const hipPos& rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool operator==(const hipExtent& lhs, const hipExtent& rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth;
}

static inline bool operator==(const hipMemcpy3DParms& lhs, const hipMemcpy3DParms& rhs) {
  return lhs.dstArray == rhs.dstArray && lhs.dstPtr == rhs.dstPtr && lhs.dstPos == rhs.dstPos &&
      lhs.srcArray == rhs.srcArray && lhs.srcPtr == rhs.srcPtr && lhs.srcPos == rhs.srcPos &&
      lhs.extent == rhs.extent && lhs.kind == rhs.kind;
}

template <bool async = false, bool graph = false, bool set_params = false>
hipError_t Memcpy3DWrapper(PtrVariant dst_ptr, hipPos dst_pos, PtrVariant src_ptr, hipPos src_pos,
                           hipExtent extent, hipMemcpyKind kind, hipStream_t stream = nullptr) {
  auto parms = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);

  if constexpr (graph) {
    hipGraph_t g = nullptr;
    HIP_CHECK(hipGraphCreate(&g, 0));
    hipGraphNode_t node = nullptr;

    if constexpr (set_params) {
      auto reversed_parms = GetMemcpy3DParms(src_ptr, src_pos, dst_ptr, dst_pos, extent,
                                             ReverseMemcpyDirection(kind));
      HIP_CHECK(hipGraphAddMemcpyNode(&node, g, nullptr, 0, &reversed_parms));
      HIP_CHECK(hipGraphMemcpyNodeSetParams(node, &parms));
    } else {
      HIP_CHECK(hipGraphAddMemcpyNode(&node, g, nullptr, 0, &parms));
    }

    hipMemcpy3DParms retrieved_params = {0};
    HIP_CHECK(hipGraphMemcpyNodeGetParams(node, &retrieved_params));
    REQUIRE(parms == retrieved_params);

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, g, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(g));

    return hipSuccess;
  }

  if constexpr (async) {
    return hipMemcpy3DAsync(&parms, stream);
  } else {
    return hipMemcpy3D(&parms);
  }
}

template <bool should_synchronize, typename F>
void Memcpy3DDeviceToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyDeviceToHost, hipMemcpyDefault);

  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  LinearAllocGuard3D<int> device_alloc(extent);

  const size_t host_pitch = GENERATE_REF(device_alloc.width(), device_alloc.width() + 64);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   host_pitch * device_alloc.height() * device_alloc.depth());

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(device_alloc.width_logical() / threads_per_block.x + 1,
                    device_alloc.height() / threads_per_block.y + 1, device_alloc.depth());
  Iota<<<blocks, threads_per_block>>>(device_alloc.ptr(), device_alloc.pitch(),
                                      device_alloc.width_logical(), device_alloc.height(),
                                      device_alloc.depth());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(memcpy_func(
      make_hipPitchedPtr(host_alloc.ptr(), host_pitch, device_alloc.width(), device_alloc.height()),
      make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), device_alloc.extent(),
      kind, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemoryVerify(host_alloc.ptr(), host_pitch, device_alloc.width_logical(),
                      device_alloc.height(), device_alloc.depth(), f);
}

template <bool should_synchronize, bool enable_peer_access, typename F>
void Memcpy3DDeviceToDeviceShell(F memcpy_func, hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyDeviceToDevice, hipMemcpyDefault);

  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(range(0, device_count));
  const auto dst_device = GENERATE_COPY(range(0, device_count));

  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  HIP_CHECK(hipSetDevice(src_device));
  if (device_count > 0 && kernel_stream != nullptr && kernel_stream != hipStreamPerThread) {
    HIP_CHECK(hipStreamCreate(&kernel_stream));
  }
  if constexpr (enable_peer_access) {
    if (src_device == dst_device) {
      if (device_count > 0 && kernel_stream != nullptr && kernel_stream != hipStreamPerThread) {
        HIP_CHECK(hipStreamDestroy(kernel_stream));
      }
      return;
    }
    int can_access_peer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
    if (!can_access_peer) {
      std::string msg = "Skipped as peer access cannot be enabled between devices " +
          std::to_string(src_device) + " " + std::to_string(dst_device);
      HipTest::HIP_SKIP_TEST(msg.c_str());
      if (device_count > 0 && kernel_stream != nullptr && kernel_stream != hipStreamPerThread) {
        HIP_CHECK(hipStreamDestroy(kernel_stream));
      }
      return;
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));
  }

  LinearAllocGuard3D<int> src_alloc(extent);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard3D<int> dst_alloc(extent);
  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   dst_alloc.width() * dst_alloc.height() * dst_alloc.depth());

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(dst_alloc.width_logical() / threads_per_block.x + 1,
                    dst_alloc.height() / threads_per_block.y + 1, dst_alloc.depth());
  // Using dst_alloc width and height to set only the elements that will be copied over to
  // dst_alloc
  Iota<<<blocks, threads_per_block>>>(src_alloc.ptr(), src_alloc.pitch(),
                                      dst_alloc.width_logical(),
                                      dst_alloc.height(), dst_alloc.depth());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(memcpy_func(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                        make_hipPos(0, 0, 0), dst_alloc.extent(), kind, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }
  if (device_count > 0 && kernel_stream != nullptr && kernel_stream != hipStreamPerThread) {
    HIP_CHECK(hipStreamDestroy(kernel_stream));
  }
  HIP_CHECK(Memcpy3DWrapper(make_hipPitchedPtr(host_alloc.ptr(), dst_alloc.width(),
                                               dst_alloc.width(), dst_alloc.height()),
                            make_hipPos(0, 0, 0), dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                            dst_alloc.extent(), hipMemcpyDeviceToHost));

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemoryVerify(host_alloc.ptr(), dst_alloc.width(), dst_alloc.width_logical(),
                      dst_alloc.height(), dst_alloc.depth(), f);
}

template <bool should_synchronize, typename F>
void Memcpy3DHostToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDefault);

  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  LinearAllocGuard3D<int> device_alloc(extent);

  const size_t host_pitch = GENERATE_REF(device_alloc.pitch(), 2 * device_alloc.pitch());

  LinearAllocGuard<int> src_host_alloc(LinearAllocs::hipHostMalloc,
                                       host_pitch * device_alloc.height() * device_alloc.depth());
  LinearAllocGuard<int> dst_host_alloc(
      LinearAllocs::hipHostMalloc,
      device_alloc.width() * device_alloc.height() * device_alloc.depth());

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemorySet(src_host_alloc.ptr(), host_pitch, device_alloc.width_logical(),
                   device_alloc.height(), device_alloc.depth(), f);

  std::fill_n(dst_host_alloc.ptr(),
              device_alloc.width_logical() * device_alloc.height() * device_alloc.depth(), 0);

  HIP_CHECK(memcpy_func(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                        make_hipPitchedPtr(src_host_alloc.ptr(), host_pitch, device_alloc.width(),
                                           device_alloc.height()),
                        make_hipPos(0, 0, 0), device_alloc.extent(), kind, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(Memcpy3DWrapper(make_hipPitchedPtr(dst_host_alloc.ptr(), device_alloc.width(),
                                               device_alloc.width(), device_alloc.height()),
                            make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                            device_alloc.extent(), hipMemcpyDeviceToHost));

  PitchedMemoryVerify(dst_host_alloc.ptr(), device_alloc.width(), device_alloc.width_logical(),
                      device_alloc.height(), device_alloc.depth(), f);
}

template <bool should_synchronize, typename F>
void Memcpy3DHostToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyHostToHost, hipMemcpyDefault);

  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  const size_t padding = GENERATE_COPY(0, 64);
  const size_t src_pitch = extent.width + padding;

  LinearAllocGuard<int> src_host(LinearAllocs::hipHostMalloc,
                                 src_pitch * extent.height * extent.depth);
  LinearAllocGuard<int> dst_host(LinearAllocs::hipHostMalloc,
                                 extent.width * extent.height * extent.depth);

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemorySet(src_host.ptr(), src_pitch, extent.width / sizeof(int), extent.height,
                   extent.depth, f);

  HIP_CHECK(
      memcpy_func(make_hipPitchedPtr(dst_host.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_host.ptr(), src_pitch, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, kind, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  PitchedMemoryVerify(dst_host.ptr(), extent.width, extent.width / sizeof(int), extent.height,
                      extent.depth, f);
}

template <bool should_synchronize, typename F>
void Memcpy3DArrayHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  constexpr hipExtent extent{127, 128, 8};

  LinearAllocGuard<int> src_host(LinearAllocs::hipHostMalloc,
                                 extent.width * sizeof(int) * extent.height * extent.depth);
  LinearAllocGuard<int> dst_host(LinearAllocs::hipHostMalloc,
                                 extent.width * sizeof(int) * extent.height * extent.depth);

  ArrayAllocGuard<int> src_array(extent);
  ArrayAllocGuard<int> dst_array(extent);

  const auto f = [extent](size_t x, size_t y, size_t z) {
    return z * extent.width * extent.height + y * extent.width + x;
  };
  PitchedMemorySet(src_host.ptr(), extent.width * sizeof(int), extent.width, extent.height,
                   extent.depth, f);

  // Host -> Array
  HIP_CHECK(memcpy_func(src_array.ptr(), make_hipPos(0, 0, 0),
                        make_hipPitchedPtr(src_host.ptr(), extent.width * sizeof(int),
                                           extent.width * sizeof(int), extent.height),
                        make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Array
  HIP_CHECK(memcpy_func(dst_array.ptr(), make_hipPos(0, 0, 0), src_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Host
  HIP_CHECK(memcpy_func(make_hipPitchedPtr(dst_host.ptr(), extent.width * sizeof(int),
                                           extent.width * sizeof(int), extent.height),
                        make_hipPos(0, 0, 0), dst_array.ptr(), make_hipPos(0, 0, 0), extent,
                        hipMemcpyDeviceToHost, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  PitchedMemoryVerify(dst_host.ptr(), extent.width * sizeof(int), extent.width, extent.height,
                      extent.depth, f);
}

template <bool should_synchronize, typename F>
void Memcpy3DArrayDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  constexpr hipExtent extent{127, 128, 8};

  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   extent.width * sizeof(int) * extent.height * extent.depth);

  ArrayAllocGuard<int> src_array(extent);
  ArrayAllocGuard<int> dst_array(extent);

  LinearAllocGuard3D<int> src_device(extent.width, extent.height, extent.depth);
  LinearAllocGuard3D<int> dst_device(extent.width, extent.height, extent.depth);

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(src_device.width_logical() / threads_per_block.x + 1,
                    src_device.height() / threads_per_block.y + 1, src_device.depth());
  Iota<<<blocks, threads_per_block>>>(src_device.ptr(), src_device.pitch(),
                                      src_device.width_logical(), src_device.height(),
                                      src_device.depth());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Device -> Array
  HIP_CHECK(memcpy_func(src_array.ptr(), make_hipPos(0, 0, 0), src_device.pitched_ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Array
  HIP_CHECK(memcpy_func(dst_array.ptr(), make_hipPos(0, 0, 0), src_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Device
  HIP_CHECK(memcpy_func(dst_device.pitched_ptr(), make_hipPos(0, 0, 0), dst_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Device -> Host
  HIP_CHECK(memcpy_func(make_hipPitchedPtr(host_alloc.ptr(), extent.width * sizeof(int),
                                           extent.width * sizeof(int), extent.height),
                        make_hipPos(0, 0, 0), dst_device.pitched_ptr(), make_hipPos(0, 0, 0),
                        dst_device.extent(), hipMemcpyDeviceToHost, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  const auto f = [extent](size_t x, size_t y, size_t z) {
    return z * extent.width * extent.height + y * extent.width + x;
  };
  PitchedMemoryVerify(host_alloc.ptr(), extent.width * sizeof(int), extent.width, extent.height,
                      extent.depth, f);
}

template <typename F>
void Memcpy3DHtoDSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  LinearAllocGuard3D<int> device_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  LinearAllocGuard<int> host_alloc(
    LA::hipHostMalloc, device_alloc.width() * device_alloc.height() * device_alloc.depth());
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func, device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                make_hipPitchedPtr(host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                                   device_alloc.height()),
                make_hipPos(0, 0, 0), device_alloc.extent(), hipMemcpyHostToDevice, kernel_stream),
      should_sync, kernel_stream);
}

template <typename F>
void Memcpy3DDtoHPageableSyncBehavior(F memcpy_func, const bool should_sync,
                                      const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard3D<int> device_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  LinearAllocGuard<int> host_alloc(
      LinearAllocs::malloc, device_alloc.width() * device_alloc.height() * device_alloc.depth());
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func,
                make_hipPitchedPtr(host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                                   device_alloc.height()),
                make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                device_alloc.extent(), hipMemcpyDeviceToHost, kernel_stream),
                should_sync, kernel_stream);
}

template <typename F>
void Memcpy3DDtoHPinnedSyncBehavior(F memcpy_func, const bool should_sync,
                                    const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard3D<int> device_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  LinearAllocGuard<int> host_alloc(
      LinearAllocs::hipHostMalloc,
      device_alloc.width() * device_alloc.height() * device_alloc.depth());
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func,
                make_hipPitchedPtr(host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                                   device_alloc.height()),
                make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                device_alloc.extent(), hipMemcpyDeviceToHost, kernel_stream),
      should_sync, kernel_stream);
}

template <typename F>
void Memcpy3DDtoDSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard3D<int> src_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  LinearAllocGuard3D<int> dst_alloc(make_hipExtent(32 * sizeof(int), 32, 8));
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func, dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                make_hipPos(0, 0, 0), dst_alloc.extent(), hipMemcpyDeviceToDevice, kernel_stream),
      should_sync, kernel_stream);
}

template <typename F>
void Memcpy3DHtoHSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto src_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);

  LinearAllocGuard<int> src_alloc(src_alloc_type, 32 * sizeof(int) * 32 * 8);
  LinearAllocGuard<int> dst_alloc(dst_alloc_type, 32 * sizeof(int) * 32 * 8);
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func,
                make_hipPitchedPtr(dst_alloc.ptr(), 32 * sizeof(int), 32 * sizeof(int), 32),
                make_hipPos(0, 0, 0),
                make_hipPitchedPtr(src_alloc.ptr(), 32 * sizeof(int), 32 * sizeof(int), 32),
                make_hipPos(0, 0, 0), make_hipExtent(32 * sizeof(int), 32, 8), hipMemcpyHostToHost,
                kernel_stream),
      should_sync, kernel_stream);
}

template <bool should_synchronize, typename F>
void Memcpy3DZeroWidthHeightDepth(F memcpy_func, const hipStream_t stream = nullptr) {
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
    HIP_CHECK(Memcpy3DWrapper(make_hipPitchedPtr(host_alloc.ptr(), dst_alloc.width(),
                                                 dst_alloc.width(), dst_alloc.height()),
                              make_hipPos(0, 0, 0), dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                              dst_alloc.extent(), hipMemcpyDeviceToHost));
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
    HIP_CHECK(Memcpy3DWrapper(make_hipPitchedPtr(dst_host_alloc.ptr(), device_alloc.width(),
                                                 device_alloc.width(), device_alloc.height()),
                              make_hipPos(0, 0, 0), device_alloc.pitched_ptr(),
                              make_hipPos(0, 0, 0), device_alloc.extent(), hipMemcpyDeviceToHost));
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

constexpr auto MemTypeHost() {
#if HT_AMD
  return hipMemoryTypeHost;
#else
  return CU_MEMORYTYPE_HOST;
#endif
}

constexpr auto MemTypeDevice() {
#if HT_AMD
  return hipMemoryTypeDevice;
#else
  return CU_MEMORYTYPE_DEVICE;
#endif
}

constexpr auto MemTypeArray() {
#if HT_AMD
  return hipMemoryTypeArray;
#else
  return CU_MEMORYTYPE_ARRAY;
#endif
}

constexpr auto MemTypeUnified() {
#if HT_AMD
  return hipMemoryTypeUnified;
#else
  return CU_MEMORYTYPE_UNIFIED;
#endif
}

using DrvPtrVariant = std::variant<hipPitchedPtr, hipArray_t>;

static inline HIP_MEMCPY3D GetDrvMemcpy3DParms(DrvPtrVariant dst_ptr, hipPos dst_pos,
                                               DrvPtrVariant src_ptr, hipPos src_pos,
                                               hipExtent extent, hipMemcpyKind kind) {
  HIP_MEMCPY3D parms = {0};

  if (std::holds_alternative<hipArray_t>(dst_ptr)) {
    parms.dstMemoryType = hipMemoryTypeArray;
    parms.dstArray = std::get<hipArray_t>(dst_ptr);
  } else {
    auto ptr = std::get<hipPitchedPtr>(dst_ptr);
    parms.dstPitch = ptr.pitch;
    switch (kind) {
      case hipMemcpyDeviceToHost:
      case hipMemcpyHostToHost:
        parms.dstMemoryType = hipMemoryTypeHost;
        parms.dstHost = ptr.ptr;
        break;
      case hipMemcpyDeviceToDevice:
      case hipMemcpyHostToDevice:
        parms.dstMemoryType = hipMemoryTypeDevice;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(ptr.ptr);
        break;
      case hipMemcpyDefault:
        parms.dstMemoryType = hipMemoryTypeUnified;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(ptr.ptr);
        break;
      default:
        assert(false);
    }
  }

  if (std::holds_alternative<hipArray_t>(src_ptr)) {
    parms.srcMemoryType = hipMemoryTypeArray;
    parms.srcArray = std::get<hipArray_t>(src_ptr);
  } else {
    auto ptr = std::get<hipPitchedPtr>(src_ptr);
    parms.srcPitch = ptr.pitch;
    switch (kind) {
      case hipMemcpyDeviceToHost:
      case hipMemcpyDeviceToDevice:
        parms.srcMemoryType = hipMemoryTypeDevice;
        parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(ptr.ptr);
        break;
      case hipMemcpyHostToDevice:
      case hipMemcpyHostToHost:
        parms.srcMemoryType = hipMemoryTypeHost;
        parms.srcHost = ptr.ptr;
        break;
      case hipMemcpyDefault:
        parms.srcMemoryType = hipMemoryTypeUnified;
        parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(ptr.ptr);
        break;
      default:
        assert(false);
    }
  }

  parms.WidthInBytes = extent.width;
  parms.Height = extent.height;
  parms.Depth = extent.depth;
  parms.srcXInBytes = src_pos.x;
  parms.srcY = src_pos.y;
  parms.srcZ = src_pos.z;
  parms.dstXInBytes = dst_pos.x;
  parms.dstY = dst_pos.y;
  parms.dstZ = dst_pos.z;

  return parms;
}

static inline bool operator==(const HIP_MEMCPY3D& lhs, const HIP_MEMCPY3D& rhs) {
  bool pos_eq = lhs.dstXInBytes == rhs.dstXInBytes && lhs.dstY == rhs.dstY &&
      lhs.dstZ == rhs.dstZ && lhs.srcXInBytes == rhs.srcXInBytes && lhs.srcY == rhs.srcY &&
      lhs.srcZ == rhs.srcZ;
  bool extent_eq =
      lhs.WidthInBytes == rhs.WidthInBytes && lhs.Height == rhs.Height && lhs.Depth == rhs.Depth;
  bool mem_eq = true;
  if (lhs.dstArray) {
    mem_eq = lhs.dstArray == rhs.dstArray && lhs.dstMemoryType == rhs.dstMemoryType;
  } else {
    mem_eq = lhs.dstPitch == rhs.dstPitch && lhs.dstMemoryType == rhs.dstMemoryType;
  }
  if (lhs.srcArray) {
    mem_eq = lhs.srcArray == rhs.srcArray && lhs.srcMemoryType == rhs.srcMemoryType;
  } else {
    mem_eq = lhs.srcPitch == rhs.srcPitch && lhs.srcMemoryType == rhs.srcMemoryType;
  }
  if (lhs.dstDevice) {
    mem_eq = mem_eq && (lhs.dstDevice == rhs.dstDevice);
  }
  if (lhs.dstHost) {
    mem_eq = mem_eq && (lhs.dstDevice == rhs.dstDevice);
  }
  if (lhs.srcDevice) {
    mem_eq = mem_eq && (lhs.srcDevice == rhs.srcDevice);
  }
  if (lhs.srcHost) {
    mem_eq = mem_eq && (lhs.srcHost == rhs.srcHost);
  }

  return pos_eq && extent_eq && mem_eq;
}

template <bool set_params = false>
hipError_t DrvMemcpy3DGraphWrapper(DrvPtrVariant dst_ptr, hipPos dst_pos, DrvPtrVariant src_ptr,
                                   hipPos src_pos, hipExtent extent, hipMemcpyKind kind,
                                   hipCtx_t context, hipStream_t stream = nullptr) {
  auto parms = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);

  hipGraph_t g = nullptr;
  HIP_CHECK(hipGraphCreate(&g, 0));
  hipGraphNode_t node = nullptr;
  if constexpr (set_params) {
    auto reversed_parms = GetDrvMemcpy3DParms(src_ptr, src_pos, dst_ptr, dst_pos, extent,
                                              ReverseMemcpyDirection(kind));
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, g, nullptr, 0, &reversed_parms, context));
    HIP_CHECK(hipDrvGraphMemcpyNodeSetParams(node, &parms));
  } else {
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, g, nullptr, 0, &parms, context));
  }

  HIP_MEMCPY3D retrieved_params = {0};
  HIP_CHECK(hipDrvGraphMemcpyNodeGetParams(node, &retrieved_params));
  REQUIRE(parms == retrieved_params);

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(g));

  return hipSuccess;
}

template <bool async = false>
hipError_t DrvMemcpy3DWrapper(DrvPtrVariant dst_ptr, hipPos dst_pos, DrvPtrVariant src_ptr,
                              hipPos src_pos, hipExtent extent, hipMemcpyKind kind,
                              hipStream_t stream = nullptr) {
  auto parms = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);

  if constexpr (async) {
    return hipDrvMemcpy3DAsync(&parms, stream);
  } else {
    return hipDrvMemcpy3D(&parms);
  }
}

template <bool should_synchronize, typename F>
void DrvMemcpy3DArrayHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  LinearAllocGuard<int> src_host(LinearAllocs::hipHostMalloc,
                                 extent.width * extent.height * extent.depth);
  LinearAllocGuard<int> dst_host(LinearAllocs::hipHostMalloc,
                                 extent.width * extent.height * extent.depth);

  DrvArrayAllocGuard<int> src_array(extent);
  DrvArrayAllocGuard<int> dst_array(extent);

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemorySet(src_host.ptr(), extent.width, extent.width / sizeof(int), extent.height,
                   extent.depth, f);

  // Host -> Array
  HIP_CHECK(
      memcpy_func(src_array.ptr(), make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_host.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Array
  HIP_CHECK(memcpy_func(dst_array.ptr(), make_hipPos(0, 0, 0), src_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Host
  HIP_CHECK(
      memcpy_func(make_hipPitchedPtr(dst_host.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), dst_array.ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  PitchedMemoryVerify(dst_host.ptr(), extent.width, extent.width / sizeof(int), extent.height,
                      extent.depth, f);
}

template <bool should_synchronize, typename F>
void DrvMemcpy3DArrayDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  constexpr hipExtent extent{127 * sizeof(int), 128, 8};

  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc,
                                   extent.width * extent.height * extent.depth);

  DrvArrayAllocGuard<int> src_array(extent);
  DrvArrayAllocGuard<int> dst_array(extent);

  LinearAllocGuard3D<int> src_device(extent);
  LinearAllocGuard3D<int> dst_device(extent);

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(src_device.width_logical() / threads_per_block.x + 1,
                    src_device.height() / threads_per_block.y + 1, src_device.depth());
  Iota<<<blocks, threads_per_block>>>(src_device.ptr(), src_device.pitch(),
                                      src_device.width_logical(), src_device.height(),
                                      src_device.depth());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Device -> Array
  HIP_CHECK(memcpy_func(src_array.ptr(), make_hipPos(0, 0, 0), src_device.pitched_ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Array
  HIP_CHECK(memcpy_func(dst_array.ptr(), make_hipPos(0, 0, 0), src_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  // Array -> Device
  HIP_CHECK(memcpy_func(dst_device.pitched_ptr(), make_hipPos(0, 0, 0), dst_array.ptr(),
                        make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(
      memcpy_func(make_hipPitchedPtr(host_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), dst_device.pitched_ptr(), make_hipPos(0, 0, 0),
                  dst_device.extent(), hipMemcpyDeviceToHost, kernel_stream));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  const auto f = [extent](size_t x, size_t y, size_t z) {
    constexpr auto width_logical = extent.width / sizeof(int);
    return z * width_logical * extent.height + y * width_logical + x;
  };
  PitchedMemoryVerify(host_alloc.ptr(), extent.width, extent.width / sizeof(int), extent.height,
                      extent.depth, f);
}
