/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

constexpr int kSparseWriteStride = 4;
constexpr int kSparseWriteSkipLane = 1;

// Deterministic channel-dependent patterns distinguish adjacent layers, pixels,
// and the post-write generation while keeping floating-point values exact.
template <typename T> __host__ __device__ T getLayeredValue(unsigned index) {
  const unsigned value = 0x73100000u ^ (index * 37u);
  if constexpr (std::is_same_v<T, unsigned int>) {
    return value;
  } else if constexpr (std::is_same_v<T, uint2>) {
    return {value, value ^ 0x39ab45cdu};
  } else if constexpr (std::is_arithmetic_v<T>) {
    return static_cast<T>((index * 37u + 11u) % 127u);
  } else {
    static_assert(std::is_same_v<T, float4>);
    return {static_cast<float>(index % 127u), static_cast<float>(index % 113u),
            static_cast<float>(index % 97u), static_cast<float>(index % 89u)};
  }
}

template <typename T, bool twoDimensional>
__global__ void surfLayeredKernelR(hipSurfaceObject_t surface, T* output, int width, int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  const int layer = blockIdx.z;
  if (x < width && y < height) {
    const size_t index = (layer * height + y) * width + x;
    if constexpr (twoDimensional) {
      surf2DLayeredread<T>(output + index, surface, x * sizeof(T), y, layer);
    } else {
      surf1DLayeredread(output + index, surface, x * sizeof(T), layer);
    }
  }
#endif
}

template <typename T, bool twoDimensional>
__global__ void surfLayeredKernelW(hipSurfaceObject_t surface, int width, int height, int layer) {
#if !__HIP_NO_IMAGE_SUPPORT
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (x < width && y < height && x % kSparseWriteStride != kSparseWriteSkipLane) {
    const T value = getLayeredValue<T>(0x8000u + (layer * height + y) * width + x);
    if constexpr (twoDimensional) {
      surf2DLayeredwrite<T>(value, surface, x * sizeof(T), y, layer);
    } else {
      surf1DLayeredwrite(value, surface, x * sizeof(T), layer);
    }
  }
#endif
}

template <typename T, bool twoDimensional> void checkLayeredSurface() {
  const int width = GENERATE(37, 67);
  constexpr int height = twoDimensional ? 17 : 1;
  constexpr int layers = 4;
  const int selectedLayer = GENERATE(1, 3);
  const size_t count = width * height * layers;
  const auto extent = make_hipExtent(width, twoDimensional ? height : 0, layers);
  ArrayAllocGuard<T> array(extent, hipArrayLayered | hipArraySurfaceLoadStore);
  LinearAllocGuard<T> device(LinearAllocs::hipMalloc, count * sizeof(T));
  std::vector<T> input(count);
  std::vector<T> output(count);
  for (size_t index = 0; index < count; ++index) {
    input[index] = getLayeredValue<T>(index);
  }

  hipMemcpy3DParms copy{};
  copy.srcPtr = make_hipPitchedPtr(input.data(), width * sizeof(T), width, height);
  copy.dstArray = array.ptr();
  copy.extent = make_hipExtent(width, height, layers);
  copy.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&copy));

  hipResourceDesc resource{};
  resource.resType = hipResourceTypeArray;
  resource.res.array.array = array.ptr();
  hipSurfaceObject_t surface = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surface, &resource));

  HIP_CHECK(hipMemset(device.ptr(), 0xa5, count * sizeof(T)));
  const dim3 grid((width + 63) / 64, height, layers);
  surfLayeredKernelR<T, twoDimensional><<<grid, 64>>>(surface, device.ptr(), width, height);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.data(), device.ptr(), count * sizeof(T), hipMemcpyDeviceToHost));
  const auto checkOutput = [&] {
    const auto mismatch = std::mismatch(output.begin(), output.end(), input.begin(),
                                        [](const T& a, const T& b) { return isEqual(a, b); });
    const size_t index = std::distance(output.begin(), mismatch.first);
    INFO("layer=" << index / (width * height) << ", y=" << (index / width) % height
                  << ", x=" << index % width);
    INFO("actual=" << (index < count ? getString(*mismatch.first) : "<end>")
                   << ", expected=" << (index < count ? getString(*mismatch.second) : "<end>"));
    REQUIRE(mismatch.first == output.end());
  };
  checkOutput();

  // A sparse write to a nonzero layer must preserve other layers and skipped pixels.
  surfLayeredKernelW<T, twoDimensional>
      <<<dim3(grid.x, height), 64>>>(surface, width, height, selectedLayer);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipDestroySurfaceObject(surface));
  copy = {};
  copy.srcArray = array.ptr();
  copy.dstPtr = make_hipPitchedPtr(output.data(), width * sizeof(T), width, height);
  copy.extent = make_hipExtent(width, height, layers);
  copy.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&copy));
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const size_t index = (selectedLayer * height + y) * width + x;
      if (x % kSparseWriteStride != kSparseWriteSkipLane) {
        input[index] = getLayeredValue<T>(0x8000u + index);
      }
    }
  }
  checkOutput();
}
