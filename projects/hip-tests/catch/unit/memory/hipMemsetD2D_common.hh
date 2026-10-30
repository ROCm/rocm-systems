/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <hip_test_common.hh>
#include <resource_guards.hh>

enum class TypedFillMode { Direct, Captured, Graph };

template <typename T, bool asynchronous> hipError_t fillTyped2D(void* destination, size_t pitch,
                                                                T value, size_t width,
                                                                size_t height, hipStream_t stream) {
  static_assert(std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>);
  const auto dst = reinterpret_cast<hipDeviceptr_t>(destination);
  if constexpr (std::is_same_v<T, uint16_t>) {
    if constexpr (asynchronous) {
      return hipMemsetD2D16Async(dst, pitch, value, width, height, stream);
    } else {
      return hipMemsetD2D16(dst, pitch, value, width, height);
    }
  } else {
    if constexpr (asynchronous) {
      return hipMemsetD2D32Async(dst, pitch, value, width, height, stream);
    } else {
      return hipMemsetD2D32(dst, pitch, value, width, height);
    }
  }
}

template <typename T, bool asynchronous> void typed2DPadding(TypedFillMode mode) {
  const size_t width = GENERATE(1, 3, 17, 37, 65);
  INFO("width=" << width << " mode=" << static_cast<int>(mode));
  constexpr size_t pitch = 512;
  constexpr size_t height = 5;
  constexpr size_t bytes = pitch * (height + 2);
  constexpr T value = static_cast<T>(0x93ab7d31);
  LinearAllocGuard<uint8_t> device(LinearAllocs::hipMalloc, bytes);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  std::vector<uint8_t> poison(bytes, 0xa5);
  std::vector<uint8_t> expected(poison);
  std::vector<uint8_t> output(bytes);
  HIP_CHECK(hipMemcpy(device.ptr(), poison.data(), bytes, hipMemcpyHostToDevice));
  auto* destination = device.ptr() + pitch + sizeof(T);
  for (size_t row = 1; row <= height; ++row) {
    for (size_t byte = 0; byte < width * sizeof(T); ++byte) {
      expected[row * pitch + sizeof(T) + byte] =
          static_cast<uint8_t>(value >> (8 * (byte % sizeof(T))));
    }
  }

  if (mode == TypedFillMode::Direct) {
    HIP_CHECK(
        (fillTyped2D<T, asynchronous>(destination, pitch, value, width, height, stream.stream())));
    HIP_CHECK(hipDeviceSynchronize());
  } else {
    hipGraph_t graph = nullptr;
    hipGraphNode_t node = nullptr;
    if (mode == TypedFillMode::Captured) {
      HIP_CHECK(hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal));
      HIP_CHECK((fillTyped2D<T, true>(destination, pitch, value, width, height, stream.stream())));
      HIP_CHECK(hipStreamEndCapture(stream.stream(), &graph));
      size_t nodeCount = 1;
      HIP_CHECK(hipGraphGetNodes(graph, &node, &nodeCount));
      REQUIRE(nodeCount == 1);
    } else {
      HIP_CHECK(hipGraphCreate(&graph, 0));
      hipMemsetParams params{};
      params.dst = destination;
      params.pitch = pitch;
      params.value = value;
      params.elementSize = sizeof(T);
      params.width = width;
      params.height = height;
      HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    }
    hipMemsetParams params{};
    HIP_CHECK(hipGraphMemsetNodeGetParams(node, &params));
    hipGraphExec_t executable = nullptr;
    HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(executable, stream.stream()));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipGraphExecDestroy(executable));
    HIP_CHECK(hipGraphDestroy(graph));
    REQUIRE(params.elementSize == sizeof(T));
    REQUIRE(params.width == width);
    REQUIRE(params.height == height);
    REQUIRE(params.pitch == pitch);
    REQUIRE(params.value == value);
  }
  HIP_CHECK(hipMemcpy(output.data(), device.ptr(), bytes, hipMemcpyDeviceToHost));
  REQUIRE(output == expected);
}

template <typename T, bool asynchronous> void typed2DInvalidGeometry() {
  const size_t height = GENERATE(1, 2);
  constexpr size_t pitch = 512;
  constexpr T value = static_cast<T>(0x93ab7d31);
  constexpr size_t maxSize = std::numeric_limits<size_t>::max();
  LinearAllocGuard<uint8_t> device(LinearAllocs::hipMalloc, pitch * 4);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  const auto fill = [&](void* dst, size_t rowPitch, size_t width, size_t rowHeight) {
    return fillTyped2D<T, asynchronous>(dst, rowPitch, value, width, rowHeight, stream.stream());
  };
  HIP_CHECK_ERROR(fill(device.ptr(), pitch, maxSize / sizeof(T) + 1, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(fill(device.ptr(), maxSize - 7, maxSize / sizeof(T) / 2 + 1, 4),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(fill(device.ptr(), pitch, pitch / sizeof(T) + 1, height), hipErrorInvalidValue);
  HIP_CHECK_ERROR(fill(device.ptr() + 1, pitch, 17, height), hipErrorInvalidValue);
  HIP_CHECK_ERROR(fill(device.ptr(), pitch + 1, 17, height), hipErrorInvalidValue);
  // A tiny logical fill must still fit its full pitched span within the allocation.
  HIP_CHECK_ERROR(fill(device.ptr() + pitch * 3, pitch, 1, 2), hipErrorInvalidValue);
  HIP_CHECK(fill(device.ptr(), pitch, 0, 2));
  HIP_CHECK(fill(device.ptr(), pitch, 17, 0));
}

template <typename Fill> void empty2DCapture(Fill fill) {
  const bool invalidate = GENERATE(false, true);
  LinearAllocGuard<uint8_t> device(LinearAllocs::hipMalloc, 512);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  std::vector<uint8_t> poison(512, 0xa5);
  std::vector<uint8_t> output(512);
  HIP_CHECK(hipMemcpy(device.ptr(), poison.data(), poison.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal));
  if (invalidate) {
    HIP_CHECK_ERROR(hipStreamSynchronize(stream.stream()), hipErrorStreamCaptureUnsupported);
  }
  for (const auto extent :
       {make_hipExtent(0, 2, 1), make_hipExtent(17, 0, 1), make_hipExtent(0, 0, 1)}) {
    const auto status = fill(device.ptr(), 64, extent.width, extent.height, stream.stream());
    REQUIRE(status == (invalidate ? hipErrorStreamCaptureInvalidated : hipSuccess));
  }
  hipGraph_t graph = nullptr;
  const auto status = hipStreamEndCapture(stream.stream(), &graph);
  if (invalidate) {
    REQUIRE(status == hipErrorStreamCaptureInvalidated);
    REQUIRE(graph == nullptr);
  } else {
    HIP_CHECK(status);
    size_t count = 0;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &count));
    HIP_CHECK(hipGraphDestroy(graph));
    REQUIRE(count == 0);
  }
  HIP_CHECK(hipMemcpy(output.data(), device.ptr(), output.size(), hipMemcpyDeviceToHost));
  REQUIRE(output == poison);
}
