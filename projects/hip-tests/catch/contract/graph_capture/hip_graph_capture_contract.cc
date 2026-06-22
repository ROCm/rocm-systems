/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kElementCount = 128;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_GraphCapture_BeginEndEmptyStream_ProducesGraph) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_GraphCapture_CapturedMemcpy_RoundTripsBytes) {
  const auto src = MakePattern(0x27);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_GraphCapture_IsCapturing_ReportsActiveDuringCapture) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamIsCapturing(stream, &status));

  REQUIRE(status == hipStreamCaptureStatusActive);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_GraphCapture_GetCaptureInfo_ReturnsActiveState) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &status, &capture_id));

  REQUIRE(status == hipStreamCaptureStatusActive);
  REQUIRE(capture_id > 0);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
