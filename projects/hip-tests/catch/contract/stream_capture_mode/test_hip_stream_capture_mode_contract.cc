/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kByteCount = 64;

void EndCaptureAndDestroyGraph(hipStream_t stream) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}
}  // namespace

HIP_TEST_CASE(Contract_StreamCaptureMode_Exchange_RoundTripsPreviousMode) {
  hipStreamCaptureMode mode = hipStreamCaptureModeThreadLocal;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));
  hipStreamCaptureMode previous_mode = mode;

  mode = hipStreamCaptureModeRelaxed;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));
  const hipStreamCaptureMode observed_previous_mode = mode;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&previous_mode));

  REQUIRE(observed_previous_mode == hipStreamCaptureModeThreadLocal);
}

#if HT_AMD
HIP_TEST_CASE(Contract_StreamCaptureMode_Exchange_NullMode_IsRejected) {
  REQUIRE(hipThreadExchangeStreamCaptureMode(nullptr) != hipSuccess);
}
#endif

HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReportsActiveDuringCapture) {
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status, &capture_id, &graph, nullptr, nullptr));

  REQUIRE(status == hipStreamCaptureStatusActive);
  REQUIRE(capture_id > 0);
  REQUIRE(graph != nullptr);

  EndCaptureAndDestroyGraph(stream);
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReportsNoneOutsideCapture) {
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status));

  REQUIRE(status == hipStreamCaptureStatusNone);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReturnsDependencyNode) {
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  const hipGraphNode_t* dependencies = nullptr;
  size_t dependency_count = 0;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(device_ptr, 0x5a, kByteCount, stream));
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status, nullptr, nullptr, &dependencies,
                                       &dependency_count));

  REQUIRE(status == hipStreamCaptureStatusActive);
  REQUIRE(dependency_count == 1);
  REQUIRE(dependencies != nullptr);
  HIP_CHECK(hipGraphNodeGetType(dependencies[0], &node_type));
  REQUIRE(node_type == hipGraphNodeTypeMemset);

  EndCaptureAndDestroyGraph(stream);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_NullStatus_IsRejected) {
  hipStream_t stream = nullptr;
  unsigned long long capture_id = 0;

  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(hipStreamGetCaptureInfo_v2(stream, nullptr, &capture_id, nullptr, nullptr, nullptr) !=
          hipSuccess);
  HIP_CHECK(hipStreamDestroy(stream));
}
