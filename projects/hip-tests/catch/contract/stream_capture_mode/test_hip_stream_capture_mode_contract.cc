/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// BACKEND-DIFF: hipStreamGetCaptureInfo_v2 exists on AMD, but on the NVIDIA
// backend it only wraps cuStreamGetCaptureInfo_v2, which CUDA provides in
// 11.3-12.x and removed in CUDA 13 (superseded by v3). Gate the tests that call
// it so they compile on AMD and on pre-13 CUDA, and skip on CUDA 13+ where the
// entry point is absent. Parity on CUDA 13+ would mean re-expressing on the v3
// capture-info API.
#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION < 13000)
#define HIP_CONTRACT_HAS_CAPTURE_INFO_V2 1
#else
#define HIP_CONTRACT_HAS_CAPTURE_INFO_V2 0
#endif

namespace {
constexpr size_t kByteCount = 64;

void EndCaptureAndDestroyGraph(hipStream_t stream) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}
}  // namespace

// @asserts: hipThreadExchangeStreamCaptureMode - exchanging the capture mode returns the thread's previous mode
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

// BACKEND-DIFF: the null-mode rejection contract runs only on AMD. On NVIDIA
// hipThreadExchangeStreamCaptureMode forwards to cuThreadExchangeStreamCaptureMode,
// which does not reject a null mode pointer the same way, so the negative check
// is not portable. Parity would require matching null-argument validation.
#if HT_AMD
// @asserts: hipThreadExchangeStreamCaptureMode - rejects a null mode pointer with a non-success error
HIP_TEST_CASE(Contract_StreamCaptureMode_Exchange_NullMode_IsRejected) {
  REQUIRE(hipThreadExchangeStreamCaptureMode(nullptr) != hipSuccess);
}
#endif

#if HIP_CONTRACT_HAS_CAPTURE_INFO_V2
// @asserts: hipStreamGetCaptureInfo_v2 - reports Active status with a nonzero capture id and graph during capture
HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReportsActiveDuringCapture) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status, &capture_id, &graph, nullptr, nullptr));

  REQUIRE(status == hipStreamCaptureStatusActive);
  REQUIRE(capture_id > 0);
  REQUIRE(graph != nullptr);

  EndCaptureAndDestroyGraph(stream);
}

// @asserts: hipStreamGetCaptureInfo_v2 - reports None status for a stream that is not capturing
HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReportsNoneOutsideCapture) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status));

  REQUIRE(status == hipStreamCaptureStatusNone);
}

// @asserts: hipStreamGetCaptureInfo_v2 - exposes the captured memset as the single current dependency node
HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_ReturnsDependencyNode) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  const hipGraphNode_t* dependencies = nullptr;
  size_t dependency_count = 0;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
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
}

// @asserts: hipStreamGetCaptureInfo_v2 - rejects a null capture-status output pointer with a non-success error
HIP_TEST_CASE(Contract_StreamCaptureMode_GetCaptureInfoV2_NullStatus_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  unsigned long long capture_id = 0;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  REQUIRE(hipStreamGetCaptureInfo_v2(stream, nullptr, &capture_id, nullptr, nullptr, nullptr) !=
          hipSuccess);
}
#endif  // HIP_CONTRACT_HAS_CAPTURE_INFO_V2
