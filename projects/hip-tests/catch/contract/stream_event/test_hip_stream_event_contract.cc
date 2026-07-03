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

HIP_TEST_CASE(Contract_Stream_CreateDestroy_Succeeds) {
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(stream != nullptr);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_Stream_SynchronizeEmptyStream_Succeeds) {
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_Stream_QueryEmptyStream_ReturnsSuccess) {
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamQuery(stream));

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_Event_CreateDestroy_Succeeds) {
  hipEvent_t event = nullptr;

  HIP_CHECK(hipEventCreate(&event));

  REQUIRE(event != nullptr);

  HIP_CHECK(hipEventDestroy(event));
}

HIP_TEST_CASE(Contract_Event_RecordThenSynchronize_Succeeds) {
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&event));

  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipEventQuery(event));

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_Event_QueryBeforeCompletion_ReturnsNotReadyOrSuccess) {
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventRecord(event, stream));

  const hipError_t query_result = hipEventQuery(event);
  REQUIRE((query_result == hipSuccess || query_result == hipErrorNotReady));

  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_Stream_WaitEvent_OrdersDependentWork) {
  const auto src = MakePattern(0x42);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t producer_stream = nullptr;
  hipStream_t consumer_stream = nullptr;
  hipEvent_t copy_ready = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  HIP_CHECK(hipStreamCreate(&producer_stream));
  HIP_CHECK(hipStreamCreate(&consumer_stream));
  HIP_CHECK(hipEventCreate(&copy_ready));

  HIP_CHECK(hipMemcpyAsync(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice,
                           producer_stream));
  HIP_CHECK(hipEventRecord(copy_ready, producer_stream));
  HIP_CHECK(hipStreamWaitEvent(consumer_stream, copy_ready, 0));
  HIP_CHECK(hipMemcpyAsync(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost,
                           consumer_stream));
  HIP_CHECK(hipStreamSynchronize(consumer_stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipEventDestroy(copy_ready));
  HIP_CHECK(hipStreamDestroy(consumer_stream));
  HIP_CHECK(hipStreamDestroy(producer_stream));
  HIP_CHECK(hipFree(device_ptr));
}
