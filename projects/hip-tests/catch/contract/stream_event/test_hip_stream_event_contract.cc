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
#include <contract_cleanup.hh>

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
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  REQUIRE(stream != nullptr);
}

HIP_TEST_CASE(Contract_Stream_SynchronizeEmptyStream_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_Stream_QueryEmptyStream_ReturnsSuccess) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipStreamQuery(stream));
}

HIP_TEST_CASE(Contract_Event_CreateDestroy_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([event] { (void)hipEventDestroy(event); });

  REQUIRE(event != nullptr);
}

HIP_TEST_CASE(Contract_Event_RecordThenSynchronize_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([event] { (void)hipEventDestroy(event); });

  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipEventQuery(event));
}

HIP_TEST_CASE(Contract_Event_QueryBeforeCompletion_ReturnsNotReadyOrSuccess) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([event] { (void)hipEventDestroy(event); });
  HIP_CHECK(hipEventRecord(event, stream));

  const hipError_t query_result = hipEventQuery(event);
  REQUIRE((query_result == hipSuccess || query_result == hipErrorNotReady));

  HIP_CHECK(hipEventSynchronize(event));
}

HIP_TEST_CASE(Contract_Stream_WaitEvent_OrdersDependentWork) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x42);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipStream_t producer_stream = nullptr;
  hipStream_t consumer_stream = nullptr;
  hipEvent_t copy_ready = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&producer_stream));
  cleanup.Add([producer_stream] { (void)hipStreamDestroy(producer_stream); });
  HIP_CHECK(hipStreamCreate(&consumer_stream));
  cleanup.Add([consumer_stream] { (void)hipStreamDestroy(consumer_stream); });
  HIP_CHECK(hipEventCreate(&copy_ready));
  cleanup.Add([copy_ready] { (void)hipEventDestroy(copy_ready); });

  HIP_CHECK(hipMemcpyAsync(device_ptr, src.data(), src.size(), hipMemcpyHostToDevice,
                           producer_stream));
  HIP_CHECK(hipEventRecord(copy_ready, producer_stream));
  HIP_CHECK(hipStreamWaitEvent(consumer_stream, copy_ready, 0));
  HIP_CHECK(hipMemcpyAsync(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost,
                           consumer_stream));
  HIP_CHECK(hipStreamSynchronize(consumer_stream));

  REQUIRE(dst == src);
}
