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

namespace {
constexpr size_t kBufferBytes = 256;

// Skips the test when no device is visible so that the stream property and event
// timing contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_StreamProps_CreateWithDefaultFlags_ReportsDefaultFlags) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  REQUIRE(stream != nullptr);

  unsigned int flags = 0xFFFFFFFFu;
  HIP_CHECK(hipStreamGetFlags(stream, &flags));

  // A stream created with the default flags must not report the non-blocking
  // bit; the runtime may report additional device-specific bits, so the
  // contract is expressed as the absence of hipStreamNonBlocking rather than
  // exact equality with hipStreamDefault.
  REQUIRE((flags & hipStreamNonBlocking) == 0u);
}

HIP_TEST_CASE(Contract_StreamProps_CreateWithNonBlockingFlags_RoundTripsFlags) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  REQUIRE(stream != nullptr);

  unsigned int flags = 0;
  HIP_CHECK(hipStreamGetFlags(stream, &flags));

  // The non-blocking request must round-trip as a set bit; the reported word
  // may carry additional bits, so a mask is used rather than exact equality.
  REQUIRE((flags & hipStreamNonBlocking) == hipStreamNonBlocking);
}

HIP_TEST_CASE(Contract_StreamProps_CreateWithPriority_ClampsWithinRange) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  int least_priority = 0;
  int greatest_priority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, greatest_priority));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  REQUIRE(stream != nullptr);

  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));

  // HIP orders priorities so that greatest (highest) is numerically smaller or
  // equal to least (lowest). The reported priority must be clamped within that
  // inclusive range regardless of any backend-specific coercion.
  REQUIRE(greatest_priority <= priority);
  REQUIRE(priority <= least_priority);
}

HIP_TEST_CASE(Contract_StreamProps_GetDevice_MatchesCurrentDevice) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  hipDevice_t stream_device = hipDevice_t{};
  const hipError_t status = hipStreamGetDevice(stream, &stream_device);
  if (status == hipErrorNotSupported) {
    // Querying a stream's owning device is an optional capability on some
    // backends; an unsupported report is a contract-compliant outcome.
    return;
  }
  HIP_CHECK(status);

  // The stream is created on the current device, so its reported owning device
  // ordinal must match the current device.
  REQUIRE(static_cast<int>(stream_device) == current_device);
}

HIP_TEST_CASE(Contract_StreamProps_GetId_DistinctStreamsDifferAndNullStreamQueryable) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t first_stream = nullptr;
  hipStream_t second_stream = nullptr;
  HIP_CHECK(hipStreamCreate(&first_stream));
  cleanup.Add([&] { (void)hipStreamDestroy(first_stream); });
  HIP_CHECK(hipStreamCreate(&second_stream));
  cleanup.Add([&] { (void)hipStreamDestroy(second_stream); });

  unsigned long long first_id = 0;
  const hipError_t first_status = hipStreamGetId(first_stream, &first_id);
  if (first_status == hipErrorNotSupported) {
    // Stream identity queries are an optional capability; an unsupported report
    // is contract-compliant.
    return;
  }
  HIP_CHECK(first_status);

  unsigned long long second_id = 0;
  HIP_CHECK(hipStreamGetId(second_stream, &second_id));

  // Two independently created streams must report distinct identities.
  REQUIRE(first_id != second_id);

  // The null (default) stream has a well-defined identity that must be queryable
  // and succeed.
  unsigned long long null_stream_id = 0;
  HIP_CHECK(hipStreamGetId(nullptr, &null_stream_id));
}

HIP_TEST_CASE(Contract_StreamProps_EventElapsedTime_NonNegativeForOrderedEvents) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  void* device_ptr = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&start_event));
  cleanup.Add([&] { (void)hipEventDestroy(start_event); });
  HIP_CHECK(hipEventCreate(&stop_event));
  cleanup.Add([&] { (void)hipEventDestroy(stop_event); });
  HIP_CHECK(hipMalloc(&device_ptr, kBufferBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipEventRecord(start_event, stream));
  HIP_CHECK(hipMemsetAsync(device_ptr, 0, kBufferBytes, stream));
  HIP_CHECK(hipEventRecord(stop_event, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  float elapsed_ms = -1.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));

  // Time between an earlier and a later event on the same stream is monotonic
  // and must be non-negative; the exact magnitude is timing-dependent and is
  // therefore not asserted.
  REQUIRE(elapsed_ms >= 0.0f);
}

HIP_TEST_CASE(Contract_StreamProps_EventRecordWithFlags_RecordsAndTimes) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  void* device_ptr = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&start_event));
  cleanup.Add([&] { (void)hipEventDestroy(start_event); });
  HIP_CHECK(hipEventCreate(&stop_event));
  cleanup.Add([&] { (void)hipEventDestroy(stop_event); });
  HIP_CHECK(hipMalloc(&device_ptr, kBufferBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const hipError_t start_status =
      hipEventRecordWithFlags(start_event, stream, hipEventRecordDefault);
  if (start_status == hipErrorNotSupported) {
    // Recording an event with explicit flags is an optional capability on some
    // backends; an unsupported report is a contract-compliant outcome.
    return;
  }
  HIP_CHECK(start_status);

  HIP_CHECK(hipMemsetAsync(device_ptr, 0, kBufferBytes, stream));
  HIP_CHECK(hipEventRecordWithFlags(stop_event, stream, hipEventRecordDefault));
  HIP_CHECK(hipStreamSynchronize(stream));

  float elapsed_ms = -1.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));

  // Events recorded with the default record flags must produce a non-negative
  // elapsed time just like the plain record path.
  REQUIRE(elapsed_ms >= 0.0f);
}
