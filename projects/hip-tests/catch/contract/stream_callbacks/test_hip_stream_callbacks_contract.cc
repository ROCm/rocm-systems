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
constexpr int32_t kInitialValue = 0;
constexpr int32_t kExpectedInvocationCount = 1;
constexpr size_t kMemsetBytes = 64;

// State observed by the stream callback. The callback only mutates this struct
// through userData; it performs no HIP API calls.
struct CallbackState {
  int32_t invocation_count = kInitialValue;
  hipError_t observed_status = hipErrorUnknown;
};

// State that proves stream ordering between prior host work and a later
// callback without any HIP calls inside the callbacks.
struct OrderingState {
  bool prior_ran = false;
  bool callback_saw_prior_work = false;
};

// State that proves stream ordering between two host functions.
struct HostFuncOrderingState {
  int32_t sequence = kInitialValue;
  int32_t first_seen_sequence = -1;
  int32_t second_seen_sequence = -1;
};

// Plain C-style stream callback: no HIP APIs, only mutates userData.
void CountingCallback(hipStream_t /*stream*/, hipError_t status, void* userData) {
  auto* state = static_cast<CallbackState*>(userData);
  state->invocation_count += 1;
  state->observed_status = status;
}

// Plain C-style stream callback that marks that prior stream work has run.
void MarkPriorRan(hipStream_t /*stream*/, hipError_t /*status*/, void* userData) {
  auto* state = static_cast<OrderingState*>(userData);
  state->prior_ran = true;
}

// Plain C-style stream callback that records whether prior host work ran first.
void ObserveOrdering(hipStream_t /*stream*/, hipError_t /*status*/, void* userData) {
  auto* state = static_cast<OrderingState*>(userData);
  state->callback_saw_prior_work = state->prior_ran;
}

// Plain C-style host function: increments a shared sequence counter.
void IncrementSequence(void* userData) {
  auto* counter = static_cast<int32_t*>(userData);
  *counter += 1;
}

// Plain C-style host function that records the sequence value it observed first.
void RecordFirst(void* userData) {
  auto* state = static_cast<HostFuncOrderingState*>(userData);
  state->sequence += 1;
  state->first_seen_sequence = state->sequence;
}

// Plain C-style host function that records the sequence value it observed second.
void RecordSecond(void* userData) {
  auto* state = static_cast<HostFuncOrderingState*>(userData);
  state->sequence += 1;
  state->second_seen_sequence = state->sequence;
}
}  // namespace

HIP_TEST_CASE(Contract_StreamCallbacks_AddCallback_InvokesExactlyOnce) {
  hip::contract::ContractCleanup cleanup;
  CallbackState state{};
  void* device_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMalloc(&device_ptr, kMemsetBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipMemsetAsync(device_ptr, 0, kMemsetBytes, stream));
  HIP_CHECK(hipStreamAddCallback(stream, CountingCallback, &state, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(state.invocation_count == kExpectedInvocationCount);
}

HIP_TEST_CASE(Contract_StreamCallbacks_AddCallback_RunsAfterPriorWork) {
  hip::contract::ContractCleanup cleanup;
  OrderingState state{};
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  // Enqueue a prior callback that flips a host-visible flag, then a second
  // callback that records whether it observed that flag already set. Stream
  // ordering guarantees the second callback runs after the first.
  HIP_CHECK(hipStreamAddCallback(stream, MarkPriorRan, &state, 0));
  HIP_CHECK(hipStreamAddCallback(stream, ObserveOrdering, &state, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(state.prior_ran);
  REQUIRE(state.callback_saw_prior_work);
}

HIP_TEST_CASE(Contract_StreamCallbacks_AddCallback_ReceivesSuccessStatus) {
  hip::contract::ContractCleanup cleanup;
  CallbackState state{};
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipStreamAddCallback(stream, CountingCallback, &state, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(state.invocation_count == kExpectedInvocationCount);
  REQUIRE(state.observed_status == hipSuccess);
}

HIP_TEST_CASE(Contract_StreamCallbacks_LaunchHostFunc_InvokesExactlyOnce) {
  hip::contract::ContractCleanup cleanup;
  int32_t counter = kInitialValue;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  const hipError_t launch_result = hipLaunchHostFunc(stream, IncrementSequence, &counter);
  if (launch_result == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipLaunchHostFunc is not supported on this device");
  }
  HIP_CHECK(launch_result);
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(counter == kExpectedInvocationCount);
}

HIP_TEST_CASE(Contract_StreamCallbacks_LaunchHostFunc_OrdersBeforeLaterWork) {
  hip::contract::ContractCleanup cleanup;
  HostFuncOrderingState state{};
  hipStream_t stream = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  // Two host functions enqueued in stream order. Each records the sequence
  // value it observes; stream ordering guarantees the first observes 1 and the
  // second observes 2.
  const hipError_t launch_result = hipLaunchHostFunc(stream, RecordFirst, &state);
  if (launch_result == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipLaunchHostFunc is not supported on this device");
  }
  HIP_CHECK(launch_result);
  HIP_CHECK(hipLaunchHostFunc(stream, RecordSecond, &state));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(state.first_seen_sequence == 1);
  REQUIRE(state.second_seen_sequence == 2);
  REQUIRE(state.first_seen_sequence < state.second_seen_sequence);
}
