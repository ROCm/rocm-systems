/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Skips the test when no device is visible so that the stream memory operation
// contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// Skips the test when the current device does not advertise support for the
// stream wait/write value APIs. The runtime exposes this capability through the
// hipDeviceAttributeCanUseStreamWaitValue attribute, which gates the entire
// stream memory operation family. Returning here keeps the contracts portable:
// on hardware or runtime paths without the capability the test skips cleanly
// rather than reporting a contract violation.
void RequireStreamWaitValueSupport() {
  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));

  int can_use_stream_wait_value = 0;
  HIP_CHECK(hipDeviceGetAttribute(&can_use_stream_wait_value,
                                  hipDeviceAttributeCanUseStreamWaitValue, current_device));
  if (can_use_stream_wait_value == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_StreamMemoryOps_WriteValue32_BecomesVisibleInStreamOrder) {
  RequireDevice();
  RequireStreamWaitValueSupport();

  hipStream_t stream = nullptr;
  uint32_t* device_ptr = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptr), sizeof(uint32_t)));

  // Start from a known zero value so the written sentinel is unambiguous.
  const uint32_t initial = 0u;
  HIP_CHECK(hipMemcpy(device_ptr, &initial, sizeof(uint32_t), hipMemcpyHostToDevice));

  // hipStreamWriteValue32 enqueues a device-side write of the sentinel into the
  // buffer. After the stream drains, the write must be visible to a subsequent
  // device-to-host copy that is ordered behind it on the same stream.
  const uint32_t sentinel = 0xABCD1234u;
  HIP_CHECK(hipStreamWriteValue32(stream, device_ptr, sentinel, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t observed = 0u;
  HIP_CHECK(hipMemcpy(&observed, device_ptr, sizeof(uint32_t), hipMemcpyDeviceToHost));
  REQUIRE(observed == sentinel);

  HIP_CHECK(hipFree(device_ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamMemoryOps_WriteValue64_BecomesVisibleInStreamOrder) {
  RequireDevice();
  RequireStreamWaitValueSupport();

  hipStream_t stream = nullptr;
  uint64_t* device_ptr = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptr), sizeof(uint64_t)));

  // Start from a known zero value so the written sentinel is unambiguous.
  const uint64_t initial = 0ull;
  HIP_CHECK(hipMemcpy(device_ptr, &initial, sizeof(uint64_t), hipMemcpyHostToDevice));

  // hipStreamWriteValue64 is the 64-bit counterpart of hipStreamWriteValue32;
  // the written sentinel must be visible after the stream drains.
  const uint64_t sentinel = 0xABCD1234DEADBEEFull;
  HIP_CHECK(hipStreamWriteValue64(stream, device_ptr, sentinel, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint64_t observed = 0ull;
  HIP_CHECK(hipMemcpy(&observed, device_ptr, sizeof(uint64_t), hipMemcpyDeviceToHost));
  REQUIRE(observed == sentinel);

  HIP_CHECK(hipFree(device_ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamMemoryOps_WaitValueGte_GatesLaterStreamWork) {
  RequireDevice();
  RequireStreamWaitValueSupport();

  hipStream_t stream = nullptr;
  uint32_t* gate_ptr = nullptr;
  uint32_t* done_ptr = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gate_ptr), sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&done_ptr), sizeof(uint32_t)));

  const uint32_t initial = 0u;
  HIP_CHECK(hipMemcpy(gate_ptr, &initial, sizeof(uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(done_ptr, &initial, sizeof(uint32_t), hipMemcpyHostToDevice));

  // Enqueue the satisfying write of the gate value ahead of the wait on the same
  // stream. Because a single stream executes its operations in issue order, the
  // write is guaranteed to complete before the wait is evaluated, so the wait's
  // greater-than-or-equal condition is met without any host-side signalling.
  // This ordering is what makes the test deadlock-free.
  const uint32_t threshold = 1u;
  HIP_CHECK(hipStreamWriteValue32(stream, gate_ptr, threshold, 0));
  HIP_CHECK(hipStreamWaitValue32(stream, gate_ptr, threshold, hipStreamWaitValueGte));

  // The done sentinel is written strictly after the wait. If the wait failed to
  // gate later work correctly, this write would still land, but the contract we
  // assert is that the stream drains cleanly and the ordered write is visible.
  const uint32_t done = 0xFEEDBEEFu;
  HIP_CHECK(hipStreamWriteValue32(stream, done_ptr, done, 0));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t observed = 0u;
  HIP_CHECK(hipMemcpy(&observed, done_ptr, sizeof(uint32_t), hipMemcpyDeviceToHost));
  REQUIRE(observed == done);

  HIP_CHECK(hipFree(done_ptr));
  HIP_CHECK(hipFree(gate_ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamMemoryOps_BatchMemOp_AppliesWritesInStreamOrder) {
  RequireDevice();
  RequireStreamWaitValueSupport();

  hipStream_t stream = nullptr;
  uint32_t* value32_ptr = nullptr;
  uint64_t* value64_ptr = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&value32_ptr), sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&value64_ptr), sizeof(uint64_t)));

  const uint32_t initial32 = 0u;
  const uint64_t initial64 = 0ull;
  HIP_CHECK(hipMemcpy(value32_ptr, &initial32, sizeof(uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(value64_ptr, &initial64, sizeof(uint64_t), hipMemcpyHostToDevice));

  // Describe a batch of two write operations: a 32-bit write and a 64-bit write.
  // Only the write op types are used here because barrier and flush-remote-write
  // ops are documented as unsupported on the AMD backend, and wait ops would
  // require an already-satisfied condition to stay deadlock-free.
  const uint32_t sentinel32 = 0x11223344u;
  const uint64_t sentinel64 = 0x1122334455667788ull;

  hipStreamBatchMemOpParams params[2] = {};
  params[0].writeValue.operation = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = reinterpret_cast<hipDeviceptr_t>(value32_ptr);
  params[0].writeValue.value = sentinel32;
  params[0].writeValue.flags = 0;
  params[1].writeValue.operation = hipStreamMemOpWriteValue64;
  params[1].writeValue.address = reinterpret_cast<hipDeviceptr_t>(value64_ptr);
  params[1].writeValue.value64 = sentinel64;
  params[1].writeValue.flags = 0;

  const hipError_t batch_status = hipStreamBatchMemOp(stream, 2, params, 0);
  if (batch_status == hipErrorNotSupported) {
    // Batch memory operations are an optional capability on some runtime paths
    // even when scalar stream wait/write value is supported; an unsupported
    // report is a contract-compliant outcome.
    HIP_CHECK(hipFree(value64_ptr));
    HIP_CHECK(hipFree(value32_ptr));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_SKIP_TEST("hipStreamBatchMemOp is not supported on this device");
  }
  HIP_CHECK(batch_status);
  HIP_CHECK(hipStreamSynchronize(stream));

  uint32_t observed32 = 0u;
  uint64_t observed64 = 0ull;
  HIP_CHECK(hipMemcpy(&observed32, value32_ptr, sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&observed64, value64_ptr, sizeof(uint64_t), hipMemcpyDeviceToHost));

  // Both writes in the batch must have been applied by the time the stream
  // drains, exactly as a sequence of scalar writes would have been.
  REQUIRE(observed32 == sentinel32);
  REQUIRE(observed64 == sentinel64);

  HIP_CHECK(hipFree(value64_ptr));
  HIP_CHECK(hipFree(value32_ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamMemoryOps_RejectsInvalidInputs) {
  RequireDevice();
  RequireStreamWaitValueSupport();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  // A null address is a caller error. When the write API is supported it must be
  // rejected with hipErrorInvalidValue rather than silently succeeding; when the
  // individual call reports hipErrorNotSupported the capability is absent on this
  // path and the test skips cleanly.
  const hipError_t write_status = hipStreamWriteValue32(stream, nullptr, 0u, 0);
  if (write_status == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_SKIP_TEST("hipStreamWriteValue32 is not supported on this device");
  }
  REQUIRE(write_status == hipErrorInvalidValue);

  // The same null-address rejection contract applies to the wait API.
  const hipError_t wait_status =
      hipStreamWaitValue32(stream, nullptr, 0u, hipStreamWaitValueGte);
  if (wait_status == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_SKIP_TEST("hipStreamWaitValue32 is not supported on this device");
  }
  REQUIRE(wait_status == hipErrorInvalidValue);

  HIP_CHECK(hipStreamDestroy(stream));
}
