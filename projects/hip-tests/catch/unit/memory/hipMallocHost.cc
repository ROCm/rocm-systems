/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

static __global__ void write_integer(int* memory, int value) { *memory = value; }

HIP_TEST_CASE(Unit_hipMallocHost_Positive) {
  int* host_memory = nullptr;

  HIP_CHECK(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));
  REQUIRE(host_memory != nullptr);
  HIP_CHECK(hipHostFree(host_memory));
}

HIP_TEST_CASE(Unit_hipMallocHost_DataValidation) {
  int validation_number = 10;
  int* host_memory = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));

  write_integer<<<1, 1>>>(host_memory, validation_number);

  SECTION("device sync") { HIP_CHECK(hipDeviceSynchronize()); }

  SECTION("event sync") {
    HIP_CHECK(hipEventCreateWithFlags(&event, 0));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
  }

  SECTION("stream sync") { HIP_CHECK(hipStreamSynchronize(nullptr)); }

  REQUIRE(*host_memory == validation_number);

  if (event != nullptr) {
    HIP_CHECK(hipEventDestroy(event));
  }

  HIP_CHECK(hipHostFree(host_memory));
}

HIP_TEST_CASE(Unit_hipMallocHost_Negative) {
  int* host_memory = nullptr;

  SECTION("host memory is nullptr") {
    HIP_CHECK_ERROR(hipMallocHost(nullptr, sizeof(int)), hipErrorInvalidValue);
  }

  SECTION("size is negative") {
    HIP_CHECK_ERROR(hipMallocHost(reinterpret_cast<void**>(&host_memory), -1), hipErrorOutOfMemory);
  }
}

HIP_TEST_CASE(Unit_hipMallocHost_Capture) {
  int* host_memory = nullptr;

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;

  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)),
                  capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error == hipSuccess) {
    REQUIRE(host_memory != nullptr);
    HIP_CHECK(hipHostFree(host_memory));
  }
}
