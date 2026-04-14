/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

static __global__ void write_integer(int* memory, int value) { *memory = value; }

TEST_CASE(Unit_hipMemAllocHost_Positive) {
  int* host_memory = nullptr;
  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));
  HIP_CHECK(hipMemAllocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));
  REQUIRE(host_memory != nullptr);
  HIP_CHECK(hipHostFree(host_memory));
  HIP_CHECK(hipCtxDestroy(ctx));
}

TEST_CASE(Unit_hipMemAllocHost_DataValidation) {
  int validation_number = 10;
  int* host_memory = nullptr;
  hipEvent_t event = nullptr;
  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, 0));
  HIP_CHECK(hipMemAllocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));

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
  HIP_CHECK(hipCtxDestroy(ctx));
}

TEST_CASE(Unit_hipMemAllocHost_Negative) {
  int* host_memory = nullptr;
  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, 0));

  SECTION("host memory is nullptr") {
    HIP_CHECK_ERROR(hipMemAllocHost(nullptr, sizeof(int)), hipErrorInvalidValue);
  }

  SECTION("size is negative") {
    HIP_CHECK_ERROR(hipMemAllocHost(reinterpret_cast<void**>(&host_memory), -1),
                    hipErrorOutOfMemory);
  }

  HIP_CHECK(hipCtxDestroy(ctx));
}

/*
 * Verify that a device can read/write to the memory of another device
 */
TEST_CASE(Unit_hipMemAllocHost_VerifyAccess) {
  int devices_number = 0;
  HIP_CHECK(hipGetDeviceCount(&devices_number));
  std::vector<int*> devices_memories(devices_number);
  std::vector<hipCtx_t> devices_ctxs(devices_number);

  for (int device_index = 0; device_index < devices_number; device_index++) {
    int support_unified_adressing = 0;

    HIP_CHECK(hipSetDevice(device_index));
    HIP_CHECK(hipDeviceGetAttribute(&support_unified_adressing, hipDeviceAttributeUnifiedAddressing,
                                    device_index));

    if (!support_unified_adressing) {
      HipTest::HIP_SKIP_TEST("Unified adressing is not supported.");
      return;
    }
  }

  for (int device_index = 0; device_index < devices_number; device_index++) {
    HIP_CHECK(hipSetDevice(device_index));

    HIP_CHECK(hipCtxCreate(&devices_ctxs[device_index], 0, device_index));
    HIP_CHECK(
        hipMemAllocHost(reinterpret_cast<void**>(&devices_memories[device_index]), sizeof(int)));
  }

  HIP_CHECK(hipSetDevice(devices_number - 1));
  write_integer<<<1, 1>>>(devices_memories[0], 0);
  HIP_CHECK(hipDeviceSynchronize());

  for (int device_index = 1; device_index < devices_number; device_index++) {
    HIP_CHECK(hipSetDevice(device_index - 1));
    write_integer<<<1, 1>>>(devices_memories[device_index], device_index);
    HIP_CHECK(hipDeviceSynchronize());
  }

  for (int device_index = 0; device_index < devices_number; device_index++) {
    REQUIRE(*devices_memories[device_index] == device_index);
    HIP_CHECK(hipFree(devices_memories[device_index]));
    HIP_CHECK(hipCtxDestroy(devices_ctxs[device_index]));
  }
}

TEST_CASE(Unit_hipMemAllocHost_Capture) {
  int* host_memory = nullptr;

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipMemAllocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)),
                  capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error == hipSuccess) {
    REQUIRE(host_memory != nullptr);
    HIP_CHECK(hipHostFree(host_memory));
  }
}
