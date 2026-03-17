/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <resource_guards.hh>

HIP_TEST_CASE(Unit_hipLaunchByPtr_Positive_Basic) {
  LinearAllocGuard<int> alloc(LinearAllocs::hipMallocManaged, 4);

  SECTION("hipConfigureCall") { HIP_CHECK(hipConfigureCall(dim3{1}, dim3{1}, 0, nullptr)); }

  SECTION("__hipPushCallConfiguration") {
    HIP_CHECK(__hipPushCallConfiguration(dim3{1}, dim3{1}, 0, nullptr));
  }

  int* arg = alloc.ptr();
  HIP_CHECK(hipSetupArgument(&arg, sizeof(int*), 0));

  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<void*>(kernel_42)));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(alloc.ptr()[0] == 42);
}

HIP_TEST_CASE(Unit_hipLaunchByPtr_Negative_Parameters) {
  HIP_CHECK(hipConfigureCall(dim3{1}, dim3{1}, 0, nullptr));
  HIP_CHECK_ERROR(hipLaunchByPtr(nullptr), hipErrorInvalidDeviceFunction);
}

HIP_TEST_CASE(Unit___hipPushCallConfiguration_Positive_Basic) {
  StreamGuard stream_guard(Streams::created);
  HIP_CHECK(__hipPushCallConfiguration(dim3{1, 2, 3}, dim3{3, 2, 1}, 1024, stream_guard.stream()));

  dim3 grid;
  dim3 block;
  size_t shmem;
  hipStream_t stream;
  HIP_CHECK(__hipPopCallConfiguration(&grid, &block, &shmem, &stream));

  REQUIRE(grid.x == 1);
  REQUIRE(grid.y == 2);
  REQUIRE(grid.z == 3);
  REQUIRE(block.x == 3);
  REQUIRE(block.y == 2);
  REQUIRE(block.z == 1);
  REQUIRE(shmem == 1024);
  REQUIRE(stream == stream_guard.stream());
}

HIP_TEST_CASE(Unit_hipLaunchByPtr_Verify_Capture) {
  LinearAllocGuard<int> alloc(LinearAllocs::hipMallocManaged, 4);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);

  HIP_CHECK(hipConfigureCall(dim3{1}, dim3{1}, 0, stream));
  int* arg = alloc.ptr();
  HIP_CHECK(hipSetupArgument(&arg, sizeof(int*), 0));
  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<void*>(kernel_42)));

  END_CAPTURE(stream);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(alloc.ptr()[0] == 42);
}
