/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kSmallAllocSize = 4096;
constexpr uintptr_t kDevicePointerAlignment = 256;
}

HIP_TEST_CASE(Contract_Memory_MallocBasic_ReturnsAlignedPointer) {
  void* ptr = nullptr;

  HIP_CHECK(hipMalloc(&ptr, kSmallAllocSize));

  REQUIRE(ptr != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr) % kDevicePointerAlignment == 0);

  HIP_CHECK(hipFree(ptr));
}

HIP_TEST_CASE(Contract_Memory_MallocZeroSize_ReturnsNull) {
  void* ptr = reinterpret_cast<void*>(0x1);

  HIP_CHECK(hipMalloc(&ptr, 0));

  REQUIRE(ptr == nullptr);
}

HIP_TEST_CASE(Contract_Memory_MallocNullOutPointer_ReturnsInvalidValue) {
  HIP_CHECK_ERROR(hipMalloc(nullptr, kSmallAllocSize), hipErrorInvalidValue);
}

HIP_TEST_CASE(Contract_Memory_FreeNull_Succeeds) {
  HIP_CHECK(hipFree(nullptr));
}

HIP_TEST_CASE(Contract_Memory_FreeAllocatedPointer_Succeeds) {
  void* ptr = nullptr;

  HIP_CHECK(hipMalloc(&ptr, kSmallAllocSize));
  REQUIRE(ptr != nullptr);

  HIP_CHECK(hipFree(ptr));
}
