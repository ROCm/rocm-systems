/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kSmallAllocSize = 4096;
constexpr uintptr_t kDevicePointerAlignment = 256;
}

// @asserts: hipMalloc - a successful allocation returns a non-null pointer aligned to at least 256 bytes
HIP_TEST_CASE(Contract_Memory_MallocBasic_ReturnsAlignedPointer) {
  hip::contract::ContractCleanup cleanup;
  void* ptr = nullptr;

  HIP_CHECK(hipMalloc(&ptr, kSmallAllocSize));
  cleanup.Add([ptr] { (void)hipFree(ptr); });

  REQUIRE(ptr != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr) % kDevicePointerAlignment == 0);
}

// @asserts: hipMalloc - a zero-byte allocation succeeds and writes back a null pointer
HIP_TEST_CASE(Contract_Memory_MallocZeroSize_ReturnsNull) {
  void* ptr = reinterpret_cast<void*>(0x1);

  HIP_CHECK(hipMalloc(&ptr, 0));

  REQUIRE(ptr == nullptr);
}

// @asserts: hipMalloc - a null out-pointer argument is rejected with hipErrorInvalidValue
HIP_TEST_CASE(Contract_Memory_MallocNullOutPointer_ReturnsInvalidValue) {
  HIP_CHECK_ERROR(hipMalloc(nullptr, kSmallAllocSize), hipErrorInvalidValue);
}

// @asserts: hipFree - freeing a null pointer is a no-op that succeeds
HIP_TEST_CASE(Contract_Memory_FreeNull_Succeeds) {
  HIP_CHECK(hipFree(nullptr));
}

// @asserts: hipFree - freeing a pointer returned by hipMalloc succeeds
HIP_TEST_CASE(Contract_Memory_FreeAllocatedPointer_Succeeds) {
  void* ptr = nullptr;

  HIP_CHECK(hipMalloc(&ptr, kSmallAllocSize));
  REQUIRE(ptr != nullptr);

  HIP_CHECK(hipFree(ptr));
}
