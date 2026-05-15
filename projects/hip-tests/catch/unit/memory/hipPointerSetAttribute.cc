/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipPointerSetAttribute hipPointerSetAttribute
 * @{
 * @ingroup MemoryTest
 * `hipPointerSetAttribute(const void* value, hipPointer_attribute attribute, hipDeviceptr_t ptr)` -
 * Set attributes on a previously allocated memory region.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * Test Description
 * ------------------------
 *  - Sets pointer attribute `HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS` and verifies behavior.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerSetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
HIP_TEST_CASE(Unit_hipPointerSetAttribute_Positive_SyncMemops) {
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, 1024);
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, 1024);

  StreamGuard stream(Streams::created);
  LaunchDelayKernel(std::chrono::milliseconds{100}, stream.stream());
  HIP_CHECK(hipMemcpy(dst.ptr(), src.ptr(), 1024, hipMemcpyDeviceToDevice));
  HIP_CHECK_ERROR(hipStreamQuery(stream.stream()), hipErrorNotReady);

  int value = 1;
  HIP_CHECK(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   reinterpret_cast<hipDeviceptr_t>(src.ptr())));
  HIP_CHECK(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   reinterpret_cast<hipDeviceptr_t>(dst.ptr())));

  LaunchDelayKernel(std::chrono::milliseconds{100}, stream.stream());
  HIP_CHECK(hipMemcpy(dst.ptr(), src.ptr(), 1024, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipStreamQuery(stream.stream()));
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipPointerSetAttribute`.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerSetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
HIP_TEST_CASE(Unit_hipPointerSetAttribute_Negative_Parameters) {
  LinearAllocGuard<int> mem(LinearAllocs::hipMalloc, 4);
  int value = 0;

  SECTION("value is nullptr") {
    HIP_CHECK_ERROR(hipPointerSetAttribute(nullptr, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, mem.ptr()),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(
        hipPointerSetAttribute(&value, static_cast<hipPointer_attribute>(-1), mem.ptr()),
        hipErrorInvalidValue);
  }

  SECTION("ptr is nullptr") {
    HIP_CHECK_ERROR(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("host pointer") {
    int mem_host;
    HIP_CHECK_ERROR(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, &mem_host),
                    hipErrorInvalidDevicePointer);
  }

#if !defined(ENABLE_ADDRESS_SANITIZER)
  SECTION("freed pointer") {
    HIP_CHECK(hipFree(mem.ptr()));
    HIP_CHECK_ERROR(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, mem.ptr()),
                    hipErrorInvalidDevicePointer);
  }
#endif
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
