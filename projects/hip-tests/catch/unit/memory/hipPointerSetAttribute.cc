/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
TEST_CASE("Unit_hipPointerSetAttribute_Positive_SyncMemops") {
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
TEST_CASE("Unit_hipPointerSetAttribute_Negative_Parameters") {
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

  SECTION("freed pointer") {
    HIP_CHECK(hipFree(mem.ptr()));
    HIP_CHECK_ERROR(hipPointerSetAttribute(&value, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, mem.ptr()),
                    hipErrorInvalidDevicePointer);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
