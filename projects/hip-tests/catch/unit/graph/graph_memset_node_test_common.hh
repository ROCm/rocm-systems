/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

template <typename T, typename Tp, typename F> void GraphMemsetNodeCommonPositive(F f) {
  const size_t width = GENERATE(1, 64, kPageSize / sizeof(T) + 1);
  const size_t height = GENERATE(1, 2, 1024);
  DYNAMIC_SECTION("Width: " << width << " Height: " << height) {
    LinearAllocGuard2D<T> alloc(width, height);

    constexpr T set_value = 42;
    Tp params = {};
    params.dst = alloc.ptr();
    params.elementSize = sizeof(T);
    params.width = width;
    params.height = height;
    params.pitch = alloc.pitch();
    params.value = set_value;

    HIP_CHECK(f(&params));

    LinearAllocGuard<T> buffer(LinearAllocs::hipHostMalloc, width * sizeof(T) * height);
    HIP_CHECK(hipMemcpy2D(buffer.ptr(), width * sizeof(T), alloc.ptr(), alloc.pitch(),
                          width * sizeof(T), height, hipMemcpyDeviceToHost));
    ArrayFindIfNot(buffer.ptr(), set_value, width * height);
  }
}

template <typename F, typename T> void MemsetCommonNegative(F f, T params) {
  SECTION("pMemsetParams == nullptr") { HIP_CHECK_ERROR(f(nullptr), hipErrorInvalidValue); }

  SECTION("pMemsetParams.dst == nullptr") {
    params.dst = nullptr;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.elementSize != 1, 2, 4") {
    params.elementSize = GENERATE(0, 3, 5);
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-204
#if HT_NVIDIA
  SECTION("pMemsetParams.width == 0") {
    params.width = 0;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }
#endif

  SECTION("pMemsetParams.width > allocation size") {
    params.width = params.width + 1000;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.height == 0") {
    params.height = 0;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-205
#if HT_NVIDIA
  SECTION("pMemsetParams.pitch < width when height > 1") {
    params.width = 2;
    params.height = 2;
    params.pitch = params.elementSize;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-206
#if HT_NVIDIA
  SECTION("pMemsetParams.pitch * height > allocation size") {
    params.width = 2;
    params.height = 2;
    params.pitch = 3 * params.elementSize;
    HIP_CHECK_ERROR(f(&params), hipErrorInvalidValue);
  }
#endif
}