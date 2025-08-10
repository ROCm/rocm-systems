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

#pragma once

/*
Positive and negative kernels used for the static_assert Test Cases that are using RTC.
*/

static constexpr auto kStaticAssert_Positive{
    R"(
    __global__ void StaticAssertPassKernel1() {
      static_assert(sizeof(int) < sizeof(long), "[StaticAssertPassKernel1]");
    }

    __global__ void StaticAssertPassKernel2() {
      static_assert(10 > 5, "[StaticAssertPassKernel2]");
    }

    __global__ void StaticAssertFailKernel1() {
      static_assert(sizeof(int) > sizeof(long), "[StaticAssertFailKernel1]");
    }

    __global__ void StaticAssertFailKernel2() {
      static_assert(10 < 5, "[StaticAssertFailKernel2]");
    }
  )"};

static constexpr auto kStaticAssert_Negative{
    R"(
    __global__ void StaticAssertErrorKernel1() {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      static_assert(tid % 2 == 1, "[StaticAssertErrorKernel1]");
    }

    __global__ void StaticAssertErrorKernel2() {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      static_assert(++tid > 2, "[StaticAssertErrorKernel2]");
    }
  )"};
