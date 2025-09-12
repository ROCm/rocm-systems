/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
Negative kernels used for the math log negative Test Cases that are using RTC.
*/

static constexpr auto kLog{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void log_kernel_v1(double* x) { double result = log(x); }
  __global__ void log_kernel_v2(Dummy x) { double result = log(x); }
  __global__ void logf_kernel_v1(float* x) { float result = logf(x); }
  __global__ void logf_kernel_v2(Dummy x) { float result = logf(x); }
)"};

static constexpr auto kLog2{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void log2_kernel_v1(double* x) { double result = log2(x); }
  __global__ void log2_kernel_v2(Dummy x) { double result = log2(x); }
  __global__ void log2f_kernel_v1(float* x) { float result = log2f(x); }
  __global__ void log2f_kernel_v2(Dummy x) { float result = log2f(x); }
)"};

static constexpr auto kLog10{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void log10_kernel_v1(double* x) { double result = log10(x); }
  __global__ void log10_kernel_v2(Dummy x) { double result = log10(x); }
  __global__ void log10f_kernel_v1(float* x) { float result = log10f(x); }
  __global__ void log10f_kernel_v2(Dummy x) { float result = log10f(x); }
)"};

static constexpr auto kLog1p{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void log1p_kernel_v1(double* x) { double result = log1p(x); }
  __global__ void log1p_kernel_v2(Dummy x) { double result = log1p(x); }
  __global__ void log1pf_kernel_v1(float* x) { float result = log1pf(x); }
  __global__ void log1pf_kernel_v2(Dummy x) { float result = log1pf(x); }
)"};

static constexpr auto kLogb{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void logb_kernel_v1(double* x) { double result = logb(x); }
  __global__ void logb_kernel_v2(Dummy x) { double result = logb(x); }
  __global__ void logbf_kernel_v1(float* x) { float result = logbf(x); }
  __global__ void logbf_kernel_v2(Dummy x) { float result = logbf(x); }
)"};

static constexpr auto kIlogb{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ilogb_kernel_v1(double* x) { double result = ilogb(x); }
  __global__ void ilogb_kernel_v2(Dummy x) { double result = ilogb(x); }
  __global__ void ilogbf_kernel_v1(float* x) { float result = ilogbf(x); }
  __global__ void ilogbf_kernel_v2(Dummy x) { float result = ilogbf(x); }
)"};
