/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
