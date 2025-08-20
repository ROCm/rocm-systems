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
Negative kernels used for the <unsigned> int/long long type casting negative Test Cases that are using RTC.
*/

static constexpr auto kInt2Float{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void int2float_rd_kernel_v1(float* result, int* x) { *result = __int2float_rd(x); }
  __global__ void int2float_rd_kernel_v2(float* result, Dummy x) { *result = __int2float_rd(x); }
  __global__ void int2float_rd_kernel_v3(Dummy* result, int x) { *result = __int2float_rd(x); }
  __global__ void int2float_rn_kernel_v1(float* result, int* x) { *result = __int2float_rn(x); }
  __global__ void int2float_rn_kernel_v2(float* result, Dummy x) { *result = __int2float_rn(x); }
  __global__ void int2float_rn_kernel_v3(Dummy* result, int x) { *result = __int2float_rn(x); }
  __global__ void int2float_ru_kernel_v1(float* result, int* x) { *result = __int2float_ru(x); }
  __global__ void int2float_ru_kernel_v2(float* result, Dummy x) { *result = __int2float_ru(x); }
  __global__ void int2float_ru_kernel_v3(Dummy* result, int x) { *result = __int2float_ru(x); }
  __global__ void int2float_rz_kernel_v1(float* result, int* x) { *result = __int2float_rz(x); }
  __global__ void int2float_rz_kernel_v2(float* result, Dummy x) { *result = __int2float_rz(x); }
  __global__ void int2float_rz_kernel_v3(Dummy* result, int x) { *result = __int2float_rz(x); }
)"};

static constexpr auto kUint2Float{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void uint2float_rd_kernel_v1(float* result, unsigned int* x) { *result = __uint2float_rd(x); }
  __global__ void uint2float_rd_kernel_v2(float* result, Dummy x) { *result = __uint2float_rd(x); }
  __global__ void uint2float_rd_kernel_v3(Dummy* result, unsigned int x) { *result = __uint2float_rd(x); }
  __global__ void uint2float_rn_kernel_v1(float* result, unsigned int* x) { *result = __uint2float_rn(x); }
  __global__ void uint2float_rn_kernel_v2(float* result, Dummy x) { *result = __uint2float_rn(x); }
  __global__ void uint2float_rn_kernel_v3(Dummy* result, unsigned int x) { *result = __uint2float_rn(x); }
  __global__ void uint2float_ru_kernel_v1(float* result, unsigned int* x) { *result = __uint2float_ru(x); }
  __global__ void uint2float_ru_kernel_v2(float* result, Dummy x) { *result = __uint2float_ru(x); }
  __global__ void uint2float_ru_kernel_v3(Dummy* result, unsigned int x) { *result = __uint2float_ru(x); }
  __global__ void uint2float_rz_kernel_v1(float* result, unsigned int* x) { *result = __uint2float_rz(x); }
  __global__ void uint2float_rz_kernel_v2(float* result, Dummy x) { *result = __uint2float_rz(x); }
  __global__ void uint2float_rz_kernel_v3(Dummy* result, unsigned int x) { *result = __uint2float_rz(x); }
)"};

static constexpr auto kLL2Float{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ll2float_rd_kernel_v1(float* result, long long int* x) { *result = __ll2float_rd(x); }
  __global__ void ll2float_rd_kernel_v2(float* result, Dummy x) { *result = __ll2float_rd(x); }
  __global__ void ll2float_rd_kernel_v3(Dummy* result, long long int x) { *result = __ll2float_rd(x); }
  __global__ void ll2float_rn_kernel_v1(float* result, long long int* x) { *result = __ll2float_rn(x); }
  __global__ void ll2float_rn_kernel_v2(float* result, Dummy x) { *result = __ll2float_rn(x); }
  __global__ void ll2float_rn_kernel_v3(Dummy* result, long long int x) { *result = __ll2float_rn(x); }
  __global__ void ll2float_ru_kernel_v1(float* result, long long int* x) { *result = __ll2float_ru(x); }
  __global__ void ll2float_ru_kernel_v2(float* result, Dummy x) { *result = __ll2float_ru(x); }
  __global__ void ll2float_ru_kernel_v3(Dummy* result, long long int x) { *result = __ll2float_ru(x); }
  __global__ void ll2float_rz_kernel_v1(float* result, long long int* x) { *result = __ll2float_rz(x); }
  __global__ void ll2float_rz_kernel_v2(float* result, Dummy x) { *result = __ll2float_rz(x); }
  __global__ void ll2float_rz_kernel_v3(Dummy* result, long long int x) { *result = __ll2float_rz(x); }
)"};

static constexpr auto kULL2Float{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ull2float_rd_kernel_v1(float* result, unsigned long long int* x) { *result = __ull2float_rd(x); }
  __global__ void ull2float_rd_kernel_v2(float* result, Dummy x) { *result = __ull2float_rd(x); }
  __global__ void ull2float_rd_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2float_rd(x); }
  __global__ void ull2float_rn_kernel_v1(float* result, unsigned long long int* x) { *result = __ull2float_rn(x); }
  __global__ void ull2float_rn_kernel_v2(float* result, Dummy x) { *result = __ull2float_rn(x); }
  __global__ void ull2float_rn_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2float_rn(x); }
  __global__ void ull2float_ru_kernel_v1(float* result, unsigned long long int* x) { *result = __ull2float_ru(x); }
  __global__ void ull2float_ru_kernel_v2(float* result, Dummy x) { *result = __ull2float_ru(x); }
  __global__ void ull2float_ru_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2float_ru(x); }
  __global__ void ull2float_rz_kernel_v1(float* result, unsigned long long int* x) { *result = __ull2float_rz(x); }
  __global__ void ull2float_rz_kernel_v2(float* result, Dummy x) { *result = __ull2float_rz(x); }
  __global__ void ull2float_rz_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2float_rz(x); }
)"};

static constexpr auto kIntAsFloat{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void int_as_float_kernel_v1(float* result, int* x) { *result = __int_as_float(x); }
  __global__ void int_as_float_kernel_v2(float* result, Dummy x) { *result = __int_as_float(x); }
  __global__ void int_as_float_kernel_v3(Dummy* result, int x) { *result = __int_as_float(x); }
)"};

static constexpr auto kUintAsFloat{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void uint_as_float_kernel_v1(float* result, unsigned int* x) { *result = __uint_as_float(x); }
  __global__ void uint_as_float_kernel_v2(float* result, Dummy x) { *result = __uint_as_float(x); }
  __global__ void uint_as_float_kernel_v3(Dummy* result, unsigned int x) { *result = __uint_as_float(x); }
)"};

static constexpr auto kInt2Double{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void int2double_rn_kernel_v1(double* result, int* x) { *result = __int2double_rn(x); }
  __global__ void int2double_rn_kernel_v2(double* result, Dummy x) { *result = __int2double_rn(x); }
  __global__ void int2double_rn_kernel_v3(Dummy* result, int x) { *result = __int2double_rn(x); }
)"};

static constexpr auto kUint2Double{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void uint2double_rn_kernel_v1(double* result, unsigned int* x) { *result = __uint2double_rn(x); }
  __global__ void uint2double_rn_kernel_v2(double* result, Dummy x) { *result = __uint2double_rn(x); }
  __global__ void uint2double_rn_kernel_v3(Dummy* result, unsigned int x) { *result = __uint2double_rn(x); }
)"};


static constexpr auto kLL2Double{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ll2double_rd_kernel_v1(double* result, long long int* x) { *result = __ll2double_rd(x); }
  __global__ void ll2double_rd_kernel_v2(double* result, Dummy x) { *result = __ll2double_rd(x); }
  __global__ void ll2double_rd_kernel_v3(Dummy* result, long long int x) { *result = __ll2double_rd(x); }
  __global__ void ll2double_rn_kernel_v1(double* result, long long int* x) { *result = __ll2double_rn(x); }
  __global__ void ll2double_rn_kernel_v2(double* result, Dummy x) { *result = __ll2double_rn(x); }
  __global__ void ll2double_rn_kernel_v3(Dummy* result, long long int x) { *result = __ll2double_rn(x); }
  __global__ void ll2double_ru_kernel_v1(double* result, long long int* x) { *result = __ll2double_ru(x); }
  __global__ void ll2double_ru_kernel_v2(double* result, Dummy x) { *result = __ll2double_ru(x); }
  __global__ void ll2double_ru_kernel_v3(Dummy* result, long long int x) { *result = __ll2double_ru(x); }
  __global__ void ll2double_rz_kernel_v1(double* result, long long int* x) { *result = __ll2double_rz(x); }
  __global__ void ll2double_rz_kernel_v2(double* result, Dummy x) { *result = __ll2double_rz(x); }
  __global__ void ll2double_rz_kernel_v3(Dummy* result, long long int x) { *result = __ll2double_rz(x); }
)"};

static constexpr auto kULL2Double{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ull2double_rd_kernel_v1(double* result, unsigned long long int* x) { *result = __ull2double_rd(x); }
  __global__ void ull2double_rd_kernel_v2(double* result, Dummy x) { *result = __ull2double_rd(x); }
  __global__ void ull2double_rd_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2double_rd(x); }
  __global__ void ull2double_rn_kernel_v1(double* result, unsigned long long int* x) { *result = __ull2double_rn(x); }
  __global__ void ull2double_rn_kernel_v2(double* result, Dummy x) { *result = __ull2double_rn(x); }
  __global__ void ull2double_rn_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2double_rn(x); }
  __global__ void ull2double_ru_kernel_v1(double* result, unsigned long long int* x) { *result = __ull2double_ru(x); }
  __global__ void ull2double_ru_kernel_v2(double* result, Dummy x) { *result = __ull2double_ru(x); }
  __global__ void ull2double_ru_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2double_ru(x); }
  __global__ void ull2double_rz_kernel_v1(double* result, unsigned long long int* x) { *result = __ull2double_rz(x); }
  __global__ void ull2double_rz_kernel_v2(double* result, Dummy x) { *result = __ull2double_rz(x); }
  __global__ void ull2double_rz_kernel_v3(Dummy* result, unsigned long long int x) { *result = __ull2double_rz(x); }
)"};

static constexpr auto kLonglongAsDouble{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void longlong_as_double_kernel_v1(double* result, long long int* x) { *result = __longlong_as_double(x); }
  __global__ void longlong_as_double_kernel_v2(double* result, Dummy x) { *result = __longlong_as_double(x); }
  __global__ void longlong_as_double_kernel_v3(Dummy* result, long long int x) { *result = __longlong_as_double(x); }
)"};

static constexpr auto kHilo2Double{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hiloint2double_kernel_v1(double* result, int* x, int y) { *result = __hiloint2double(x, y); }
  __global__ void hiloint2double_kernel_v2(double* result, int x, int* y) { *result = __hiloint2double(x, y); }
  __global__ void hiloint2double_kernel_v3(double* result, Dummy x, int y) { *result = __hiloint2double(x, y); }
  __global__ void hiloint2double_kernel_v4(double* result, int x, Dummy y) { *result = __hiloint2double(x, y); }
  __global__ void hiloint2double_kernel_v5(Dummy* result, int x, int y) { *result = __hiloint2double(x, y); }
)"};


