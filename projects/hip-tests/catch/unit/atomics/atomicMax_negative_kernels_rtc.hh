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
Negative kernels used for the atomics negative Test Cases that are using RTC.
*/

static constexpr auto kAtomicMax_int{
    R"(
    __global__ void atomicMax_int_v1(int* address, int* result) {
      *result = atomicMax(&address, 1234);
    }

    __global__ void atomicMax_int_v2(int* address, int* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_int_v3(int* address, int* result) {
      *result = atomicMax(1234, 1234);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_int_v4(Dummy* address, int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_int_v5(char* address, int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_int_v6(short* address, int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_int_v7(long* address, int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_int_v8(long long* address, int* result) {
      *result = atomicMax(address, 1234);
    }
  )"};

static constexpr auto kAtomicMax_uint{
    R"(
    __global__ void atomicMax_uint_v1(unsigned int* address, unsigned int* result) {
      *result = atomicMax(&address, 1234);
    }

    __global__ void atomicMax_uint_v2(unsigned int* address, unsigned int* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_uint_v3(unsigned int* address, unsigned int* result) {
      *result = atomicMax(1234, 1234);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_uint_v4(Dummy* address, unsigned int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_uint_v5(char* address, unsigned int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_uint_v6(short* address, unsigned int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_uint_v7(long* address, unsigned int* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_uint_v8(long long* address, unsigned int* result) {
      *result = atomicMax(address, 1234);
    }
  )"};

static constexpr auto kAtomicMax_ulong{
    R"(
    __global__ void atomicMax_ulong_v1(unsigned long* address, unsigned long* result) {
      *result = atomicMax(&address, 1234);
    }

    __global__ void atomicMax_ulong_v2(unsigned long* address, unsigned long* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_ulong_v3(unsigned long* address, unsigned long* result) {
      *result = atomicMax(1234, 1234);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_ulong_v4(Dummy* address, unsigned long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulong_v5(char* address, unsigned long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulong_v6(short* address, unsigned long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulong_v7(long* address, unsigned long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulong_v8(long long* address, unsigned long* result) {
      *result = atomicMax(address, 1234);
    }
  )"};

static constexpr auto kAtomicMax_ulonglong{
    R"(
    __global__ void atomicMax_ulonglong_v1(unsigned long long* address, unsigned long long* result) {
      *result = atomicMax(&address, 1234);
    }

    __global__ void atomicMax_ulonglong_v2(unsigned long long* address, unsigned long long* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_ulonglong_v3(unsigned long long* address, unsigned long long* result) {
      *result = atomicMax(1234, 1234);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_ulonglong_v4(Dummy* address, unsigned long long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulonglong_v5(char* address, unsigned long long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulonglong_v6(short* address, unsigned long long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulonglong_v7(long* address, unsigned long long* result) {
      *result = atomicMax(address, 1234);
    }

    __global__ void atomicMax_ulonglong_v8(long long* address, unsigned long long* result) {
      *result = atomicMax(address, 1234);
    }
  )"};

static constexpr auto kAtomicMax_float{
    R"(
    __global__ void atomicMax_float_v1(float* address, float* result) {
      *result = atomicMax(&address, 1234.f);
    }

    __global__ void atomicMax_float_v2(float* address, float* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_float_v3(float* address, float* result) {
      *result = atomicMax(1234.f, 1234.f);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_float_v4(Dummy* address, float* result) {
      *result = atomicMax(address, 1234.f);
    }

    __global__ void atomicMax_float_v5(char* address, float* result) {
      *result = atomicMax(address, 1234.f);
    }

    __global__ void atomicMax_float_v6(short* address, float* result) {
      *result = atomicMax(address, 1234.f);
    }

    __global__ void atomicMax_float_v7(long* address, float* result) {
      *result = atomicMax(address, 1234.f);
    }

    __global__ void atomicMax_float_v8(long long* address, float* result) {
      *result = atomicMax(address, 1234);
    }
  )"};

static constexpr auto kAtomicMax_double{
    R"(
    __global__ void atomicMax_double_v1(double* address, double* result) {
      *result = atomicMax(&address, 1234.0);
    }

    __global__ void atomicMax_double_v2(double* address, double* result) {
      *result = atomicMax(address, address);
    }

    __global__ void atomicMax_double_v3(double* address, double* result) {
      *result = atomicMax(1234.0, 1234.0);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicMax_double_v4(Dummy* address, double* result) {
      *result = atomicMax(address, 1234.0);
    }

    __global__ void atomicMax_double_v5(char* address, double* result) {
      *result = atomicMax(address, 1234.0);
    }

    __global__ void atomicMax_double_v6(short* address, double* result) {
      *result = atomicMax(address, 1234.0);
    }

    __global__ void atomicMax_double_v7(long* address, double* result) {
      *result = atomicMax(address, 1234.0);
    }

    __global__ void atomicMax_double_v8(long long* address, double* result) {
      *result = atomicMax(address, 1234.0);
    }
  )"};
