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

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

/* int atomicOr(int* address, int val) */
__global__ void atomicOr_int_v1(int* address, int* result) { *result = atomicOr(&address, 1234); }

__global__ void atomicOr_int_v2(int* address, int* result) { *result = atomicOr(address, address); }

__global__ void atomicOr_int_v3(int* address, int* result) { *result = atomicOr(1234, 1234); }

__global__ void atomicOr_int_v4(Dummy* address, int* result) { *result = atomicOr(address, 1234); }

__global__ void atomicOr_int_v5(char* address, int* result) { *result = atomicOr(address, 1234); }

__global__ void atomicOr_int_v6(short* address, int* result) { *result = atomicOr(address, 1234); }

__global__ void atomicOr_int_v7(long* address, int* result) { *result = atomicOr(address, 1234); }

__global__ void atomicOr_int_v8(long long* address, int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_int_v9(float* address, int* result) { *result = atomicOr(address, 1234); }

__global__ void atomicOr_int_v10(double* address, int* result) {
  *result = atomicOr(address, 1234);
}

/* unsigned int atomicOr(unsigned int* address, unsigned int val) */
__global__ void atomicOr_uint_v1(unsigned int* address, unsigned int* result) {
  *result = atomicOr(&address, 1234);
}

__global__ void atomicOr_uint_v2(unsigned int* address, unsigned int* result) {
  *result = atomicOr(address, address);
}

__global__ void atomicOr_uint_v3(unsigned int* address, unsigned int* result) {
  *result = atomicOr(1234, 1234);
}

__global__ void atomicOr_uint_v4(Dummy* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v5(char* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v6(short* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v7(long* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v8(long long* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v9(float* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_uint_v10(double* address, unsigned int* result) {
  *result = atomicOr(address, 1234);
}

/* atomicOr(unsigned long* address, unsigned long val) */
__global__ void atomicOr_ulong_v1(unsigned long* address, unsigned long* result) {
  *result = atomicOr(&address, 1234);
}

__global__ void atomicOr_ulong_v2(unsigned long* address, unsigned long* result) {
  *result = atomicOr(address, address);
}

__global__ void atomicOr_ulong_v3(unsigned long* address, unsigned long* result) {
  *result = atomicOr(1234, 1234);
}

__global__ void atomicOr_ulong_v4(Dummy* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v5(char* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v6(short* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v7(long* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v8(long long* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v9(float* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulong_v10(double* address, unsigned long* result) {
  *result = atomicOr(address, 1234);
}

/* atomicOr(unsigned long long* address, unsigned long long val) */
__global__ void atomicOr_ulonglong_v1(unsigned long long* address, unsigned long long* result) {
  *result = atomicOr(&address, 1234);
}

__global__ void atomicOr_ulonglong_v2(unsigned long long* address, unsigned long long* result) {
  *result = atomicOr(address, address);
}

__global__ void atomicOr_ulonglong_v3(unsigned long long* address, unsigned long long* result) {
  *result = atomicOr(1234, 1234);
}

__global__ void atomicOr_ulonglong_v4(Dummy* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v5(char* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v6(short* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v7(long* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v8(long long* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v9(float* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}

__global__ void atomicOr_ulonglong_v10(double* address, unsigned long long* result) {
  *result = atomicOr(address, 1234);
}
