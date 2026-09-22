/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

/* int atomicSub(int* address, int val) */
__global__ void atomicSub_int_v1(int* address, int* result) { *result = atomicSub(&address, 1234); }

__global__ void atomicSub_int_v2(int* address, int* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_int_v3(int* address, int* result) { *result = atomicSub(1234, 1234); }

__global__ void atomicSub_int_v4(Dummy* address, int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_int_v5(char* address, int* result) { *result = atomicSub(address, 1234); }

__global__ void atomicSub_int_v6(short* address, int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_int_v7(long* address, int* result) { *result = atomicSub(address, 1234); }

__global__ void atomicSub_int_v8(long long* address, int* result) {
  *result = atomicSub(address, 1234);
}

/* unsigned int atomicSub(unsigned int* address, unsigned int val) */
__global__ void atomicSub_uint_v1(unsigned int* address, unsigned int* result) {
  *result = atomicSub(&address, 1234);
}

__global__ void atomicSub_uint_v2(unsigned int* address, unsigned int* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_uint_v3(unsigned int* address, unsigned int* result) {
  *result = atomicSub(1234, 1234);
}

__global__ void atomicSub_uint_v4(Dummy* address, unsigned int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_uint_v5(char* address, unsigned int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_uint_v6(short* address, unsigned int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_uint_v7(long* address, unsigned int* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_uint_v8(long long* address, unsigned int* result) {
  *result = atomicSub(address, 1234);
}

/* atomicSub(unsigned long* address, unsigned long val) */
__global__ void atomicSub_ulong_v1(unsigned long* address, unsigned long* result) {
  *result = atomicSub(&address, 1234);
}

__global__ void atomicSub_ulong_v2(unsigned long* address, unsigned long* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_ulong_v3(unsigned long* address, unsigned long* result) {
  *result = atomicSub(1234, 1234);
}

__global__ void atomicSub_ulong_v4(Dummy* address, unsigned long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulong_v5(char* address, unsigned long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulong_v6(short* address, unsigned long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulong_v7(long* address, unsigned long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulong_v8(long long* address, unsigned long* result) {
  *result = atomicSub(address, 1234);
}

/* atomicSub(unsigned long long* address, unsigned long long val) */
__global__ void atomicSub_ulonglong_v1(unsigned long long* address, unsigned long long* result) {
  *result = atomicSub(&address, 1234);
}

__global__ void atomicSub_ulonglong_v2(unsigned long long* address, unsigned long long* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_ulonglong_v3(unsigned long long* address, unsigned long long* result) {
  *result = atomicSub(1234, 1234);
}

__global__ void atomicSub_ulonglong_v4(Dummy* address, unsigned long long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulonglong_v5(char* address, unsigned long long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulonglong_v6(short* address, unsigned long long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulonglong_v7(long* address, unsigned long long* result) {
  *result = atomicSub(address, 1234);
}

__global__ void atomicSub_ulonglong_v8(long long* address, unsigned long long* result) {
  *result = atomicSub(address, 1234);
}

/* atomicSub(float* address, float val) */
__global__ void atomicSub_float_v1(float* address, float* result) {
  *result = atomicSub(&address, 1234.f);
}

__global__ void atomicSub_float_v2(float* address, float* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_float_v3(float* address, float* result) {
  *result = atomicSub(1234.f, 1234.f);
}

__global__ void atomicSub_float_v4(Dummy* address, float* result) {
  *result = atomicSub(address, 1234.f);
}

__global__ void atomicSub_float_v5(char* address, float* result) {
  *result = atomicSub(address, 1234.f);
}

__global__ void atomicSub_float_v6(short* address, float* result) {
  *result = atomicSub(address, 1234.f);
}

__global__ void atomicSub_float_v7(long* address, float* result) {
  *result = atomicSub(address, 1234.f);
}

__global__ void atomicSub_float_v8(long long* address, float* result) {
  *result = atomicSub(address, 1234);
}

/* atomicSub(double* address, double val) */
__global__ void atomicSub_double_v1(double* address, double* result) {
  *result = atomicSub(&address, 1234.0);
}

__global__ void atomicSub_double_v2(double* address, double* result) {
  *result = atomicSub(address, address);
}

__global__ void atomicSub_double_v3(double* address, double* result) {
  *result = atomicSub(1234.0, 1234.0);
}

__global__ void atomicSub_double_v4(Dummy* address, double* result) {
  *result = atomicSub(address, 1234.0);
}

__global__ void atomicSub_double_v5(char* address, double* result) {
  *result = atomicSub(address, 1234.0);
}

__global__ void atomicSub_double_v6(short* address, double* result) {
  *result = atomicSub(address, 1234.0);
}

__global__ void atomicSub_double_v7(long* address, double* result) {
  *result = atomicSub(address, 1234.0);
}

__global__ void atomicSub_double_v8(long long* address, double* result) {
  *result = atomicSub(address, 1234.0);
}
