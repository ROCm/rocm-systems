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

/* int atomicAnd(int* address, int val) */
__global__ void atomicAnd_int_v1(int* address, int* result) { *result = atomicAnd(&address, 1234); }

__global__ void atomicAnd_int_v2(int* address, int* result) {
  *result = atomicAnd(address, address);
}

__global__ void atomicAnd_int_v3(int* address, int* result) { *result = atomicAnd(1234, 1234); }

__global__ void atomicAnd_int_v4(Dummy* address, int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_int_v5(char* address, int* result) { *result = atomicAnd(address, 1234); }

__global__ void atomicAnd_int_v6(short* address, int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_int_v7(long* address, int* result) { *result = atomicAnd(address, 1234); }

__global__ void atomicAnd_int_v8(long long* address, int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_int_v9(float* address, int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_int_v10(double* address, int* result) {
  *result = atomicAnd(address, 1234);
}

/* unsigned int atomicAnd(unsigned int* address, unsigned int val) */
__global__ void atomicAnd_uint_v1(unsigned int* address, unsigned int* result) {
  *result = atomicAnd(&address, 1234);
}

__global__ void atomicAnd_uint_v2(unsigned int* address, unsigned int* result) {
  *result = atomicAnd(address, address);
}

__global__ void atomicAnd_uint_v3(unsigned int* address, unsigned int* result) {
  *result = atomicAnd(1234, 1234);
}

__global__ void atomicAnd_uint_v4(Dummy* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v5(char* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v6(short* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v7(long* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v8(long long* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v9(float* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_uint_v10(double* address, unsigned int* result) {
  *result = atomicAnd(address, 1234);
}

/* atomicAnd(unsigned long* address, unsigned long val) */
__global__ void atomicAnd_ulong_v1(unsigned long* address, unsigned long* result) {
  *result = atomicAnd(&address, 1234);
}

__global__ void atomicAnd_ulong_v2(unsigned long* address, unsigned long* result) {
  *result = atomicAnd(address, address);
}

__global__ void atomicAnd_ulong_v3(unsigned long* address, unsigned long* result) {
  *result = atomicAnd(1234, 1234);
}

__global__ void atomicAnd_ulong_v4(Dummy* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v5(char* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v6(short* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v7(long* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v8(long long* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v9(float* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulong_v10(double* address, unsigned long* result) {
  *result = atomicAnd(address, 1234);
}

/* atomicAnd(unsigned long long* address, unsigned long long val) */
__global__ void atomicAnd_ulonglong_v1(unsigned long long* address, unsigned long long* result) {
  *result = atomicAnd(&address, 1234);
}

__global__ void atomicAnd_ulonglong_v2(unsigned long long* address, unsigned long long* result) {
  *result = atomicAnd(address, address);
}

__global__ void atomicAnd_ulonglong_v3(unsigned long long* address, unsigned long long* result) {
  *result = atomicAnd(1234, 1234);
}

__global__ void atomicAnd_ulonglong_v4(Dummy* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v5(char* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v6(short* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v7(long* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v8(long long* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v9(float* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}

__global__ void atomicAnd_ulonglong_v10(double* address, unsigned long long* result) {
  *result = atomicAnd(address, 1234);
}
