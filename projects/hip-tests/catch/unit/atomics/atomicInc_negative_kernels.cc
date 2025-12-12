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

/* unsigned int atomicInc(unsigned int* address, unsigned int val) */
__global__ void atomicInc_uint_v1(unsigned int* address, unsigned int* result) {
  *result = atomicInc(&address, 1234);
}

__global__ void atomicInc_uint_v2(unsigned int* address, unsigned int* result) {
  *result = atomicInc(address, address);
}

__global__ void atomicInc_uint_v3(unsigned int* address, unsigned int* result) {
  *result = atomicInc(1234, 1234);
}

__global__ void atomicInc_uint_v4(Dummy* address, unsigned int* result) {
  *result = atomicInc(address, 1234);
}

__global__ void atomicInc_uint_v5(char* address, unsigned int* result) {
  *result = atomicInc(address, 1234);
}

__global__ void atomicInc_uint_v6(short* address, unsigned int* result) {
  *result = atomicInc(address, 1234);
}

__global__ void atomicInc_uint_v7(long* address, unsigned int* result) {
  *result = atomicInc(address, 1234);
}

__global__ void atomicInc_uint_v8(long long* address, unsigned int* result) {
  *result = atomicInc(address, 1234);
}