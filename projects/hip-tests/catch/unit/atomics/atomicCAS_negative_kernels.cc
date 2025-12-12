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

#define ATOMIC_CAS_NEGATIVE_KERNEL(type_name)                                                      \
  __global__ void atomicCAS_v1(type_name* address, type_name* result) {                            \
    *result = atomicCAS(&address, 12, 13);                                                         \
  }                                                                                                \
  __global__ void atomicCAS_v2(type_name* address, type_name* result) {                            \
    *result = atomicCAS(address, address, 13);                                                     \
  }                                                                                                \
  __global__ void atomicCAS_v3(type_name* address, type_name* result) {                            \
    *result = atomicCAS(address, 12, address);                                                     \
  }                                                                                                \
  __global__ void atomicCAS_v4(Dummy* address, type_name* result) {                                \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v5(char* address, type_name* result) {                                 \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v6(short* address, type_name* result) {                                \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v7(long* address, type_name* result) {                                 \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v8(long long* address, type_name* result) {                            \
    *result = atomicCAS(address, 12, 13);                                                          \
  }

ATOMIC_CAS_NEGATIVE_KERNEL(int)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned int)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned long)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned long long)
ATOMIC_CAS_NEGATIVE_KERNEL(float)
ATOMIC_CAS_NEGATIVE_KERNEL(double)
