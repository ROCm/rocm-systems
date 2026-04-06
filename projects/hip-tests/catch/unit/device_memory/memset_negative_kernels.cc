/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

struct Dummy {
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

/*void* memset(void* ptr, int val, size_t size)*/
__global__ void memset_n1(int* ptr, int* val, size_t size) { memset(ptr, val, size); }
__global__ void memset_n2(int ptr, int val, size_t size) { memset(ptr, val, size); }
__global__ void memset_n3(int* ptr, int val, size_t* size) { memset(ptr, val, size); }
__global__ void memset_n4(int* ptr, int val, Dummy size) { memset(ptr, val, size); }