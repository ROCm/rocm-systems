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

/*void* memcpy(void* dst, const void* src, size_t size)*/
__global__ void memcpy_n1(int* dst, const int src, size_t size) { memcpy(dst, src, size); }
__global__ void memcpy_n2(int dst, const int* src, size_t size) { memcpy(dst, src, size); }
__global__ void memcpy_n3(int* dst, const int* src, size_t* size) { memcpy(dst, src, size); }
__global__ void memcpy_n4(int* dst, const int* src, Dummy size) { memcpy(dst, src, size); }