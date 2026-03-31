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

#define NEGATIVE_KERNELS_SHELL(func_name)                                                          \
  __global__ void func_name##_char_n1(char4* out, char in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_char_n2(char4* out, char4 in1, char in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_char_n3(char* out, char4 in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int_n1(char4* out, int in1, char4 in2) {                             \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int_n2(char4* out, char4 in1, int in2) {                             \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int_n3(int* out, char4 in1, char4 in2) {                             \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long_n1(char4* out, long in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long_n2(char4* out, char4 in1, long in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long_n3(long* out, char4 in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float_n1(char4* out, float in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float_n2(char4* out, char4 in1, float in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float_n3(float* out, char4 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_char4_n1(char4* out, char4 in1, char4 in2) {                         \
    out = func_name(in1, in2);                                                                     \
  }                                                                                                \
  __global__ void func_name##_char4_n2(char4* out, char4 in1, char4 in2) {                         \
    *out = func_name(&in1, in2);                                                                   \
  }                                                                                                \
  __global__ void func_name##_char4_n3(char4* out, char4 in1, char4 in2) {                         \
    *out = func_name(in1, &in2);                                                                   \
  }                                                                                                \
  __global__ void func_name##_char2_n1(char4* out, char2 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_char2_n2(char4* out, char4 in1, char2 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_char2_n3(char2* out, char4 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int4_n1(char4* out, int4 in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int4_n2(char4* out, char4 in1, int4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_int4_n3(int4* out, char4 in1, char4 in2) {                           \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long4_n1(char4* out, long4 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long4_n2(char4* out, char4 in1, long4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_long4_n3(long4* out, char4 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float4_n1(char4* out, float4 in1, char4 in2) {                       \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float4_n2(char4* out, char4 in1, float4 in2) {                       \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_float4_n3(float4* out, char4 in1, char4 in2) {                       \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_dummy_n1(char4* out, Dummy in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_dummy_n2(char4* out, char4 in1, Dummy in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }                                                                                                \
  __global__ void func_name##_dummy_n3(Dummy* out, char4 in1, char4 in2) {                         \
    *out = func_name(in1, in2);                                                                    \
  }

NEGATIVE_KERNELS_SHELL(__hip_hc_add8pk)
NEGATIVE_KERNELS_SHELL(__hip_hc_sub8pk)
NEGATIVE_KERNELS_SHELL(__hip_hc_mul8pk)