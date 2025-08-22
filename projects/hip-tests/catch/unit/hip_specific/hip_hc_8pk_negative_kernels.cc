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