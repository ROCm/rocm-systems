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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hip_complex.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_SHELL_TWO_ARG_FLOAT(func_name)                                                    \
  __global__ void func_name##_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,               \
                                        hipFloatComplex y) {                                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(hipFloatComplex* result, hipFloatComplex x,                \
                                        hipFloatComplex* y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,               \
                                        hipFloatComplex y) {                                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(hipFloatComplex* result, hipFloatComplex x,                \
                                        hipDoubleComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,              \
                                         hipFloatComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {    \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v3(hipFloatComplex* result, float x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v4(hipFloatComplex* result, hipFloatComplex x, float y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v9(float* result, hipFloatComplex x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {           \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {                      \
    *result = func_name(x, y);                                                                     \
  }

NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCaddf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCsubf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCmulf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCdivf)