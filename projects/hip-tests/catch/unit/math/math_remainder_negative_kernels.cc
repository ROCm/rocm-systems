/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#define NEGATIVE_KERNELS_SHELL(func_name)                                                          \
  __global__ void func_name##_kernel_v1(double* x, double y) { auto result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v2(double x, double* y) { auto result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { auto result = func_name(x, y); }      \
  __global__ void func_name##_kernel_v4(double x, Dummy y) { auto result = func_name(x, y); }      \
  __global__ void func_name##f_kernel_v1(float* x, float y) { auto result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v2(float x, float* y) { auto result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v3(Dummy x, float y) { auto result = func_name##f(x, y); }   \
  __global__ void func_name##f_kernel_v4(float x, Dummy y) { auto result = func_name##f(x, y); }

NEGATIVE_KERNELS_SHELL(fmod)
NEGATIVE_KERNELS_SHELL(remainder)

__global__ void remquo_kernel_v1(double* x, double y, int* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v2(Dummy x, double y, int* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v3(double x, double* y, int* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v4(double x, Dummy y, int* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v5(double x, double y, char* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v6(double x, double y, short* quo) {
  auto result = remquo(x, y, quo);
}
__global__ void remquo_kernel_v7(double x, double y, long* quo) { auto result = remquo(x, y, quo); }
__global__ void remquo_kernel_v8(double x, double y, long long* quo) {
  auto result = remquo(x, y, quo);
}
__global__ void remquo_kernel_v9(double x, double y, float* quo) {
  auto result = remquo(x, y, quo);
}
__global__ void remquo_kernel_v10(double x, double y, double* quo) {
  auto result = remquo(x, y, quo);
}
__global__ void remquo_kernel_v11(double x, double y, Dummy* quo) {
  auto result = remquo(x, y, quo);
}
__global__ void remquo_kernel_v12(double x, double y, const int* quo) {
  auto result = remquo(x, y, quo);
}

__global__ void remquof_kernel_v1(float* x, float y, int* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v2(Dummy x, float y, int* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v3(float x, float* y, int* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v4(float x, Dummy y, int* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v5(float x, float y, char* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v6(float x, float y, short* quo) {
  auto result = remquof(x, y, quo);
}
__global__ void remquof_kernel_v7(float x, float y, long* quo) { auto result = remquof(x, y, quo); }
__global__ void remquof_kernel_v8(float x, float y, long long* quo) {
  auto result = remquof(x, y, quo);
}
__global__ void remquof_kernel_v9(float x, float y, float* quo) {
  auto result = remquof(x, y, quo);
}
__global__ void remquof_kernel_v10(float x, float y, double* quo) {
  auto result = remquof(x, y, quo);
}
__global__ void remquof_kernel_v11(float x, float y, Dummy* quo) {
  auto result = remquof(x, y, quo);
}
__global__ void remquof_kernel_v12(float x, float y, const int* quo) {
  auto result = remquof(x, y, quo);
}

__global__ void modf_kernel_v1(double* x, double* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v2(Dummy x, double* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v3(double x, int* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v4(double x, char* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v5(double x, short* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v6(double x, long* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v7(double x, long long* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v8(double x, float* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v9(double x, Dummy* iptr) { auto result = modf(x, iptr); }
__global__ void modf_kernel_v10(double x, const double* iptr) { auto result = modf(x, iptr); }

__global__ void modff_kernel_v1(float* x, float* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v2(Dummy x, float* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v3(float x, int* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v4(float x, char* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v5(float x, short* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v6(float x, long* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v7(float x, long long* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v8(float x, double* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v9(float x, Dummy* iptr) { auto result = modff(x, iptr); }
__global__ void modff_kernel_v10(float x, const float* iptr) { auto result = modff(x, iptr); }

NEGATIVE_KERNELS_SHELL(fdim)
