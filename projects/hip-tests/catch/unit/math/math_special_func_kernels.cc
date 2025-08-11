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

#define NEGATIVE_KERNELS_SHELL_ONE_ARG(func_name)                                                  \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }                 \
  __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }             \
  __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }

#define NEGATIVE_KERNELS_SHELL_TWO_ARGS(func_name)                                                 \
  __global__ void func_name##_kernel_v1(int* x, double y) { double result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v2(int x, double* y) { double result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(int x, Dummy y) { double result = func_name(x, y); }       \
  __global__ void func_name##f_kernel_v1(int* x, float y) { float result = func_name##f(x, y); }   \
  __global__ void func_name##f_kernel_v2(int x, float* y) { float result = func_name##f(x, y); }   \
  __global__ void func_name##f_kernel_v3(Dummy x, float y) { float result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v4(int x, Dummy y) { float result = func_name##f(x, y); }

NEGATIVE_KERNELS_SHELL_ONE_ARG(erf)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfc)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfcinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfcx)
NEGATIVE_KERNELS_SHELL_ONE_ARG(normcdf)
NEGATIVE_KERNELS_SHELL_ONE_ARG(normcdfinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(lgamma)
NEGATIVE_KERNELS_SHELL_ONE_ARG(tgamma)
NEGATIVE_KERNELS_SHELL_ONE_ARG(j0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(j1)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(jn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(y0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(y1)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(yn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(cyl_bessel_i0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(cyl_bessel_i1)
