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

#define NEGATIVE_KERNELS_SHELL(func_name, T)                                                       \
  __global__ void func_name##_kernel_v1(T* result, float* x) { *result = func_name(x); }           \
  __global__ void func_name##_kernel_v2(T* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v3(Dummy* result, float x) { *result = func_name(x); }

NEGATIVE_KERNELS_SHELL(__float2int_rd, int)
NEGATIVE_KERNELS_SHELL(__float2int_rn, int)
NEGATIVE_KERNELS_SHELL(__float2int_ru, int)
NEGATIVE_KERNELS_SHELL(__float2int_rz, int)
NEGATIVE_KERNELS_SHELL(__float2uint_rd, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_rn, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_ru, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_rz, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2ll_rd, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_rn, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_ru, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_rz, long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rd, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rn, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_ru, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rz, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float_as_int, int)
NEGATIVE_KERNELS_SHELL(__float_as_uint, unsigned int)