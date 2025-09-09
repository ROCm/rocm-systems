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
#include <hip/hip_fp16.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_KERNELS_SHELL(func_name, T)                                                       \
  __global__ void func_name##_kernel_v1(__half* result, T* x) { *result = func_name(x); }          \
  __global__ void func_name##_kernel_v2(__half* result, Dummy x) { *result = func_name(x); }       \
  __global__ void func_name##_kernel_v3(Dummy* result, T x) { *result = func_name(x); }

NEGATIVE_KERNELS_SHELL(__int2half_rn, int)
NEGATIVE_KERNELS_SHELL(__int2half_rz, int)
NEGATIVE_KERNELS_SHELL(__int2half_rd, int)
NEGATIVE_KERNELS_SHELL(__int2half_ru, int)
NEGATIVE_KERNELS_SHELL(__uint2half_rn, unsigned int)
NEGATIVE_KERNELS_SHELL(__uint2half_rz, unsigned int)
NEGATIVE_KERNELS_SHELL(__uint2half_rd, unsigned int)
NEGATIVE_KERNELS_SHELL(__uint2half_ru, unsigned int)
NEGATIVE_KERNELS_SHELL(__short2half_rn, short)
NEGATIVE_KERNELS_SHELL(__short2half_rz, short)
NEGATIVE_KERNELS_SHELL(__short2half_rd, short)
NEGATIVE_KERNELS_SHELL(__short2half_ru, short)
NEGATIVE_KERNELS_SHELL(__short_as_half, short)
NEGATIVE_KERNELS_SHELL(__ushort2half_rn, unsigned short)
NEGATIVE_KERNELS_SHELL(__ushort2half_rz, unsigned short)
NEGATIVE_KERNELS_SHELL(__ushort2half_rd, unsigned short)
NEGATIVE_KERNELS_SHELL(__ushort2half_ru, unsigned short)
NEGATIVE_KERNELS_SHELL(__ushort_as_half, unsigned short)
NEGATIVE_KERNELS_SHELL(__ll2half_rn, long long)
NEGATIVE_KERNELS_SHELL(__ll2half_rz, long long)
NEGATIVE_KERNELS_SHELL(__ll2half_rd, long long)
NEGATIVE_KERNELS_SHELL(__ll2half_ru, long long)
NEGATIVE_KERNELS_SHELL(__ull2half_rn, unsigned long long)
NEGATIVE_KERNELS_SHELL(__ull2half_rz, unsigned long long)
NEGATIVE_KERNELS_SHELL(__ull2half_rd, unsigned long long)
NEGATIVE_KERNELS_SHELL(__ull2half_ru, unsigned long long)