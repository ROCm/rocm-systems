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

/*int atomicExch(int*, int)*/
__global__ void atomicExch_int_n1(int* p, int v) { atomicExch(p, p); }
__global__ void atomicExch_int_n2(int* p, int v) { atomicExch(&p, v); }
__global__ void atomicExch_int_n3(char* p, int v) { atomicExch(p, v); }
__global__ void atomicExch_int_n4(short* p, int v) { atomicExch(p, v); }
__global__ void atomicExch_int_n5(long* p, int v) { atomicExch(p, v); }
__global__ void atomicExch_int_n6(long long* p, int v) { atomicExch(p, v); }
__global__ void atomicExch_int_n7(Dummy* p, int v) { atomicExch(p, v); }
__global__ void atomicExch_int_n8(int* p, Dummy v) { atomicExch(p, v); }

/*unsigned int atomicExch(unsigned int*, unsigned int)*/
__global__ void atomicExch_unsigned_int_n1(unsigned int* p, unsigned int v) { atomicExch(p, p); }
__global__ void atomicExch_unsigned_int_n2(unsigned int* p, unsigned int v) { atomicExch(&p, v); }
__global__ void atomicExch_unsigned_int_n3(char* p, unsigned int v) { atomicExch(p, v); }
__global__ void atomicExch_unsigned_int_n4(short* p, unsigned int v) { atomicExch(p, v); }
__global__ void atomicExch_unsigned_int_n5(long* p, unsigned int v) { atomicExch(p, v); }
__global__ void atomicExch_unsigned_int_n6(long long* p, unsigned int v) { atomicExch(p, v); }
__global__ void atomicExch_unsigned_int_n7(Dummy* p, unsigned int v) { atomicExch(p, v); }
__global__ void atomicExch_unsigned_int_n8(unsigned int* p, Dummy v) { atomicExch(p, v); }

// /*unsigned long long atomicExch(unsigned long long*, unsigned long long)*/
__global__ void atomicExch_unsigned_long_long_n1(unsigned long long* p, unsigned long long v) {
  atomicExch(p, p);
}
__global__ void atomicExch_unsigned_long_long_n2(unsigned long long* p, unsigned long long v) {
  atomicExch(&p, v);
}
__global__ void atomicExch_unsigned_long_long_n3(char* p, unsigned long long v) {
  atomicExch(p, v);
}
__global__ void atomicExch_unsigned_long_long_n4(short* p, unsigned long long v) {
  atomicExch(p, v);
}
__global__ void atomicExch_unsigned_long_long_n5(long* p, unsigned long long v) {
  atomicExch(p, v);
}
__global__ void atomicExch_unsigned_long_long_n6(long long* p, unsigned long long v) {
  atomicExch(p, v);
}
__global__ void atomicExch_unsigned_long_long_n7(Dummy* p, unsigned long long v) {
  atomicExch(p, v);
}
__global__ void atomicExch_unsigned_long_long_n8(unsigned long long* p, Dummy v) {
  atomicExch(p, v);
}

// /*float atomicExch(float*, float)*/
__global__ void atomicExch_float_n1(float* p, float v) { atomicExch(p, p); }
__global__ void atomicExch_float_n2(float* p, float v) { atomicExch(&p, v); }
__global__ void atomicExch_float_n3(char* p, float v) { atomicExch(p, v); }
__global__ void atomicExch_float_n4(short* p, float v) { atomicExch(p, v); }
__global__ void atomicExch_float_n5(long* p, float v) { atomicExch(p, v); }
__global__ void atomicExch_float_n6(long long* p, float v) { atomicExch(p, v); }
__global__ void atomicExch_float_n7(Dummy* p, float v) { atomicExch(p, v); }
__global__ void atomicExch_float_n8(float* p, Dummy v) { atomicExch(p, v); }

// /*double atomicExch(double*, double)*/
__global__ void atomicExch_double_n1(double* p, double v) { atomicExch(p, p); }
__global__ void atomicExch_double_n2(double* p, double v) { atomicExch(&p, v); }
__global__ void atomicExch_double_n3(char* p, double v) { atomicExch(p, v); }
__global__ void atomicExch_double_n4(short* p, double v) { atomicExch(p, v); }
__global__ void atomicExch_double_n5(long* p, double v) { atomicExch(p, v); }
__global__ void atomicExch_double_n6(long long* p, double v) { atomicExch(p, v); }
__global__ void atomicExch_double_n7(Dummy* p, double v) { atomicExch(p, v); }
__global__ void atomicExch_double_n8(double* p, Dummy v) { atomicExch(p, v); }