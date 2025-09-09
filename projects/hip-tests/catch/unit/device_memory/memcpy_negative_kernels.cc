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

/*void* memcpy(void* dst, const void* src, size_t size)*/
__global__ void memcpy_n1(int* dst, const int src, size_t size) { memcpy(dst, src, size); }
__global__ void memcpy_n2(int dst, const int* src, size_t size) { memcpy(dst, src, size); }
__global__ void memcpy_n3(int* dst, const int* src, size_t* size) { memcpy(dst, src, size); }
__global__ void memcpy_n4(int* dst, const int* src, Dummy size) { memcpy(dst, src, size); }