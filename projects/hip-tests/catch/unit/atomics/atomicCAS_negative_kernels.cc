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

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define ATOMIC_CAS_NEGATIVE_KERNEL(type_name)                                                      \
  __global__ void atomicCAS_v1(type_name* address, type_name* result) {                            \
    *result = atomicCAS(&address, 12, 13);                                                         \
  }                                                                                                \
  __global__ void atomicCAS_v2(type_name* address, type_name* result) {                            \
    *result = atomicCAS(address, address, 13);                                                     \
  }                                                                                                \
  __global__ void atomicCAS_v3(type_name* address, type_name* result) {                            \
    *result = atomicCAS(address, 12, address);                                                     \
  }                                                                                                \
  __global__ void atomicCAS_v4(Dummy* address, type_name* result) {                                \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v5(char* address, type_name* result) {                                 \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v6(short* address, type_name* result) {                                \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v7(long* address, type_name* result) {                                 \
    *result = atomicCAS(address, 12, 13);                                                          \
  }                                                                                                \
  __global__ void atomicCAS_v8(long long* address, type_name* result) {                            \
    *result = atomicCAS(address, 12, 13);                                                          \
  }

ATOMIC_CAS_NEGATIVE_KERNEL(int)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned int)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned long)
ATOMIC_CAS_NEGATIVE_KERNEL(unsigned long long)
ATOMIC_CAS_NEGATIVE_KERNEL(float)
ATOMIC_CAS_NEGATIVE_KERNEL(double)
