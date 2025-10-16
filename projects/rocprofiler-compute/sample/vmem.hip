/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



An example to explore global/generic instructions.
Written by Nicholas Curtis [AMD].
*/

#include "common.h"

typedef int __attribute__((address_space(0)))* generic_ptr;

__attribute__((noinline)) __device__ void generic_store(generic_ptr ptr, int zero) { *ptr = zero; }
__attribute__((noinline)) __device__ int generic_load(generic_ptr ptr) { return *ptr; }
__attribute__((noinline)) __device__ void generic_atomic(generic_ptr ptr, int zero) { atomicAdd((int*)ptr, zero); }

__global__ void global_write(int* ptr, int zero) {
  ptr[threadIdx.x] = zero;
}

__global__ void generic_write(int* ptr, int zero, int filter) {
  __shared__ int lds[1024];
  int* generic = (threadIdx.x < filter) ? &ptr[threadIdx.x] : &lds[threadIdx.x];
  generic_store((generic_ptr)generic, zero);
}

__global__ void global_read(int* ptr, int zero) {
  int x = ptr[threadIdx.x];
  if (x != zero) {
    ptr[threadIdx.x] = x + 1;
  }
}

__global__ void generic_read(int* ptr, int zero, int filter) {
  __shared__ int lds[1024];
  if (static_cast<int>(filter - 1) == zero) {
    lds[threadIdx.x] = 0; // initialize to zero to avoid conditional, but hide behind _another_ conditional
  }
  int* generic;
  if (static_cast<int>(threadIdx.x) > filter - 1) {
    generic = &ptr[threadIdx.x];
  } else {
    generic = &lds[threadIdx.x];
    abort();
  }
  int x = generic_load((generic_ptr)generic);
  if (x != zero) {
    ptr[threadIdx.x] = x + 1;
  }
}


__global__ void global_atomic(int* ptr, int zero) {
  atomicAdd(ptr, zero);
}

__global__ void generic_atomic(int* ptr, int filter, int zero) {
  __shared__ int lds[1024];
  int* generic = (threadIdx.x % 2 == filter) ? &ptr[threadIdx.x] : &lds[threadIdx.x];
  generic_atomic((generic_ptr)generic, zero);
}

int main() {
  int* ptr;
  hipCheck(hipMalloc(&ptr, sizeof(int)));
  hipCheck(hipMemset(ptr, 0, sizeof(int)));
  global_write<<<1,1>>>(ptr, 0);
  generic_write<<<1,1>>>(ptr, 0, 0);
  global_read<<<1,1>>>(ptr, 0);
  generic_read<<<1,1>>>(ptr, 0, 0);
  global_atomic<<<1,1>>>(ptr, 0);
  generic_atomic<<<1,1>>>(ptr, 0, 0);
  hipCheck(hipDeviceSynchronize());
  hipCheck(hipFree(ptr));
}
