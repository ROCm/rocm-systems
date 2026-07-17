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
#include"hip/hip_runtime.h"

__device__ int data = 0;
extern "C" __global__ void first(int* res, int ticks_for_100us) {
  uint64_t start = wall_clock64();
  while (__hip_atomic_load(&data, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 1) {
    __builtin_amdgcn_s_sleep(10);
    if (wall_clock64() - start >= ticks_for_100us) {
      *res = 2;
      return;
    }
  }
  *res = 1;
}
extern "C" __global__ void second() {
  atomicAdd(&data, 1);
}