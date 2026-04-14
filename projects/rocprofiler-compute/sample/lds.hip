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



An example to explore LDS bandwidth and bank conflicts, written by Nicholas Curtis [AMD].
*/


#include "common.h"

constexpr unsigned max_threads = 256;
constexpr unsigned nbanks = 32;

__global__ void load(int* out, int flag) {
  __shared__ int array[max_threads];
  int index = threadIdx.x;
  // fake a store to the LDS array to avoid unwanted behavior
  if (flag)
    array[max_threads - index] = index;
  __syncthreads();
  int x = array[index];
  if (x == int(-1234567))
    out[threadIdx.x] = x;
}

__global__ void conflicts(int* out, int flag) {
  constexpr unsigned nelements = nbanks * max_threads;
  __shared__ int array[nelements];
  // each thread reads from the same bank
  int index = threadIdx.x * nbanks;
  // fake a store to the LDS array to avoid unwanted behavior
  if (flag)
    array[max_threads - index] = index;
  __syncthreads();
  int x = array[index];
  if (x == int(-1234567))
    out[threadIdx.x] = x;
}

void bandwidth_demo(int N) {
  for (int i = 1; i <= N; ++i)
    load<<<1,i>>>(nullptr, 0);
  hipCheck(hipDeviceSynchronize());
}

void conflicts_demo(int N) {
  for (int i = 1; i <= N; ++i)
    conflicts<<<1,i>>>(nullptr, 0);
  hipCheck(hipDeviceSynchronize());
}

int main() {
  bandwidth_demo(max_threads);
  conflicts_demo(max_threads);
}
