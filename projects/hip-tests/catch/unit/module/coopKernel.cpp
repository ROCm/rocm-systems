
/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include <hip/hip_cooperative_groups.h>

using namespace cooperative_groups;
extern "C" {
__global__ void cooperativeKernelEx(int* output, int totalThreads) {
  grid_group grid = this_grid();
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < totalThreads) {
    output[tid] = tid * 3;
  }
  grid.sync();
  if (tid == 0) {
    output[0] = 2222;
  }
}

/*
 * Kernel which doesn't use cooperative groups and without any arguments
 */
__global__ void emptyKernel() {}

/*
 * Kernel which doesn't use cooperative groups and takes an argument
 * and updates the value with 100
 */
__global__ void argKernel(int* val) { *val = 100; }

/*
 * Kernel which uses cooperative groups and without any arguments
 */
__global__ void coopEmptykernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();
}

/*
 * Kernel which uses cooperative groups and takes arguments
 * and performs below operations
 * 1) All the threads in the block fills the element in the arr array
 *    based on the blockIdx.x
 * 2) Wait for all the blocks completes it's operations
 * 3) Fill each element in the output array with sum of elements in arr
 */
__global__ void coopFillArrayKernel(int* arr, int* output, int N) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  if (blockIdx.x == 0)
    arr[0] = 10;
  else if (blockIdx.x == 1)
    arr[1] = 20;
  else if (blockIdx.x == 2)
    arr[2] = 30;
  else if (blockIdx.x == 3)
    arr[3] = 40;
  else if (blockIdx.x == 4)
    arr[4] = 50;
  else if (blockIdx.x == 5)
    arr[5] = 60;
  else if (blockIdx.x == 6)
    arr[6] = 70;
  else if (blockIdx.x == 7)
    arr[7] = 80;
  else if (blockIdx.x == 8)
    arr[8] = 90;
  else if (blockIdx.x == 9)
    arr[9] = 100;

  grid.sync();

  for (int i = 0; i < N; i++) {
    output[blockIdx.x] = output[blockIdx.x] + arr[i];
  }
}
}
