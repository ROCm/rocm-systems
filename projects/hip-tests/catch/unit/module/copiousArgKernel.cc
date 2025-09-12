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

#include "hip/hip_runtime.h"

extern "C" __global__ void kernelMultipleArgsSaxpy(int a1, int a2, int* x1, int b1, int b2, int* x2,
                                                   int c1, int c2, int* x3, int d1, int d2, int* x4,
                                                   int e1, int e2, int* x5, int f1, int f2,
                                                   int* x6) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  x1[id] = a1 * x1[id] + a2;
  x2[id] = b1 * x2[id] + b2;
  x3[id] = c1 * x3[id] + c2;
  x4[id] = d1 * x4[id] + d2;
  x5[id] = e1 * x5[id] + e2;
  x6[id] = f1 * x6[id] + f2;
}
