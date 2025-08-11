/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>

#include "hip/hip_runtime.h"

#define WIDTH 256
#define HEIGHT 128

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

__global__ void vectoradd_att(float* __restrict__ a, const float* __restrict__ b,
                                const float* __restrict__ c, int width, int height)
  {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < width * height)
    a[i] = b[i] + c[i];
}

int main() {
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  hostA = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));
  hostB = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));
  hostC = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));

  // initialize the input data
  for (size_t i = 0; i < NUM; i++) {
    hostB[i] = static_cast<float>(i);
    hostC[i] = static_cast<float>(i) * 100.0f;
  }

  hipMalloc(reinterpret_cast<void**>(&deviceA), NUM * sizeof(float));
  hipMalloc(reinterpret_cast<void**>(&deviceB), NUM * sizeof(float));
  hipMalloc(reinterpret_cast<void**>(&deviceC), NUM * sizeof(float));

  hipMemcpy(deviceB, hostB, NUM * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(deviceC, hostC, NUM * sizeof(float), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(vectoradd_att,
                    dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                    0, 0, deviceA, deviceB, deviceC, WIDTH, HEIGHT);

  hipMemcpy(hostA, deviceA, NUM * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(deviceA);
  hipFree(deviceB);
  hipFree(deviceC);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
