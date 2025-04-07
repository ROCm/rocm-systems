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
#include <mpi.h>

#include <math.h>
#include <stdio.h>

#include <iostream>

#include "hip/hip_runtime.h"
#include "utils/test_helper.h"

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

int main(int argc, char* argv[]) {
  int N = 1 << 20;
  float* x = new float[N];
  float* y = new float[N];
  float* d_x;
  float* d_y;

  int myId;
  int devId;
  int numRank;
  int deviceCount;

  // init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &numRank);
  HIP_RC(hipGetDeviceCount(&deviceCount));

  std::cout << "device count and rank is" << deviceCount << ": " << numRank << std::endl;

  // set the device ID to the rank ID mod deviceCount (in this case 4 since
  // there are 4 devices on a node)
  devId = myId % deviceCount;
  // set the device ID
  HIP_RC(hipSetDevice(devId));

  printf("Rank Id: %d | Device Id : %d | Num Devices: %d\n", myId, devId, deviceCount);
  fflush(stdout);

  //   Allocate Unified Memory -- accessible from CPU or GPU
  HIP_RC(hipMallocManaged(&d_x, N * sizeof(float)));
  HIP_RC(hipMallocManaged(&d_y, N * sizeof(float)));

  //   initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  HIP_RC(hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_RC(hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice));

  //   Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  HIP_KL(hipLaunchKernelGGL(add, numBlocks, blockSize, 0, 0, N, d_x, d_y));

  //   Wait for GPU to finish before accessing on host
  HIP_RC(hipDeviceSynchronize());

  HIP_RC(hipMemcpy(x, d_x, N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_RC(hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

  //   Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i] - 3.0f));
  printf("Max error: %f\n", maxError);

  //   Free memory
  HIP_RC(hipFree(d_x));
  HIP_RC(hipFree(d_y));

  delete[] x;
  delete[] y;

  MPI_Finalize();
  return 0;
}
