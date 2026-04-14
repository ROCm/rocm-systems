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
*/

#include "hip/hip_runtime.h"
#include <assert.h>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

// HIP kernel. Each thread takes care of one element of c
__global__ void vecCopy(double *a, double *b, double *c, int n, int stride) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
      c[id] = a[id];
}

// Duplicate of vecCopy kernel. Included for testing purposes
__global__ void vecCopy_2(double *a, double *b, double *c, int n, int stride) {
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
      c[id] = a[id];
}

void usage() {
  std::cout << "Usage: vcopy [OPTIONS]\n";
  std::cout << "Required:\n";
  std::cout << "  -n/--numThreads <value>   Set the num of threads\n";
  std::cout << "  -b/--blockSize <value>    Set the block size\n";
  std::cout << "Optional:\n";
  std::cout << "  -d/--dev <value>          Set the device ID [Default: 0]\n";
  std::cout << "  -i/--iter <value>         Set the num of iterations [Default: 1]\n";
  std::cout << "  -h/--help                 Display this help message\n";
  exit(1);
  return;
}

int main(int argc, char* argv[]) {
  // Size of vectors
  int n; //64 MB
  int blockSize, gridSize;

  // Launch multiple kernels
  bool multiKernel = false;

  // Host input vectors
  double *h_a;
  double *h_b;
  //Host output vector
  double *h_c;
  //Host output vector for verification
  double *h_verify_c;

  // Device input vectors
  double *d_a;
  double *d_b;
  // Device output vector
  double *d_c;

  int stride = 1;
  int devId = 0;
  int numIter = 1;

  hipError_t hip_status;

  for (int i = 0; i < argc; i++){
    std::string arg = argv[i];
    if ((arg == "--blockSize" || arg == "-b") && i+1 < argc)
      blockSize = std::atoi(argv[i+1]);

    else if ((arg == "--vec" || arg == "-n") && i+1 < argc)
      n = std::atoi(argv[i+1]);

    else if ((arg == "--device" || arg == "-d") && i+1 < argc)
      devId = std::atoi(argv[i+1]);

    else if ((arg == "--iter" || arg == "-i") && i+1 < argc)
      numIter = std::atoi(argv[i+1]);

    else if (arg == "--multikernel")
      multiKernel = true;

    else if (arg == "--help" || arg == "-h")
      usage();
  }

  if (blockSize == 0)
    usage();

  if (n == 0)
    usage();


  int numGpuDevices;
  HIP_ASSERT(hipGetDeviceCount(&numGpuDevices));
  if(devId >= numGpuDevices)
    devId = 0;
  HIP_ASSERT(hipSetDevice(devId));

  printf("vcopy testing on GCD %d\n", devId);

  assert(n > 0);
  assert(blockSize > 0);

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double)*stride;

  // Allocate memory for each vector on host
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);
  h_verify_c = (double*)malloc(bytes);

  printf("Finished allocating vectors on the CPU\n");

  // Allocate memory for each vector on GPU
  HIP_ASSERT(hipMalloc(&d_a, bytes));
  HIP_ASSERT(hipMalloc(&d_b, bytes));
  HIP_ASSERT(hipMalloc(&d_c, bytes));

  printf("Finished allocating vectors on the GPU\n");

  // Initialize vectors on host
  for(int i = 0; i < n; i++) {
      h_a[i] = i;
      h_b[i] = i;
  }

  // Copy host vectors to device
  HIP_ASSERT(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

  printf("Finished copying vectors to the GPU\n");

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n/blockSize);
  int tot_waves = (blockSize*gridSize)/64;
  float num_bytes_kb = ((sizeof(double))*n)/(1024);
  float num_bytes_wave = (1.0*num_bytes_kb)/(1.0*tot_waves);

  printf("sw thinks it moved %f KB per wave \n", (2.0*num_bytes_wave));
  printf("Total threads: %d, Grid Size: %d block Size:%d, Wavefronts:%d:\n", n, gridSize, blockSize, tot_waves);
  printf("Launching the  kernel on the GPU\n");

  // Execute the kernel
  for(int i = 0; i < numIter; i++){
    hipLaunchKernelGGL(vecCopy, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n, stride);
    hip_status = hipDeviceSynchronize();
    printf("Finished executing kernel\n");
    // Optionally, launch a second kernel. Only here for testing purposes
    if (multiKernel){
      hipLaunchKernelGGL(vecCopy_2, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n, stride);
      hip_status = hipDeviceSynchronize();
      printf("Finished executing kernel\n");
    }
  }

  // Copy array back to host
  HIP_ASSERT(hipMemcpy( h_c, d_c, bytes, hipMemcpyDeviceToHost));
  printf("Finished copying the output vector from the GPU to the CPU\n");

  // Compute for CPU
  for(int i=0; i<n; i++) {
    // h_verify_c[i*stride] = h_a[i*stride] + h_b[i*stride];
    h_verify_c[i*stride] = h_a[i*stride] ;
  }

  // Verfiy results
  for(int i = 0; i < n; i++) {
    if (abs(h_verify_c[i*stride] - h_c[i*stride]) > 1e-5)
      printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_c[i], d_c[i]);
  }
  //printf("Printing few elements from the output vector\n");
  for(int i = 0; i < 20; i++) {
    //printf("Output[%d]:%f\n",i, h_c[i]);
  }

  printf("Releasing GPU memory\n");

  // Release device memory
  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));
  HIP_ASSERT(hipFree(d_c));

  // Release host memory
  printf("Releasing CPU memory\n");
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
