/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.

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

// OpenMP program to print Hello World
// using C language is supported by HIP

// OpenMP header
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

// HIP header
#include <hip/hip_runtime.h>
#include "utils/test_helper.h"

#define NUM_THREADS 16

__global__ void hip_helloworld(unsigned omp_id, int* A_d) {
  // Note: the printf command will only work if printf is enabled in your build.
  // printf("Hello World... from HIP thread = %u\n", omp_id);

  A_d[omp_id] = omp_id;
}

int main(int argc, char* argv[]) {
  int *A_h, *A_d;
  size_t Nbytes = NUM_THREADS * sizeof(int);

  hipDeviceProp_t props;
  HIP_RC(hipGetDeviceProperties(&props, 0 /*deviceID*/));
  // printf("info: running on device %s\n", props.name);

  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  HIP_RC(hipMalloc(&A_d, Nbytes));
  for (int i = 0; i < NUM_THREADS; i++) {
    A_h[i] = 0;
  }
  HIP_RC(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

// Beginning of parallel region
#pragma omp parallel num_threads(NUM_THREADS)
  {
    // fprintf(stderr, "Hello World... from OMP thread = %d\n",
    //  omp_get_thread_num());

    HIP_KL(hipLaunchKernelGGL(hip_helloworld, dim3(1), dim3(1), 0, 0, omp_get_thread_num(), A_d));
  }
  // Ending of parallel region

  HIP_RC(hipStreamSynchronize(0));
  HIP_RC(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  // printf("Device Results:\n");
  for (int i = 0; i < NUM_THREADS; i++) {
    // printf("  A_d[%d] = %d\n", i, A_h[i]);
  }

  printf("PASSED!\n");

  free(A_h);
  HIP_RC(hipFree(A_d));
  return 0;
}
