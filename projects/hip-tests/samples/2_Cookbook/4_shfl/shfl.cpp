/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#define WIDTH 4

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  float val = in[x];

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) out[i * width + j] = __shfl(val, j * width + i);
  }
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
  for (unsigned int j = 0; j < width; j++) {
    for (unsigned int i = 0; i < width; i++) {
      output[i * width + j] = input[j * width + i];
    }
  }
}

int main() {
  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;

  float* gpuMatrix;
  float* gpuTransposeMatrix;

  hipDeviceProp_t devProp;
  checkHipErrors(hipGetDeviceProperties(&devProp, 0));

  std::cout << "Device name " << devProp.name << std::endl;

  int i;
  int errors;

  Matrix = (float*)malloc(NUM * sizeof(float));
  TransposeMatrix = (float*)malloc(NUM * sizeof(float));
  cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  // allocate the memory on the device side
  checkHipErrors(hipMalloc((void**)&gpuMatrix, NUM * sizeof(float)));
  checkHipErrors(hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float)));

  // Memory transfer from host to device
  checkHipErrors(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

  // Lauching kernel from host
  hipLaunchKernelGGL(matrixTranspose, dim3(1), dim3(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y), 0,
                     0, gpuTransposeMatrix, gpuMatrix, WIDTH);

  // Memory transfer from device to host
  checkHipErrors(
      hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

  // verify the results
  errors = 0;
  double eps = 1.0E-6;
  for (i = 0; i < NUM; i++) {
    if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      printf("%d cpu: %f gpu  %f\n", i, cpuTransposeMatrix[i], TransposeMatrix[i]);
      errors++;
    }
  }
  if (errors != 0) {
    printf("FAILED: %d errors\n", errors);
  } else {
    printf("PASSED!\n");
  }

  // free the resources on device side
  checkHipErrors(hipFree(gpuMatrix));
  checkHipErrors(hipFree(gpuTransposeMatrix));

  // free the resources on host side
  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return errors;
}
