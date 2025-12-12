/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#define LENGTH 4

#define SIZE (LENGTH * LENGTH)

#define THREADS_PER_BLOCK 1
#define BLOCKS_PER_GRID LENGTH

// CPU function - basically scan each row and save the output in array
void matrixRowSum(int* input, int* output, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      output[i] += input[i * width + j];
    }
  }
}

// Device (kernel) function
__global__ void gpuMatrixRowSum(int* input, int* output, int width) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < width; i++) {
    output[index] += input[index * width + i];
  }
}

int main() {
  int* Matrix;
  int* sumMatrix;
  int* cpuSumMatrix;

  int* gpuMatrix;
  int* gpuSumMatrix;

  hipDeviceProp_t devProp;
  checkHipErrors(hipGetDeviceProperties(&devProp, 0));

  std::cout << "Device name " << devProp.name << std::endl;

  Matrix = (int*)malloc(sizeof(int) * SIZE);
  sumMatrix = (int*)malloc(sizeof(int) * LENGTH);
  cpuSumMatrix = (int*)malloc(sizeof(int) * LENGTH);

  for (int i = 0; i < SIZE; i++) {
    Matrix[i] = i * 2;
  }

  for (int i = 0; i < LENGTH; i++) {
    cpuSumMatrix[i] = 0;
  }

  // Allocated Device Memory
  checkHipErrors(hipMalloc((void**)&gpuMatrix, SIZE * sizeof(int)));
  checkHipErrors(hipMalloc((void**)&gpuSumMatrix, LENGTH * sizeof(int)));

  // Memory Copy to Device
  checkHipErrors(hipMemcpy(gpuMatrix, Matrix, SIZE * sizeof(int), hipMemcpyHostToDevice));
  checkHipErrors(
      hipMemcpy(gpuSumMatrix, cpuSumMatrix, LENGTH * sizeof(float), hipMemcpyHostToDevice));

  // Launch device kernels
  hipLaunchKernelGGL(gpuMatrixRowSum, dim3(BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK), 0, 0,
                     gpuMatrix, gpuSumMatrix, LENGTH);

  // Memory copy back to device
  checkHipErrors(hipMemcpy(sumMatrix, gpuSumMatrix, LENGTH * sizeof(int), hipMemcpyDeviceToHost));

  // Cpu implementation
  matrixRowSum(Matrix, cpuSumMatrix, LENGTH);


  // verify the results
  int errors = 0;
  for (int i = 0; i < LENGTH; i++) {
    if (sumMatrix[i] != cpuSumMatrix[i]) {
      printf("%d - cpu: %d gpu: %d\n", i, sumMatrix[i], cpuSumMatrix[i]);
      errors++;
    }
  }

  if (errors == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED with %d errors\n", errors);
  }

  // GPU Free
  checkHipErrors(hipFree(gpuMatrix));
  checkHipErrors(hipFree(gpuSumMatrix));

  // CPU Free
  free(Matrix);
  free(sumMatrix);
  free(cpuSumMatrix);

  return errors;
}
