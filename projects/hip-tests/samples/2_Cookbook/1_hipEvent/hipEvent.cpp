/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  out[y * width + x] = in[x * width + y];
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

  hipEvent_t start, stop;
  checkHipErrors(hipEventCreate(&start));
  checkHipErrors(hipEventCreate(&stop));
  float eventMs = 1.0f;

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

  // Record the start event
  checkHipErrors(hipEventRecord(start, NULL));

  // Memory transfer from host to device
  checkHipErrors(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

  // Record the stop event
  checkHipErrors(hipEventRecord(stop, NULL));
  checkHipErrors(hipEventSynchronize(stop));

  checkHipErrors(hipEventElapsedTime(&eventMs, start, stop));

  printf("hipMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

  // Record the start event
  checkHipErrors(hipEventRecord(start, NULL));

  // Lauching kernel from host
  hipLaunchKernelGGL(
      matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
      dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix, gpuMatrix, WIDTH);

  // Record the stop event
  checkHipErrors(hipEventRecord(stop, NULL));
  checkHipErrors(hipEventSynchronize(stop));

  checkHipErrors(hipEventElapsedTime(&eventMs, start, stop));

  printf("kernel Execution time             = %6.3fms\n", eventMs);

  // Record the start event
  checkHipErrors(hipEventRecord(start, NULL));

  // Memory transfer from device to host
  checkHipErrors(
      hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

  // Record the stop event
  checkHipErrors(hipEventRecord(stop, NULL));
  checkHipErrors(hipEventSynchronize(stop));

  checkHipErrors(hipEventElapsedTime(&eventMs, start, stop));

  printf("hipMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

  checkHipErrors(hipEventDestroy(start));
  checkHipErrors(hipEventDestroy(stop));

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

  // verify the results
  errors = 0;
  double eps = 1.0E-6;
  for (i = 0; i < NUM; i++) {
    if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
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
