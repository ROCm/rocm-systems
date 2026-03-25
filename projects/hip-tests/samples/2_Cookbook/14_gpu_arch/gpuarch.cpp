/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"
#include <iostream>
#include "hip_helper.h"

#define THREADS_PER_BLOCK 64
#define BLOCKS_PER_GRID 4
#define SIZE (BLOCKS_PER_GRID * THREADS_PER_BLOCK)
#define NOT_SUPPORTED -99  // dummy number indicates unsupported operation

// Using __gfx*__ macro one can have GPU architecture specific code flow
// For example: If below kernel runs on gfx908 it will increment 'in' by 'value' and store into
// 'out'
//              but it will update with "NOT_SUPPORTED" for any other gfx archs.
__global__ void incrementKernel(int32_t* in, int32_t* out, int32_t value, size_t buffSize) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < buffSize) {
#if defined(__gfx908__)
    out[index] = in[index] + value;
#else
    out[index] = NOT_SUPPORTED;
#endif
  }
}

int main() {
  int32_t incrementValue = 10;
  // Device pointers
  int32_t* dInput = nullptr;
  int32_t* dOutput = nullptr;

  size_t NBytes = SIZE * sizeof(int32_t);
  // Host pointers
  int32_t* hInput = static_cast<int32_t*>(malloc(NBytes));
  int32_t* hOutput = static_cast<int32_t*>(malloc(NBytes));

  checkHipErrors(hipMalloc(&dInput, NBytes));
  checkHipErrors(hipMalloc(&dOutput, NBytes));

  // Initialize host input/output buffers
  for (int i = 0; i < SIZE; ++i) {
    hInput[i] = i;
    hOutput[i] = 0;
  }

  // Initialize device input buffer
  checkHipErrors(hipMemcpy(dInput, hInput, NBytes, hipMemcpyHostToDevice));

  // Launch kernel
  hipLaunchKernelGGL(incrementKernel, dim3(BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK), 0, 0, dInput,
                     dOutput, incrementValue, SIZE);

  // Copy result back to host buffer
  checkHipErrors(hipMemcpy(hOutput, dOutput, NBytes, hipMemcpyDeviceToHost));

  bool flag = true;
  // verify data
  for (int i = 0; i < SIZE; ++i) {
    if (hOutput[i] != NOT_SUPPORTED && hOutput[i] != (hInput[i] + incrementValue)) {
      std::cout << "Error : Data mismatch found";
      exit(0);
    } else if (hOutput[i] == NOT_SUPPORTED) {
      flag &= false;
    }
  }
  if (flag == false) {
    std::cout << "Error: Kernel is supported for gfx908 architecture\n";
  } else {
    std::cout << "success\n";
  }
  free(hInput);
  free(hOutput);
  checkHipErrors(hipFree(dInput));
  checkHipErrors(hipFree(dOutput));
  return 0;
}
