/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "hip_helper.h"

#define LEN 64
#define SIZE LEN << 2

#define fileName "vcpy_kernel.code"
#define kernel_name "hello_world"

int main() {
  float *A, *B;
  hipDeviceptr_t Ad, Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  hipInit(0);
  hipDevice_t device;
  hipCtx_t context;
  checkHipErrors(hipDeviceGet(&device, 0));
  checkHipErrors(hipCtxCreate(&context, 0, device));

  checkHipErrors(hipMalloc((void**)&Ad, SIZE));
  checkHipErrors(hipMalloc((void**)&Bd, SIZE));

  checkHipErrors(hipMemcpyHtoD(Ad, A, SIZE));
  checkHipErrors(hipMemcpyHtoD(Bd, B, SIZE));

  hipModule_t Module;
  hipFunction_t Function;
  checkHipErrors(hipModuleLoad(&Module, fileName));
  checkHipErrors(hipModuleGetFunction(&Function, Module, kernel_name));

  void* args[2] = {&Ad, &Bd};

  checkHipErrors(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, args, nullptr));

  checkHipErrors(hipMemcpyDtoH(B, Bd, SIZE));
  int mismatchCount = 0;
  for (uint32_t i = 0; i < LEN; i++) {
    if (A[i] != B[i]) {
      mismatchCount++;
      std::cout << "error: mismatch " << A[i] << " != " << B[i] << std::endl;
    }
  }

  if (mismatchCount == 0) {
    std::cout << "PASSED!\n";
  } else {
    std::cout << "FAILED!\n";
  };

  checkHipErrors(hipFree(Ad));
  checkHipErrors(hipFree(Bd));
  delete[] A;
  delete[] B;
  checkHipErrors(hipCtxDestroy(context));
  return 0;
}
