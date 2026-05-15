/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "hip_helper.h"

__global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
#ifdef __HIP_PLATFORM_AMD__
    C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVIDIA__ or other path */
    C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
  }
}


int main(int argc, char* argv[]) {
  uint32_t *A_d, *C_d;
  uint32_t *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(uint32_t);

#ifdef __HIP_ENABLE_PCH
  // Verify hip_pch.o
  const char* pch = nullptr;
  unsigned int size = 0;
  __hipGetPCH(&pch, &size);
  printf("pch size: %u\n", size);
  if (size == 0) {
    printf("__hipGetPCH failed!\n");
    return -1;
  } else {
    printf("__hipGetPCH succeeded!\n");
  }
#endif

  int deviceId;
  checkHipErrors(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  checkHipErrors(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);


  printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  A_h = (uint32_t*)malloc(Nbytes);
  checkHipErrors(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (uint32_t*)malloc(Nbytes);
  checkHipErrors(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  checkHipErrors(hipMalloc(&A_d, Nbytes));
  checkHipErrors(hipMalloc(&C_d, Nbytes));

  printf("info: copy Host2Device\n");
  checkHipErrors(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

  printf("info: launch 'bit_extract_kernel' \n");
  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;
  hipLaunchKernelGGL(bit_extract_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

  printf("info: copy Device2Host\n");
  checkHipErrors(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  printf("info: check result\n");
  for (size_t i = 0; i < N; i++) {
    unsigned Agold = ((A_h[i] & 0xf00) >> 8);
    if (C_h[i] != Agold) {
      fprintf(stderr, "mismatch detected.\n");
      printf("%zu: %08x =? %08x (Ain=%08x)\n", i, C_h[i], Agold, A_h[i]);
      checkHipErrors(hipErrorUnknown);
    }
  }

  checkHipErrors(hipFree(A_d));
  checkHipErrors(hipFree(C_d));
  free(A_h);
  free(C_h);

  printf("PASSED!\n");
}
