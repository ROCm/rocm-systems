/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#define LEN 1024
#define SIZE (LEN << 2)

__global__ static void cpy(uint32_t* Out, uint32_t* In) {
  int tx = threadIdx.x;
  memcpy(Out + tx, In + tx, sizeof(uint32_t));
}

__global__ static void set(uint32_t* ptr, uint8_t val) {
  int tx = threadIdx.x;
  memset(ptr + tx, val, sizeof(uint32_t));
}

TEST_CASE(Unit_ToAndFroMemCpyToDevice) {
  uint32_t *A, *Ad, *B, *Bd;
  A = new uint32_t[LEN];
  B = new uint32_t[LEN];
  for (int i = 0; i < LEN; i++) {
    A[i] = i;
    B[i] = 0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(cpy, dim3(1), dim3(LEN), 0, 0, Bd, Ad);

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  for (int i = LEN - 16; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  hipLaunchKernelGGL(set, dim3(1), dim3(LEN), 0, 0, Bd, 0x1);

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  for (int i = LEN - 16; i < LEN; i++) {
    REQUIRE(0x01010101 == B[i]);
  }

  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
}
