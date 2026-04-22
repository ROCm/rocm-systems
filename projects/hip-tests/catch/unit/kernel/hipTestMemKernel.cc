/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>


#define LEN8 8 * 4
#define LEN9 9 * 4
#define LEN10 10 * 4
#define LEN11 11 * 4
#define LEN12 12 * 4

__global__ void MemCpy8(uint8_t* In, uint8_t* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memcpy(Out + tid * 8, In + tid * 8, 8);
}

__global__ void MemCpy9(uint8_t* In, uint8_t* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memcpy(Out + tid * 9, In + tid * 9, 9);
}

__global__ void MemCpy10(uint8_t* In, uint8_t* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memcpy(Out + tid * 10, In + tid * 10, 10);
}

__global__ void MemCpy11(uint8_t* In, uint8_t* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memcpy(Out + tid * 11, In + tid * 11, 11);
}

__global__ void MemCpy12(uint8_t* In, uint8_t* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memcpy(Out + tid * 12, In + tid * 12, 12);
}

__global__ void MemSet8(uint8_t* In) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memset(In + tid * 8, 1, 8);
}

__global__ void MemSet9(uint8_t* In) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memset(In + tid * 9, 1, 9);
}

__global__ void MemSet10(uint8_t* In) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memset(In + tid * 10, 1, 10);
}

__global__ void MemSet11(uint8_t* In) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memset(In + tid * 11, 1, 11);
}

__global__ void MemSet12(uint8_t* In) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  memset(In + tid * 12, 1, 12);
}
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check memcpy and memset via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/hipTestMemKernel.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

HIP_TEST_CASE(Unit_kernel_MemoryOperationsViaKernels) {
  uint8_t *A, *Ad, *B, *Bd, *C, *Cd;
  A = new uint8_t[LEN8];
  B = new uint8_t[LEN8];
  C = new uint8_t[LEN8];
  for (uint32_t i = 0; i < LEN8; i++) {
    A[i] = i;
    B[i] = 0;
    C[i] = 0;
  }
  HIP_CHECK(hipMalloc(&Ad, LEN8));
  HIP_CHECK(hipMalloc(&Bd, LEN8));
  HIP_CHECK(hipMalloc(&Cd, LEN8));
  HIP_CHECK(hipMemcpy(Ad, A, LEN8, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(MemCpy8, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
  hipLaunchKernelGGL(MemSet8, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
  HIP_CHECK(hipMemcpy(B, Bd, LEN8, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(C, Cd, LEN8, hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < LEN8; i++) {
    REQUIRE(A[i] == B[i]);
    REQUIRE(C[i] == 1);
  }

  delete[] A;
  delete[] B;
  delete[] C;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));

  SECTION("MemCpySet1") {
    A = new uint8_t[LEN9];
    B = new uint8_t[LEN9];
    C = new uint8_t[LEN9];
    for (uint32_t i = 0; i < LEN9; i++) {
      A[i] = i;
      B[i] = 0;
      C[i] = 0;
    }
    HIP_CHECK(hipMalloc(&Ad, LEN9));
    HIP_CHECK(hipMalloc(&Bd, LEN9));
    HIP_CHECK(hipMalloc(&Cd, LEN9));
    HIP_CHECK(hipMemcpy(Ad, A, LEN9, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(MemCpy9, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet9, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, LEN9, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, LEN9, hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < LEN9; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
  }

  SECTION("MemCpySet2") {
    A = new uint8_t[LEN10];
    B = new uint8_t[LEN10];
    C = new uint8_t[LEN10];
    for (uint32_t i = 0; i < LEN10; i++) {
      A[i] = i;
      B[i] = 0;
      C[i] = 0;
    }
    HIP_CHECK(hipMalloc(&Ad, LEN10));
    HIP_CHECK(hipMalloc(&Bd, LEN10));
    HIP_CHECK(hipMalloc(&Cd, LEN10));
    HIP_CHECK(hipMemcpy(Ad, A, LEN10, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(MemCpy10, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet10, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, LEN10, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, LEN10, hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < LEN10; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
  }

  SECTION("MemCpySet3") {
    A = new uint8_t[LEN11];
    B = new uint8_t[LEN11];
    C = new uint8_t[LEN11];
    for (uint32_t i = 0; i < LEN11; i++) {
      A[i] = i;
      B[i] = 0;
      C[i] = 0;
    }
    HIP_CHECK(hipMalloc(&Ad, LEN11));
    HIP_CHECK(hipMalloc(&Bd, LEN11));
    HIP_CHECK(hipMalloc(&Cd, LEN11));
    HIP_CHECK(hipMemcpy(Ad, A, LEN11, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(MemCpy11, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet11, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, LEN11, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, LEN11, hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < LEN11; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
  }

  SECTION("MemCpySet4") {
    A = new uint8_t[LEN12];
    B = new uint8_t[LEN12];
    C = new uint8_t[LEN12];
    for (uint32_t i = 0; i < LEN12; i++) {
      A[i] = i;
      B[i] = 0;
      C[i] = 0;
    }
    HIP_CHECK(hipMalloc(&Ad, LEN12));
    HIP_CHECK(hipMalloc(&Bd, LEN12));
    HIP_CHECK(hipMalloc(&Cd, LEN12));
    HIP_CHECK(hipMemcpy(Ad, A, LEN12, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(MemCpy12, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet12, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, LEN12, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, LEN12, hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < LEN12; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
  }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
