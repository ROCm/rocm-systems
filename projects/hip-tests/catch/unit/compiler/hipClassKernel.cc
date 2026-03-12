/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipClassKernel.h"

__global__ void ovrdClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testOvrD tobj1;
  result_ecd[tid] = (tobj1.ovrdFunc1() == 30);
}

__global__ void ovldClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testFuncOvld tfo1;
  result_ecd[tid] = (tfo1.func1(10) == 20) && (tfo1.func1(10, 10) == 30);
}

TEST_CASE(Unit_hipClassKernel_Overload_Override) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(ovrdClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);

  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(ovldClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}

// check for friend
__global__ void friendClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testFrndB tfb1;
  result_ecd[tid] = (tfb1.showA() == 10);
}

TEST_CASE(Unit_hipClassKernel_Friend) {
  bool* result_ecd;
  result_ecd = AllocateDeviceMemory();
  hipLaunchKernelGGL(friendClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);
  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipFree(result_ecd));
}

// check sizeof empty class is 1
__global__ void emptyClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testClassEmpty ob1, ob2;
  result_ecd[tid] = (sizeof(testClassEmpty) == 1) && (&ob1 != &ob2);
}

TEST_CASE(Unit_hipClassKernel_Empty) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(emptyClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}

// tests for classes >8 bytes
__global__ void sizeClassBKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result_ecd[tid] = (sizeof(testSizeB) == 12) && (sizeof(testSizeC) == 16) &&
                    (sizeof(testSizeP1) == 6) && (sizeof(testSizeP2) == 13) &&
                    (sizeof(testSizeP3) == 8);
}

TEST_CASE(Unit_hipClassKernel_BSize) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(sizeClassBKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}

__global__ void sizeClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result_ecd[tid] = (sizeof(testSizeA) == 16) && (sizeof(testSizeDerived) == 24) &&
                    (sizeof(testSizeDerived2) == 20);
}

TEST_CASE(Unit_hipClassKernel_Size) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  hipLaunchKernelGGL(sizeClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}

__global__ void sizeVirtualClassKernel(bool* result_ecd, refStructSizes structSizes) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result_ecd[tid] = (structSizes.sizeOftestSizeDV == sizeof(testSizeDV)) &&
                    (structSizes.sizeOftestSizeDerivedDV == sizeof(testSizeDerivedDV)) &&
                    (structSizes.sizeOftestSizeVirtDer = sizeof(testSizeVirtDer)) &&
                    (structSizes.sizeOftestSizeVirtDerPack = sizeof(testSizeVirtDerPack)) &&
                    (structSizes.sizeOftestSizeDerMulti = sizeof(testSizeDerMulti));
}

TEST_CASE(Unit_hipClassKernel_Virtual) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  struct refStructSizes structSizes;
  structSizes.sizeOftestSizeDV = sizeof(testSizeDV);
  structSizes.sizeOftestSizeDerivedDV = sizeof(testSizeDerivedDV);
  structSizes.sizeOftestSizeVirtDer = sizeof(testSizeVirtDer);
  structSizes.sizeOftestSizeVirtDerPack = sizeof(testSizeVirtDerPack);
  structSizes.sizeOftestSizeDerMulti = sizeof(testSizeDerMulti);

  hipLaunchKernelGGL(sizeVirtualClassKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0,
                     result_ecd, structSizes);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}

// check pass by value
__global__ void passByValueKernel(testPassByValue obj, bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result_ecd[tid] = (obj.exI == 10) && (obj.exC == 'C');
}

TEST_CASE(Unit_hipClassKernel_Value) {
  bool *result_ecd, *result_ech;
  result_ech = AllocateHostMemory();
  result_ecd = AllocateDeviceMemory();

  testPassByValue exObj;
  exObj.exI = 10;
  exObj.exC = 'C';
  hipLaunchKernelGGL(passByValueKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), 0, 0, exObj,
                     result_ecd);

  VerifyResult(result_ech, result_ecd);
  FreeMem(result_ech, result_ecd);
}
