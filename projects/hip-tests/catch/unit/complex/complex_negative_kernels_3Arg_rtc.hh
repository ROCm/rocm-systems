/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kComplexFma{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCfmaf_kernel_v1(hipComplex* result, hipFloatComplex* x, hipFloatComplex y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v2(hipComplex* result, hipFloatComplex x, hipFloatComplex* y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v3(hipComplex* result, hipFloatComplex x, hipFloatComplex y,
                                     hipFloatComplex* z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v4(hipComplex* result, float x, hipFloatComplex y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v5(hipComplex* result, hipFloatComplex x, float y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v6(hipComplex* result, hipFloatComplex x, hipFloatComplex y,
                                     float z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v7(hipComplex* result, hipDoubleComplex x, hipFloatComplex y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v8(hipComplex* result, hipFloatComplex x, hipDoubleComplex y,
                                     hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v9(hipComplex* result, hipFloatComplex x, hipFloatComplex y,
                                     hipDoubleComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v10(hipComplex* result, Dummy x, hipFloatComplex y,
                                      hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v11(hipComplex* result, hipFloatComplex x, Dummy y,
                                      hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v12(hipComplex* result, hipFloatComplex x, hipFloatComplex y,
                                      Dummy z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v13(float* result, hipFloatComplex x, hipFloatComplex y,
                                      hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v14(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y,
                                      hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfmaf_kernel_v15(Dummy* result, hipFloatComplex x, hipFloatComplex y,
                                      hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  __global__ void hipCfma_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v3(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                                    hipDoubleComplex* z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v4(hipDoubleComplex* result, double x, hipDoubleComplex y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x, double y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                                    double z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v7(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v9(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                                    hipFloatComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v10(hipDoubleComplex* result, Dummy x, hipDoubleComplex y,
                                    hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v11(hipDoubleComplex* result, hipDoubleComplex x, Dummy y,
                                     hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v12(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                                     Dummy z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v13(double* result, hipDoubleComplex x, hipDoubleComplex y,
                                     hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v14(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                                     hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  __global__ void hipCfma_kernel_v15(Dummy* result, hipDoubleComplex x, hipDoubleComplex y,
                                     hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfmaf_v1(hipComplex* result, hipFloatComplex* x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v2(hipComplex* result, hipFloatComplex x, hipFloatComplex* y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v3(hipComplex* result, hipFloatComplex x, hipFloatComplex y, hipFloatComplex* z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v4(hipComplex* result, float x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v5(hipComplex* result, hipFloatComplex x, float y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v6(hipComplex* result, hipFloatComplex x, hipFloatComplex y, float z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v7(hipComplex* result, hipDoubleComplex x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v8(hipComplex* result, hipFloatComplex x, hipDoubleComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v9(hipComplex* result, hipFloatComplex x, hipFloatComplex y, hipDoubleComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v10(hipComplex* result, Dummy x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v11(hipComplex* result, hipFloatComplex x, Dummy y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v12(hipComplex* result, hipFloatComplex x, hipFloatComplex y, Dummy z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v13(float* result, hipFloatComplex x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v14(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y,
                    hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfmaf_v15(Dummy* result, hipFloatComplex x, hipFloatComplex y, hipFloatComplex z) {
    *result = hipCfmaf(x, y, z);
  }
  void hipCfma_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y,
                  hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y,
                  hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v3(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                  hipDoubleComplex* z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v4(hipDoubleComplex* result, double x, hipDoubleComplex y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v5(hipDoubleComplex* result, hipDoubleComplex x, double y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v6(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y, double z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v7(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y,
                  hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v8(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y,
                  hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v9(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                  hipFloatComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v10(hipDoubleComplex* result, Dummy x, hipDoubleComplex y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v11(hipDoubleComplex* result, hipDoubleComplex x, Dummy y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v12(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex y, Dummy z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v13(double* result, hipDoubleComplex x, hipDoubleComplex y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v14(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y,
                   hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
  void hipCfma_v15(Dummy* result, hipDoubleComplex x, hipDoubleComplex y, hipDoubleComplex z) {
    *result = hipCfma(x, y, z);
  }
)"};
