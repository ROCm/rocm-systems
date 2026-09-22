/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kMakeHipComplex{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void make_hipComplex_kernel_v1(hipComplex* result, float* x, float y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v2(hipComplex* result, float x, float* y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v3(hipComplex* result, hipComplex x, float y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v4(hipComplex* result, float x, hipComplex y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v5(hipComplex* result, Dummy x, float y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v6(hipComplex* result, float x, Dummy y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v7(float* result, float x, float y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v8(hipDoubleComplex* result, float x, float y) {
    *result = make_hipComplex(x, y);
  }
  __global__ void make_hipComplex_kernel_v9(Dummy* result, float x, float y) {
    *result = make_hipComplex(x, y);
  }
  void make_hipComplex_v1(hipComplex* result, float* x, float y) { *result = make_hipComplex(x, y); }
  void make_hipComplex_v2(hipComplex* result, float x, float* y) { *result = make_hipComplex(x, y); }
  void make_hipComplex_v3(hipComplex* result, hipComplex x, float y) {
  *result = make_hipComplex(x, y);
  }
  void make_hipComplex_v4(hipComplex* result, float x, hipComplex y) {
    *result = make_hipComplex(x, y);
  }
  void make_hipComplex_v5(hipComplex* result, Dummy x, float y) { *result = make_hipComplex(x, y); }
  void make_hipComplex_v6(hipComplex* result, float x, Dummy y) { *result = make_hipComplex(x, y); }
  void make_hipComplex_v7(float* result, float x, float y) { *result = make_hipComplex(x, y); }
  void make_hipComplex_v8(hipDoubleComplex* result, float x, float y) {
    *result = make_hipComplex(x, y);
  }
  void make_hipComplex_v9(Dummy* result, float x, float y) { *result = make_hipComplex(x, y); }
)"};

static constexpr auto kMakeHipFloatComplex{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void make_hipFloatComplex_kernel_v1(hipFloatComplex* result, float* x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v2(hipFloatComplex* result, float x, float* y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v3(hipFloatComplex* result, hipFloatComplex x,
                                                 float y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v4(hipFloatComplex* result, float x,
                                                 hipFloatComplex y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v5(hipFloatComplex* result, Dummy x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v6(hipFloatComplex* result, float x, Dummy y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v7(float* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v8(hipDoubleComplex* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  __global__ void make_hipFloatComplex_kernel_v9(Dummy* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v1(hipFloatComplex* result, float* x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v2(hipFloatComplex* result, float x, float* y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v3(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v4(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v5(hipFloatComplex* result, Dummy x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v6(hipFloatComplex* result, float x, Dummy y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v7(float* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v8(hipDoubleComplex* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
  void make_hipFloatComplex_v9(Dummy* result, float x, float y) {
    *result = make_hipFloatComplex(x, y);
  }
)"};

static constexpr auto kMakeHipDoubleComplex{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void make_hipDoubleComplex_kernel_v1(hipDoubleComplex* result, double* x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v2(hipDoubleComplex* result, double x, double* y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v3(hipDoubleComplex* result, hipDoubleComplex x,
                                                  double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v4(hipDoubleComplex* result, double x,
                                                  hipDoubleComplex y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v5(hipDoubleComplex* result, Dummy x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v6(hipDoubleComplex* result, double x, Dummy y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v7(double* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v8(hipFloatComplex* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  __global__ void make_hipDoubleComplex_kernel_v9(Dummy* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v1(hipDoubleComplex* result, double* x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v2(hipDoubleComplex* result, double x, double* y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v3(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v4(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v5(hipDoubleComplex* result, Dummy x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v6(hipDoubleComplex* result, double x, Dummy y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v7(float* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v8(hipFloatComplex* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
  void make_hipDoubleComplex_v9(Dummy* result, double x, double y) {
    *result = make_hipDoubleComplex(x, y);
  }
)"};
