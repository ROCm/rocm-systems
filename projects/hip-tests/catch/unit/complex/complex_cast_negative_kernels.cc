/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_complex.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

__global__ void hipComplexDoubleToFloat_kernel_v1(hipFloatComplex* result, hipDoubleComplex* x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v2(hipFloatComplex* result, hipFloatComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v3(hipFloatComplex* result, double x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v4(hipFloatComplex* result, Dummy x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v5(float* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
__global__ void hipComplexDoubleToFloat_kernel_v7(Dummy* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v1(hipFloatComplex* result, hipDoubleComplex* x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v2(hipFloatComplex* result, hipFloatComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v3(hipFloatComplex* result, double x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v4(hipFloatComplex* result, Dummy x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v5(float* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v6(hipDoubleComplex* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}
void hipComplexDoubleToFloat_v7(Dummy* result, hipDoubleComplex x) {
  *result = hipComplexDoubleToFloat(x);
}

__global__ void hipComplexFloatToDouble_kernel_v1(hipDoubleComplex* result, hipFloatComplex* x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v3(hipDoubleComplex* result, float x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v4(hipDoubleComplex* result, Dummy x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v5(double* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v6(hipFloatComplex* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}
__global__ void hipComplexFloatToDouble_kernel_v7(Dummy* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v1(hipDoubleComplex* result, hipFloatComplex* x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v2(hipDoubleComplex* result, hipDoubleComplex x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v3(hipDoubleComplex* result, float x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v4(hipDoubleComplex* result, Dummy x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v5(double* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v6(hipFloatComplex* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}
void hipComplexFloatToDouble_v7(Dummy* result, hipFloatComplex x) {
  *result = hipComplexFloatToDouble(x);
}