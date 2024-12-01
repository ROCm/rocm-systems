/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

static constexpr auto kComplexConj{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void hipConjf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x) {
    *result = hipConjf(x);
  }
  __global__ void hipConjf_kernel_v2(hipFloatComplex* result, hipDoubleComplex x) {
    *result = hipConjf(x);
  }
  __global__ void hipConjf_kernel_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
  __global__ void hipConjf_kernel_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
  __global__ void hipConjf_kernel_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
  __global__ void hipConjf_kernel_v6(hipDoubleComplex* result, hipFloatComplex x) {
    *result = hipConjf(x);
  }
  __global__ void hipConjf_kernel_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }
  __global__ void hipConj_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x) {
    *result = hipConj(x);
  }
  __global__ void hipConj_kernel_v2(hipDoubleComplex* result, hipFloatComplex x) {
    *result = hipConj(x);
  }
  __global__ void hipConj_kernel_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
  __global__ void hipConj_kernel_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
  __global__ void hipConj_kernel_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
  __global__ void hipConj_kernel_v6(hipFloatComplex* result, hipDoubleComplex x) {
    *result = hipConj(x);
  }
  __global__ void hipConj_kernel_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }
  void hipConjf_v1(hipFloatComplex* result, hipFloatComplex* x) { *result = hipConjf(x); }
  void hipConjf_v2(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConjf(x); }
  void hipConjf_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
  void hipConjf_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
  void hipConjf_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
  void hipConjf_v6(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConjf(x); }
  void hipConjf_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }
  void hipConj_v1(hipDoubleComplex* result, hipDoubleComplex* x) { *result = hipConj(x); }
  void hipConj_v2(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConj(x); }
  void hipConj_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
  void hipConj_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
  void hipConj_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
  void hipConj_v6(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConj(x); }
  void hipConj_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }
)"};

static constexpr auto kComplexReal{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCrealf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v3(float* result, float x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v4(float* result, Dummy x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) {
    *result = hipCrealf(x);
  }
  __global__ void hipCrealf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCrealf(x); }
  __global__ void hipCreal_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v2(double* result, hipFloatComplex x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v3(double* result, double x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v4(double* result, Dummy x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) {
    *result = hipCreal(x);
  }
  __global__ void hipCreal_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCreal(x); }
  void hipCrealf_v1(float* result, hipFloatComplex* x) { *result = hipCrealf(x); }
  void hipCrealf_v2(float* result, hipDoubleComplex x) { *result = hipCrealf(x); }
  void hipCrealf_v3(float* result, float x) { *result = hipCrealf(x); }
  void hipCrealf_v4(float* result, Dummy x) { *result = hipCrealf(x); }
  void hipCrealf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCrealf(x); }
  void hipCrealf_v6(Dummy* result, hipFloatComplex x) { *result = hipCrealf(x); }
  void hipCreal_v1(double* result, hipDoubleComplex* x) { *result = hipCreal(x); }
  void hipCreal_v2(double* result, hipFloatComplex x) { *result = hipCreal(x); }
  void hipCreal_v3(double* result, double x) { *result = hipCreal(x); }
  void hipCreal_v4(double* result, Dummy x) { *result = hipCreal(x); }
  void hipCreal_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCreal(x); }
  void hipCreal_v6(Dummy* result, hipDoubleComplex x) { *result = hipCreal(x); }
)"};

static constexpr auto kComplexImag{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCimagf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v3(float* result, float x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v4(float* result, Dummy x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) {
    *result = hipCimagf(x);
  }
  __global__ void hipCimagf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCimagf(x); }
  __global__ void hipCimag_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v2(double* result, hipFloatComplex x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v3(double* result, double x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v4(double* result, Dummy x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) {
    *result = hipCimag(x);
  }
  __global__ void hipCimag_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCimag(x); }
  void hipCimagf_v1(float* result, hipFloatComplex* x) { *result = hipCimagf(x); }
  void hipCimagf_v2(float* result, hipDoubleComplex x) { *result = hipCimagf(x); }
  void hipCimagf_v3(float* result, float x) { *result = hipCimagf(x); }
  void hipCimagf_v4(float* result, Dummy x) { *result = hipCimagf(x); }
  void hipCimagf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCimagf(x); }
  void hipCimagf_v6(Dummy* result, hipFloatComplex x) { *result = hipCimagf(x); }
  void hipCimag_v1(double* result, hipDoubleComplex* x) { *result = hipCimag(x); }
  void hipCimag_v2(double* result, hipFloatComplex x) { *result = hipCimag(x); }
  void hipCimag_v3(double* result, double x) { *result = hipCimag(x); }
  void hipCimag_v4(double* result, Dummy x) { *result = hipCimag(x); }
  void hipCimag_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCimag(x); }
  void hipCimag_v6(Dummy* result, hipDoubleComplex x) { *result = hipCimag(x); }
)"};

static constexpr auto kComplexAbs{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCabsf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCabsf(x); }
  __global__ void hipCabsf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCabsf(x); }
  __global__ void hipCabsf_kernel_v3(float* result, float x) { *result = hipCabsf(x); }
  __global__ void hipCabsf_kernel_v4(float* result, Dummy x) { *result = hipCabsf(x); }
  __global__ void hipCabsf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) {
    *result = hipCabsf(x);
  }
  __global__ void hipCabsf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCabsf(x); }
  __global__ void hipCabs_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCabs(x); }
  __global__ void hipCabs_kernel_v2(double* result, hipFloatComplex x) { *result = hipCabs(x); }
  __global__ void hipCabs_kernel_v3(double* result, double x) { *result = v(x); }
  __global__ void hipCabs_kernel_v4(double* result, Dummy x) { *result = hipCabs(x); }
  __global__ void hipCabs_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) {
    *result = hipCabs(x);
  }
  __global__ void hipCabs_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCabs(x); }
  void hipCabsf_v1(float* result, hipFloatComplex* x) { *result = hipCabsf(x); }
  void hipCabsf_v2(float* result, hipDoubleComplex x) { *result = hipCabsf(x); }
  void hipCabsf_v3(float* result, float x) { *result = hipCabsf(x); }
  void hipCabsf_v4(float* result, Dummy x) { *result = hipCabsf(x); }
  void hipCabsf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCabsf(x); }
  void hipCabsf_v6(Dummy* result, hipFloatComplex x) { *result = hipCabsf(x); }
  void hipCabs_v1(double* result, hipDoubleComplex* x) { *result = hipCabs(x); }
  void hipCabs_v2(double* result, hipFloatComplex x) { *result = hipCabs(x); }
  void hipCabs_v3(double* result, double x) { *result = hipCabs(x); }
  void hipCabs_v4(double* result, Dummy x) { *result = hipCabs(x); }
  void hipCabs_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCabs(x); }
  void hipCabs_v6(Dummy* result, hipDoubleComplex x) { *result = hipCabs(x); }
)"};

static constexpr auto kComplexSqabs{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCsqabsf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCsqabsf(x); }
  __global__ void hipCsqabsf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCsqabsf(x); }
  __global__ void hipCsqabsf_kernel_v3(float* result, float x) { *result = hipCsqabsf(x); }
  __global__ void hipCsqabsf_kernel_v4(float* result, Dummy x) { *result = hipCsqabsf(x); }
  __global__ void hipCsqabsf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) {
    *result = hipCsqabsf(x);
  }
  __global__ void hipCsqabsf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCsqabs(x); }
  __global__ void hipCsqabs_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCsqabs(x); }
  __global__ void hipCsqabs_kernel_v2(double* result, hipFloatComplex x) { *result = hipCsqabs(x); }
  __global__ void hipCsqabs_kernel_v3(double* result, double x) { *result = hipCsqabs(x); }
  __global__ void hipCsqabs_kernel_v4(double* result, Dummy x) { *result = hipCsqabs(x); }
  __global__ void hipCsqabs_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) {
    *result = hipCsqabs(x);
  }
  __global__ void hipCsqabs_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCsqabs(x); }
  void hipCsqabsf_v1(float* result, hipFloatComplex* x) { *result = hipCsqabsf(x); }
  void hipCsqabsf_v2(float* result, hipDoubleComplex x) { *result = hipCsqabsf(x); }
  void hipCsqabsf_v3(float* result, float x) { *result = hipCsqabsf(x); }
  void hipCsqabsf_v4(float* result, Dummy x) { *result = hipCsqabsf(x); }
  void hipCsqabsf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCsqabsf(x); }
  void hipCsqabsf_v6(Dummy* result, hipFloatComplex x) { *result = hipCsqabsf(x); }
  void hipCsqabs_v1(double* result, hipDoubleComplex* x) { *result = hipCsqabs(x); }
  void hipCsqabs_v2(double* result, hipFloatComplex x) { *result = hipCsqabs(x); }
  void hipCsqabs_v3(double* result, double x) { *result = hipCsqabs(x); }
  void hipCsqabs_v4(double* result, Dummy x) { *result = hipCsqabs(x); }
  void hipCsqabs_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCsqabs(x); }
  void hipCsqabs_v6(Dummy* result, hipDoubleComplex x) { *result = hipCsqabs(x); }
)"};
