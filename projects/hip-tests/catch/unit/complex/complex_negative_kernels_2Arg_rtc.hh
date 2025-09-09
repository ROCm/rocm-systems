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

static constexpr auto kComplexAdd{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCaddf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,
                                        hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v2(hipFloatComplex* result, hipFloatComplex x,
                                        hipFloatComplex* y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v6(hipFloatComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,
                                         hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCaddf_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  __global__ void hipCadd_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x,
                                        hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipDoubleComplex* y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v5(hipDoubleComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v10(hipFloatComplex* result, hipDoubleComplex x,
                                         hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  __global__ void hipCadd_kernel_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCaddf_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCaddf_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCadd_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v5(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v6(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v10(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
  void hipCadd_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCadd(x, y);
  }
)"};

static constexpr auto kComplexSub{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCsubf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,
                                        hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v2(hipFloatComplex* result, hipFloatComplex x,
                                        hipFloatComplex* y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v6(hipFloatComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,
                                         hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsubf_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  __global__ void hipCsub_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x,
                                        hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipDoubleComplex* y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v5(hipDoubleComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v10(hipFloatComplex* result, hipDoubleComplex x,
                                         hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  __global__ void hipCsub_kernel_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsubf_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCsubf(x, y);
  }
  void hipCaddf_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCaddf(x, y);
  }
  void hipCsubf_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsubf_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCsubf(x, y);
  }
  void hipCsub_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v5(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v6(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v10(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
  void hipCsub_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCsub(x, y);
  }
)"};

static constexpr auto kComplexMul{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCmulf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,
                                        hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v2(hipFloatComplex* result, hipFloatComplex x,
                                        hipFloatComplex* y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v6(hipFloatComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,
                                         hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmulf_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  __global__ void hipCmul_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x,
                                        hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipDoubleComplex* y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v5(hipDoubleComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v10(hipFloatComplex* result, hipDoubleComplex x,
                                         hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  __global__ void hipCmul_kernel_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmulf_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmulf_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCmulf(x, y);
  }
  void hipCmul_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v5(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v6(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v10(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
  void hipCmul_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCmul(x, y);
  }
)"};

static constexpr auto kComplexDiv{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCdivf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,
                                        hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v2(hipFloatComplex* result, hipFloatComplex x,
                                        hipFloatComplex* y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v6(hipFloatComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,
                                         hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdivf_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  __global__ void hipCdiv_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x,
                                        hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipDoubleComplex* y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v5(hipDoubleComplex* result, hipFloatComplex x,
                                        hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x,
                                        hipFloatComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v10(hipFloatComplex* result, hipDoubleComplex x,
                                         hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  __global__ void hipCdiv_kernel_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdivf_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v3(hipFloatComplex* result, float x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v4(hipFloatComplex* result, hipFloatComplex x, float y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v9(float* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdivf_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {
    *result = hipCdivf(x, y);
  }
  void hipCdiv_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v5(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v6(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v10(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
  void hipCdiv_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {
    *result = hipCdiv(x, y);
  }
)"};
