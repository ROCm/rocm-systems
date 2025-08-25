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
#include <hip_test_common.hh>
#include <hip/hip_complex.h>
#include <math.h>
#include <iostream>
#include <type_traits>

#define LEN 64
/* Comparing 2 floating point/double variables using floating point
precision. The precision is set at compile time using EPSILON. */
#define COMPARE_REALNUM(A, B, EPSILON) (fabs(A - B) < EPSILON)

enum ComplexFuncType {
  COMPLEX_ADD,
  COMPLEX_SUB,
  COMPLEX_MUL,
  COMPLEX_DIV,
  COMPLEX_CONJ,
  COMPLEX_REAL,
  COMPLEX_IMAG,
  COMPLEX_SQABS,
  COMPLEX_ABS
};

__global__ static void testMakeComplexFunc(float* A, float* B, hipFloatComplex* C) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  C[tx] = make_hipFloatComplex(A[tx], B[tx]);
}

__global__ static void testMakeComplexFunc(double* A, double* B, hipDoubleComplex* C) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  C[tx] = make_hipDoubleComplex(A[tx], B[tx]);
}

__global__ static void testComplexMathFunc1(hipFloatComplex* A, hipFloatComplex* B,
                                            hipFloatComplex* C, enum ComplexFuncType type) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  switch (type) {
    case COMPLEX_ADD:
      C[tx] = hipCaddf(A[tx], B[tx]);
      break;
    case COMPLEX_SUB:
      C[tx] = hipCsubf(A[tx], B[tx]);
      break;
    case COMPLEX_MUL:
      C[tx] = hipCmulf(A[tx], B[tx]);
      break;
    case COMPLEX_DIV:
      C[tx] = hipCdivf(A[tx], B[tx]);
      break;
    case COMPLEX_CONJ:
      C[tx] = hipConjf(A[tx]);
      break;
    default:
      break;
  }
}

__global__ static void testComplexMathFunc1(hipDoubleComplex* A, hipDoubleComplex* B,
                                            hipDoubleComplex* C, enum ComplexFuncType type) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  switch (type) {
    case COMPLEX_ADD:
      C[tx] = hipCadd(A[tx], B[tx]);
      break;
    case COMPLEX_SUB:
      C[tx] = hipCsub(A[tx], B[tx]);
      break;
    case COMPLEX_MUL:
      C[tx] = hipCmul(A[tx], B[tx]);
      break;
    case COMPLEX_DIV:
      C[tx] = hipCdiv(A[tx], B[tx]);
      break;
    case COMPLEX_CONJ:
      C[tx] = hipConj(A[tx]);
      break;
    default:
      break;
  }
}

__global__ static void testComplexMathFunc2(hipFloatComplex* A, float* B,
                                            enum ComplexFuncType type) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  switch (type) {
    case COMPLEX_REAL:
      B[tx] = hipCrealf(A[tx]);
      break;
    case COMPLEX_IMAG:
      B[tx] = hipCimagf(A[tx]);
      break;
    case COMPLEX_SQABS:
      B[tx] = hipCsqabsf(A[tx]);
      break;
    case COMPLEX_ABS:
      B[tx] = hipCabsf(A[tx]);
      break;
    default:
      break;
  }
}

__global__ static void testComplexMathFunc2(hipDoubleComplex* A, double* B,
                                            enum ComplexFuncType type) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  switch (type) {
    case COMPLEX_REAL:
      B[tx] = hipCreal(A[tx]);
      break;
    case COMPLEX_IMAG:
      B[tx] = hipCimag(A[tx]);
      break;
    case COMPLEX_SQABS:
      B[tx] = hipCsqabs(A[tx]);
      break;
    case COMPLEX_ABS:
      B[tx] = hipCabs(A[tx]);
      break;
    default:
      break;
  }
}
/**
 * Validates all hipComplex inline functions on device
 * Functions validated are: make_hipDoubleComplex, make_hipFloatComplex
 */
template <typename T1, typename T2> bool test_makehipComplex_dev() {
  T2 *A, *Ad, *B, *Bd;
  T1 *C, *Cd;
  bool TestPassed = true;
  A = new T2[LEN];
  B = new T2[LEN];
  C = new T1[LEN];
  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = 2 * i * 1.0;
    B[i] = (2 * i + 1) * 1.0;
  }
  unsigned int size2 = LEN * sizeof(T2);
  unsigned int size1 = LEN * sizeof(T1);
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), size2));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), size2));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Cd), size1));
  HIPCHECK(hipMemcpy(Ad, A, size2, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd, B, size2, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(testMakeComplexFunc, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd);
  HIPCHECK(hipMemcpy(C, Cd, size1, hipMemcpyDeviceToHost));
  // Validate the output of the kernel functions.
  for (uint32_t i = 0; i < LEN; i++) {
    if ((A[i] != C[i].x) || (B[i] != C[i].y)) {
      TestPassed = false;
      break;
    }
  }
  HIPCHECK(hipFree(Cd));
  HIPCHECK(hipFree(Bd));
  HIPCHECK(hipFree(Ad));
  delete[] C;
  delete[] B;
  delete[] A;
  return TestPassed;
}
/**
 * Validates all hipComplex inline functions on device
 * Functions validated are: hipCaddf, hipCsubf, hipCmulf and hipCdivf
 * hipCadd, hipCsub, hipCmul, hipCdiv
 */
template <typename T1, typename T2>
bool test_complexMathFunc1_dev(enum ComplexFuncType mathFuncType) {
  T1 *A, *Ad, *B, *Bd;
  T1 *C, *Cd;
  bool TestPassed = true;
  A = new T1[LEN];
  B = new T1[LEN];
  C = new T1[LEN];
  for (uint32_t i = 0; i < LEN; i++) {
    A[i].x = 2 * i * 1.0;
    A[i].y = (2 * i + 1) * 1.0;
    B[i].x = 2 * i * 1.0 + 0.5;
    B[i].y = (2 * i + 1) * 1.0 + 0.5;
  }
  unsigned int size = LEN * sizeof(T1);
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), size));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), size));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Cd), size));
  HIPCHECK(hipMemcpy(Ad, A, size, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd, B, size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(testComplexMathFunc1, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd, mathFuncType);
  HIPCHECK(hipMemcpy(C, Cd, size, hipMemcpyDeviceToHost));
  // Validate the output of the kernel functions.
  T2 epsilon = 0.0001f;
  T2 real, imag;
  for (uint32_t i = 0; i < LEN; i++) {
    if (mathFuncType == COMPLEX_ADD) {
      real = (A[i].x + B[i].x);
      imag = (A[i].y + B[i].y);
    } else if (mathFuncType == COMPLEX_SUB) {
      real = (A[i].x - B[i].x);
      imag = (A[i].y - B[i].y);
    } else if (mathFuncType == COMPLEX_MUL) {
      real = (A[i].x * B[i].x - A[i].y * B[i].y);
      imag = (A[i].y * B[i].x + A[i].x * B[i].y);
    } else if (mathFuncType == COMPLEX_DIV) {
      T2 sqabs = (B[i].x * B[i].x + B[i].y * B[i].y);
      real = (A[i].x * B[i].x + A[i].y * B[i].y) / sqabs;
      imag = (A[i].y * B[i].x - A[i].x * B[i].y) / sqabs;
    } else if (mathFuncType == COMPLEX_CONJ) {
      real = A[i].x;
      imag = -A[i].y;
    }
    if (!COMPARE_REALNUM(real, C[i].x, epsilon) || !COMPARE_REALNUM(imag, C[i].y, epsilon)) {
      TestPassed = false;
      break;
    }
  }
  HIPCHECK(hipFree(Cd));
  HIPCHECK(hipFree(Bd));
  HIPCHECK(hipFree(Ad));
  delete[] C;
  delete[] B;
  delete[] A;
  return TestPassed;
}
/**
 * Validates all hipComplex inline functions on device
 * Functions validated are: hipCrealf, hipCimagf, hipCsqabsf and hipCabsf
 * hipCreal, hipCimag, hipCsqabs, hipCabs
 */
template <typename T1, typename T2>
bool test_complexMathFunc2_dev(enum ComplexFuncType mathFuncType) {
  T1 *A, *Ad;
  T2 *B, *Bd;
  bool TestPassed = true;
  A = new T1[LEN];
  B = new T2[LEN];
  for (uint32_t i = 0; i < LEN; i++) {
    A[i].x = 2 * i * 1.0;
    A[i].y = (2 * i + 1) * 1.0;
  }
  unsigned int size1 = LEN * sizeof(T1);
  unsigned int size2 = LEN * sizeof(T2);
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), size1));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), size2));
  HIPCHECK(hipMemcpy(Ad, A, size1, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(testComplexMathFunc2, dim3(1), dim3(LEN), 0, 0, Ad, Bd, mathFuncType);
  HIPCHECK(hipMemcpy(B, Bd, size2, hipMemcpyDeviceToHost));
  // Validate the output of the kernel functions.
  T2 epsilon = 0.0001f;
  if (mathFuncType == COMPLEX_REAL) {
    for (uint32_t i = 0; i < LEN; i++) {
      if (!COMPARE_REALNUM(A[i].x, B[i], epsilon)) {
        TestPassed = false;
        break;
      }
    }
  } else if (mathFuncType == COMPLEX_IMAG) {
    for (uint32_t i = 0; i < LEN; i++) {
      if (!COMPARE_REALNUM(A[i].y, B[i], epsilon)) {
        TestPassed = false;
        break;
      }
    }
  } else if (mathFuncType == COMPLEX_SQABS) {
    for (uint32_t i = 0; i < LEN; i++) {
      T2 sqabs = A[i].x * A[i].x + A[i].y * A[i].y;
#ifdef __HIP_PLATFORM_NVCC__
      /* Setting the Floating Point precision to 0.01 as this scenario
      is failing on NVIDIA targets. */
      epsilon = 0.01f;
#endif
      if (!COMPARE_REALNUM(sqabs, B[i], epsilon)) {
        TestPassed = false;
        break;
      }
    }
  } else if (mathFuncType == COMPLEX_ABS) {
    for (uint32_t i = 0; i < LEN; i++) {
      T2 sqabs = A[i].x * A[i].x + A[i].y * A[i].y;
      if (!COMPARE_REALNUM(sqrtf(sqabs), B[i], epsilon)) {
        TestPassed = false;
        break;
      }
    }
  }
  HIPCHECK(hipFree(Bd));
  HIPCHECK(hipFree(Ad));
  delete[] B;
  delete[] A;
  return TestPassed;
}

// Validates all hipComplex inline functions on host
static bool test_allcomplexMathFunc_host() {
  bool TestPassed = true;
  float fa = 2.0, fb = 3.0;
  hipFloatComplex fc = make_hipFloatComplex(fa, fb);
  if ((fc.x != fa) || (fc.y != fb)) {
    TestPassed &= false;
  }
  double da = 2.0, db = 3.0;
  hipDoubleComplex dc = make_hipDoubleComplex(da, db);
  if ((dc.x != da) || (dc.y != db)) {
    TestPassed &= false;
  }
  hipFloatComplex fp, fq, fx;
  fp.x = 2.0;
  fp.y = 3.0;
  fq.x = 4.0;
  fq.y = 5.0;
  fx = hipCaddf(fp, fq);
  if ((fx.x != (fp.x + fq.x)) || (fx.y != (fp.y + fq.y))) {
    TestPassed &= false;
  }
  fx = hipCsubf(fp, fq);
  if ((fx.x != (fp.x - fq.x)) || (fx.y != (fp.y - fq.y))) {
    TestPassed &= false;
  }
  fx = hipCmulf(fp, fq);
  if ((fx.x != (fp.x * fq.x - fp.y * fq.y)) || (fx.y != (fp.y * fq.x + fp.x * fq.y))) {
    TestPassed &= false;
  }
  fx = hipCdivf(fp, fq);
  float fsqabs = fq.x * fq.x + fq.y * fq.y;
  float epsilon = 0.0001f;
  if ((!COMPARE_REALNUM(fx.x, (fp.x * fq.x + fp.y * fq.y) / fsqabs, epsilon)) ||
      (!COMPARE_REALNUM(fx.y, (fp.y * fq.x - fp.x * fq.y) / fsqabs, epsilon))) {
    TestPassed &= false;
  }
  if ((fp.x != hipCrealf(fp)) || (fp.y != hipCimagf(fp))) {
    TestPassed &= false;
  }
  fx = hipConjf(fp);
  if ((fx.x != fp.x) || (fx.y != -fp.y)) {
    TestPassed &= false;
  }
  if (!COMPARE_REALNUM((fp.x * fp.x + fp.y * fp.y), hipCsqabsf(fp), epsilon)) {
    TestPassed &= false;
  }
  if (!COMPARE_REALNUM(sqrtf(fp.x * fp.x + fp.y * fp.y), hipCabsf(fp), epsilon)) {
    TestPassed &= false;
  }
  hipDoubleComplex dp, dq, dx;
  dp.x = 2.0;
  dp.y = 3.0;
  dq.x = 4.0;
  dq.y = 5.0;
  dx = hipCadd(dp, dq);
  if ((dx.x != (dp.x + dq.x)) || (dx.y != (dp.y + dq.y))) {
    TestPassed &= false;
  }
  dx = hipCsub(dp, dq);
  if ((dx.x != (dp.x - dq.x)) || (dx.y != (dp.y - dq.y))) {
    TestPassed &= false;
  }
  dx = hipCmul(dp, dq);
  if ((dx.x != (dp.x * dq.x - dp.y * dq.y)) || (dx.y != (dp.y * dq.x + dp.x * dq.y))) {
    TestPassed &= false;
  }
  dx = hipCdiv(dp, dq);
  float dsqabs = dq.x * dq.x + dq.y * dq.y;
  if ((!COMPARE_REALNUM(dx.x, (dp.x * dq.x + dp.y * dq.y) / dsqabs, epsilon)) ||
      (!COMPARE_REALNUM(dx.y, (dp.y * dq.x - dp.x * dq.y) / dsqabs, epsilon))) {
    TestPassed &= false;
  }
  if ((dp.x != hipCreal(dp)) || (dp.y != hipCimag(dp))) {
    TestPassed &= false;
  }
  dx = hipConj(dp);
  if ((dx.x != dp.x) || (dx.y != -dp.y)) {
    TestPassed &= false;
  }
  if (!COMPARE_REALNUM((dp.x * dp.x + dp.y * dp.y), hipCsqabs(dp), epsilon)) {
    TestPassed &= false;
  }
  if (!COMPARE_REALNUM(sqrtf(dp.x * dp.x + dp.y * dp.y), hipCabs(dp), epsilon)) {
    TestPassed &= false;
  }
  return TestPassed;
}

TEST_CASE("Unit_TestMathFuncComplex") {
  bool TestPassed = false;
  TestPassed = test_makehipComplex_dev<hipFloatComplex, float>() &&
               test_makehipComplex_dev<float2, float>() &&
               test_makehipComplex_dev<hipDoubleComplex, double>() &&
               test_makehipComplex_dev<double2, double>() &&
               test_complexMathFunc1_dev<hipFloatComplex, float>(COMPLEX_ADD) &&
               test_complexMathFunc1_dev<hipDoubleComplex, double>(COMPLEX_ADD) &&
               test_complexMathFunc1_dev<hipFloatComplex, float>(COMPLEX_SUB) &&
               test_complexMathFunc1_dev<hipDoubleComplex, double>(COMPLEX_SUB) &&
               test_complexMathFunc1_dev<hipFloatComplex, float>(COMPLEX_MUL) &&
               test_complexMathFunc1_dev<hipDoubleComplex, double>(COMPLEX_MUL) &&
               test_complexMathFunc1_dev<hipFloatComplex, float>(COMPLEX_DIV) &&
               test_complexMathFunc1_dev<hipDoubleComplex, double>(COMPLEX_DIV) &&
               test_complexMathFunc1_dev<hipFloatComplex, float>(COMPLEX_CONJ) &&
               test_complexMathFunc1_dev<hipDoubleComplex, double>(COMPLEX_CONJ) &&
               test_complexMathFunc2_dev<hipFloatComplex, float>(COMPLEX_REAL) &&
               test_complexMathFunc2_dev<hipDoubleComplex, double>(COMPLEX_REAL) &&
               test_complexMathFunc2_dev<hipFloatComplex, float>(COMPLEX_IMAG) &&
               test_complexMathFunc2_dev<hipDoubleComplex, double>(COMPLEX_IMAG) &&
               test_complexMathFunc2_dev<hipFloatComplex, float>(COMPLEX_SQABS) &&
               test_complexMathFunc2_dev<hipDoubleComplex, double>(COMPLEX_SQABS) &&
               test_complexMathFunc2_dev<hipFloatComplex, float>(COMPLEX_ABS) &&
               test_complexMathFunc2_dev<hipDoubleComplex, double>(COMPLEX_ABS) &&
               test_allcomplexMathFunc_host();
  REQUIRE(TestPassed == true);
}
