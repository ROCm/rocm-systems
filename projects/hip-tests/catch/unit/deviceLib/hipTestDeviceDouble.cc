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
#include <hip/math_functions.h>

#define N 512
#define SIZE (N * sizeof(double))

__global__ void test_sincos(double* a, double* b, double* c) {
  int tid = threadIdx.x;
  sincos(a[tid], b + tid, c + tid);
}

__global__ void test_sincospi(double* a, double* b, double* c) {
  int tid = threadIdx.x;
  sincospi(a[tid], b + tid, c + tid);
}

__global__ void test_llrint(double* a, int64_t* b) {
  int tid = threadIdx.x;
  b[tid] = llrint(a[tid]);
}

__global__ void test_lrint(double* a, int64_t* b) {
  int tid = threadIdx.x;
  b[tid] = lrint(a[tid]);
}

__global__ void test_rint(double* a, double* b) {
  int tid = threadIdx.x;
  b[tid] = rint(a[tid]);
}

__global__ void test_llround(double* a, int64_t* b) {
  int tid = threadIdx.x;
  b[tid] = llround(a[tid]);
}

__global__ void test_lround(double* a, int64_t* b) {
  int tid = threadIdx.x;
  b[tid] = lround(a[tid]);
}

__global__ void test_rhypot(double* a, double* b, double* c) {
  int tid = threadIdx.x;
  c[tid] = rhypot(a[tid], b[tid]);
}

__global__ void test_norm3d(double* a, double* b, double* c, double* d) {
  int tid = threadIdx.x;
  d[tid] = norm3d(a[tid], b[tid], c[tid]);
}

__global__ void test_norm4d(double* a, double* b, double* c, double* d, double* e) {
  int tid = threadIdx.x;
  e[tid] = norm4d(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_rnorm3d(double* a, double* b, double* c, double* d) {
  int tid = threadIdx.x;
  d[tid] = rnorm3d(a[tid], b[tid], c[tid]);
}

__global__ void test_rnorm4d(double* a, double* b, double* c, double* d, double* e) {
  int tid = threadIdx.x;
  e[tid] = rnorm4d(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_rnorm(double* a, double* b) {
  int tid = threadIdx.x;
  b[tid] = rnorm(N, a);
}

__global__ void test_erfinv(double* a, double* b) {
  int tid = threadIdx.x;
  b[tid] = erf(erfinv(a[tid]));
}

bool run_sincos() {
  double *A, *Ad, *B, *C, *Bd, *Cd;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_sincos, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if ((B[i] == sin(1.0)) && (C[i] == cos(1.0))) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_sincospi() {
  double *A, *Ad, *B, *C, *Bd, *Cd;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_sincospi, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if ((B[i] - sin(3.14 * 1.0) < 0.1) && (C[i] - cos(3.14 * 1.0) < 0.1)) {
      passed = 1;
    }
  }
  delete[] A;
  delete[] B;
  delete[] C;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_llrint() {
  double *A, *Ad;
  int64_t *B, *Bd;
  A = new double[N];
  B = new int64_t[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.345;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_llrint, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    int x = round(A[i]);
    if (B[i] == x) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_lrint() {
  double *A, *Ad;
  int64_t *B, *Bd;
  A = new double[N];
  B = new int64_t[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.345;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_lrint, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    int64_t x = round(A[i]);
    if (B[i] == x) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_rint() {
  double *A, *Ad;
  double *B, *Bd;
  A = new double[N];
  B = new double[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.345;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_rint, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    double x = round(A[i]);
    if (B[i] == x) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_llround() {
  double *A, *Ad;
  int64_t *B, *Bd;
  A = new double[N];
  B = new int64_t[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.345;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_llround, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    int64_t x = round(A[i]);
    if (B[i] == x) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_lround() {
  double *A, *Ad;
  int64_t *B, *Bd;
  A = new double[N];
  B = new int64_t[N];
  for (int i = 0; i < N; i++) {
    A[i] = 1.345;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_lround, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    int64_t x = round(A[i]);
    if (B[i] == x) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_norm3d() {
  double *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  D = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 3.0;
  }
  val = sqrt(1.0 + 4.0 + 9.0);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_norm3d, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
  HIP_CHECK(hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (D[i] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  HIP_CHECK(hipFree(Dd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_norm4d() {
  double *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  D = new double[N];
  E = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 3.0;
    D[i] = 4.0;
  }
  val = sqrt(1.0 + 4.0 + 9.0 + 16.0);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ed), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_norm4d, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);
  HIP_CHECK(hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (E[i] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  delete[] E;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  HIP_CHECK(hipFree(Dd));
  HIP_CHECK(hipFree(Ed));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_rhypot() {
  double *A, *Ad, *B, *Bd, *C, *Cd;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
  }
  val = 1 / sqrt(1.0 + 4.0);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_rhypot, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
  HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (C[i] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_rnorm3d() {
  double *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  D = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 3.0;
  }
  val = 1 / sqrt(1.0 + 4.0 + 9.0);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_rnorm3d, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
  HIP_CHECK(hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (D[i] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  HIP_CHECK(hipFree(Dd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_rnorm4d() {
  double *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
  A = new double[N];
  B = new double[N];
  C = new double[N];
  D = new double[N];
  E = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 3.0;
    D[i] = 4.0;
  }
  val = 1 / sqrt(1.0 + 4.0 + 9.0 + 16.0);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ed), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_rnorm4d, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);
  HIP_CHECK(hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (E[i] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  delete[] E;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  HIP_CHECK(hipFree(Dd));
  HIP_CHECK(hipFree(Ed));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_rnorm() {
  double *A, *Ad, *B, *Bd;
  A = new double[N];
  B = new double[N];
  double val = 0.0;
  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 0.0;
    val += 1.0;
  }
  val = 1 / sqrt(val);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_rnorm, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (B[0] - val < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

bool run_erfinv() {
  double *A, *Ad, *B, *Bd;
  A = new double[N];
  B = new double[N];
  for (int i = 0; i < N; i++) {
    A[i] = -0.6;
    B[i] = 0.0;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_erfinv, dim3(1), dim3(N), 0, 0, Ad, Bd);
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
  int passed = 0;
  for (int i = 0; i < 512; i++) {
    if (B[i] - A[i] < 0.000001) {
      passed = 1;
    }
  }

  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));

  if (passed == 1) {
    return true;
  }
  return false;
}

TEST_CASE("Unit_hipTrigDeviceFunc_Double") {
  bool result = false;
  result = run_sincos() && run_sincospi() && run_llrint() && run_norm3d() && run_norm4d() &&
           run_rnorm3d() && run_rnorm4d() && run_rnorm() && run_lround() && run_llround() &&
           run_rint() && run_rhypot() && run_erfinv();
  REQUIRE(result == true);
}
