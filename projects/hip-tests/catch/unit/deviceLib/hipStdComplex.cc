/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <complex>

#pragma clang diagnostic ignored "-Wunused-variable"
// Tolerance for error
const double tolerance = 1e-6;

#define LEN 64

#define ALL_FUN                                                                                    \
  OP(add)                                                                                          \
  OP(sub)                                                                                          \
  OP(mul)                                                                                          \
  OP(div)                                                                                          \
  OP(abs)                                                                                          \
  OP(arg)                                                                                          \
  OP(sin)                                                                                          \
  OP(cos)

#define OP(x) CK_##x,
enum CalcKind { ALL_FUN };
#undef OP

#define OP(x)                                                                                      \
  case CK_##x:                                                                                     \
    return #x;
std::string getName(enum CalcKind CK) {
  switch (CK) { ALL_FUN }
  return "";  // To prevent compile warning
}
#undef OP

// Calculates function.
// If the function has one argument, B is ignored.
// If the function returns real number, converts it to a complex number.
#define ONE_ARG(func)                                                                              \
  case CK_##func:                                                                                  \
    return std::complex<FloatT>(func(A));

template <typename FloatT> __device__ __host__ std::complex<FloatT> calc(std::complex<FloatT> A,
                                                                         std::complex<FloatT> B,
                                                                         enum CalcKind CK) {
  switch (CK) {
    case CK_add:
      return A + B;
    case CK_sub:
      return A - B;
    case CK_mul:
      return A * B;
    case CK_div:
      return A / B;

      ONE_ARG(abs)
      ONE_ARG(arg)
      ONE_ARG(sin)
      ONE_ARG(cos)
  }
  return A;  // To prevent compile warning
}

template <typename FloatT> __global__ void kernel(std::complex<FloatT>* A, std::complex<FloatT>* B,
                                                  std::complex<FloatT>* C, enum CalcKind CK) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  C[tx] = calc<FloatT>(A[tx], B[tx], CK);
}

template <typename FloatT> void test() {
  typedef std::complex<FloatT> ComplexT;

  ComplexT *A, *Ad, *B, *Bd, *C, *Cd, *D;
  A = new ComplexT[LEN];
  B = new ComplexT[LEN];
  C = new ComplexT[LEN];
  D = new ComplexT[LEN];
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), sizeof(ComplexT) * LEN));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), sizeof(ComplexT) * LEN));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), sizeof(ComplexT) * LEN));

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = ComplexT((i + 1) * 1.0f, (i + 2) * 1.0f);
    B[i] = A[i];
    C[i] = A[i];
  }
  HIP_CHECK(hipMemcpy(Ad, A, sizeof(ComplexT) * LEN, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, sizeof(ComplexT) * LEN, hipMemcpyHostToDevice));

  // Run kernel for a calculation kind and verify by comparing with host
  // calculation result. Returns false if fails.
  auto test_fun = [&](enum CalcKind CK) {
    hipLaunchKernelGGL(kernel<FloatT>, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd, CK);
    HIP_CHECK(hipMemcpy(C, Cd, sizeof(ComplexT) * LEN, hipMemcpyDeviceToHost));
    bool pass = true;
    for (int i = 0; i < LEN; i++) {
      ComplexT Expected = calc(A[i], B[i], CK);
      FloatT error = abs(C[i] - Expected);
      if (abs(Expected) > tolerance) error /= abs(Expected);
      pass &= error < tolerance;
    }
    return pass;
  };

#define OP(x) assert(test_fun(CK_##x));
  ALL_FUN
#undef OP

  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
}

#if HT_AMD
TEST_CASE(Unit_StdComplex) {
  SECTION("Test run with float") { test<float>(); }
  SECTION("Test run with double") { test<double>(); }
}
#endif
