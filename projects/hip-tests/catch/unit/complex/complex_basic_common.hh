/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_complex.h>

template <typename T>
__host__ __device__ T MakeComplexType(decltype(T().x) input_val1, decltype(T().x) input_val2) {
  if constexpr (std::is_same_v<T, hipFloatComplex>) {
    return make_hipFloatComplex(input_val1, input_val2);
  } else {
    return make_hipDoubleComplex(input_val1, input_val2);
  }
}

template <typename T>
__global__ void MakeComplexTypeKernel(T* const output_val, decltype(T().x) const input_val1,
                                      decltype(T().x) const input_val2) {
  *output_val = MakeComplexType<T>(input_val1, input_val2);
}
#if HT_AMD  // EXSWHTEC-321
__global__ void MakeHipComplexTypeKernel(hipComplex* const output_val, float const input_val1,
                                         float const input_val2) {
  *output_val = make_hipComplex(input_val1, input_val2);
}
#endif
template <typename T> struct CastType {};

template <> struct CastType<hipFloatComplex> {
  using type = hipDoubleComplex;
};

template <> struct CastType<hipDoubleComplex> {
  using type = hipFloatComplex;
};

template <typename T> using CastType_t = typename CastType<T>::type;

template <typename T1, typename T2> __device__ __host__ T1 CastComplexType(T2 const input_val) {
  if constexpr (std::is_same_v<hipDoubleComplex, T2>) {
    return hipComplexDoubleToFloat(input_val);
  } else if constexpr (std::is_same_v<hipFloatComplex, T2>) {
    return hipComplexFloatToDouble(input_val);
  }
}

template <typename T1, typename T2>
__global__ void CastComplexTypeKernel(T1* const output_val, T2 const input_val) {
  *output_val = CastComplexType<T1, T2>(input_val);
}

template <typename T> void CompareValues(T actual_val, T ref_val, double margin) {
  if (!std::isnan(ref_val)) {
    REQUIRE_THAT(actual_val, Catch::Matchers::WithinAbs(ref_val, margin));
  }
}
