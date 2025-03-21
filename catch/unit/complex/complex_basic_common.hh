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
    REQUIRE_THAT(actual_val, Catch::WithinAbs(ref_val, margin));
  }
}
