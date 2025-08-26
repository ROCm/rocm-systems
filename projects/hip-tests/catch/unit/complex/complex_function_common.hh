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

#include "complex_basic_common.hh"

enum class ComplexFunction { kReal, kImag, kConj, kAdd, kSub, kMul, kDiv, kAbs, kSqabs, kFma };

inline std::string to_string(ComplexFunction function) {
  switch (function) {
    case ComplexFunction::kReal:
      return "real";
    case ComplexFunction::kImag:
      return "imaginary";
    case ComplexFunction::kConj:
      return "conjugate ";
    case ComplexFunction::kAdd:
      return "addition";
    case ComplexFunction::kSub:
      return "subtract";
    case ComplexFunction::kMul:
      return "multiply";
    case ComplexFunction::kDiv:
      return "divide";
    case ComplexFunction::kAbs:
      return "absolute";
    case ComplexFunction::kSqabs:
      return "square absolute";
    case ComplexFunction::kFma:
      return "fused multiply";
    default:
      return "Unknown";
  }
}

// Function that validates complex functions with complex type result
template <typename T>
void ValidateComplexResultFunction(ComplexFunction function, T input_val1, T input_val2,
                                   T input_val3, T actual_val) {
  decltype(T().x) ref_val_r;
  decltype(T().x) ref_val_i;
  double margin = 0;

  switch (function) {
    case ComplexFunction::kAdd: {
      ref_val_r = input_val1.x + input_val2.x;
      ref_val_i = input_val1.y + input_val2.y;
      break;
    }
    case ComplexFunction::kSub: {
      ref_val_r = input_val1.x - input_val2.x;
      ref_val_i = input_val1.y - input_val2.y;
      break;
    }
    case ComplexFunction::kMul: {
      ref_val_r = input_val1.x * input_val2.x - input_val1.y * input_val2.y;
      ref_val_i = input_val1.y * input_val2.x + input_val1.x * input_val2.y;
      break;
    }
    case ComplexFunction::kDiv: {
      decltype(T().x) sqabs = input_val2.x * input_val2.x + input_val2.y * input_val2.y;
      ref_val_r = (input_val1.x * input_val2.x + input_val1.y * input_val2.y) / sqabs;
      ref_val_i = (input_val1.y * input_val2.x - input_val1.x * input_val2.y) / sqabs;
#if HT_NVIDIA
      // Nvidia implementation uses scaling to guard against intermediate underflow and overflow
      margin = 0.000001;
#endif
      break;
    }
    case ComplexFunction::kConj: {
      ref_val_r = input_val1.x;
      ref_val_i = -input_val1.y;
      break;
    }
    case ComplexFunction::kFma: {
      ref_val_r = (input_val1.x * input_val2.x) + input_val3.x;
      ref_val_i = (input_val2.x * input_val1.y) + input_val3.y;

      ref_val_r = -(input_val1.y * input_val2.y) + ref_val_r;
      ref_val_i = (input_val1.x * input_val2.y) + ref_val_i;
      break;
    }
    default: {
      ref_val_r = input_val1.x;
      ref_val_i = input_val1.y;
      break;
    }
  }

  CompareValues(actual_val.x, ref_val_r, margin);
  CompareValues(actual_val.y, ref_val_i, margin);
}

// Function that validates complex functions with scalar type result
template <typename T>
void ValidateScalarResultFunction(ComplexFunction function, T input_val,
                                  decltype(T().x) actual_val) {
  decltype(T().x) ref_val;

  switch (function) {
    case ComplexFunction::kReal: {
      ref_val = input_val.x;
      break;
    }
    case ComplexFunction::kImag: {
      ref_val = input_val.y;
      break;
    }
    case ComplexFunction::kAbs: {
      decltype(T().x) sqabs = input_val.x * input_val.x + input_val.y * input_val.y;
      ref_val = std::sqrt(sqabs);
      break;
    }
    case ComplexFunction::kSqabs: {
      ref_val = input_val.x * input_val.x + input_val.y * input_val.y;
      break;
    }
    default: {
      ref_val = input_val.x;
      break;
    }
  }

  CompareValues(actual_val, ref_val, 0);
}

// Function that performs complex functions with complex type result on host/device
template <typename T>
__device__ __host__ void PerformComplexResultFunction(ComplexFunction function, T* output_val,
                                                      T input_val1, T input_val2, T input_val3) {
  if (function == ComplexFunction::kAdd) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCaddf(input_val1, input_val2);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCadd(input_val1, input_val2);
    }
  } else if (function == ComplexFunction::kSub) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCsubf(input_val1, input_val2);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCsub(input_val1, input_val2);
    }
  } else if (function == ComplexFunction::kMul) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCmulf(input_val1, input_val2);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCmul(input_val1, input_val2);
    }
  } else if (function == ComplexFunction::kDiv) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCdivf(input_val1, input_val2);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCdiv(input_val1, input_val2);
    }
  } else if (function == ComplexFunction::kConj) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipConjf(input_val1);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipConj(input_val1);
    }
  } else if (function == ComplexFunction::kFma) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCfmaf(input_val1, input_val2, input_val3);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCfma(input_val1, input_val2, input_val3);
    }
  } else {
    *output_val = input_val1;
  }
}

// Function that performs complex functions with scalar type result on host/device
template <typename T>
__device__ __host__ void PerformScalarResultFunction(ComplexFunction function,
                                                     decltype(T().x)* output_val, T input_val) {
  if (function == ComplexFunction::kReal) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCrealf(input_val);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCreal(input_val);
    }
  } else if (function == ComplexFunction::kImag) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCimagf(input_val);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCimag(input_val);
    }
  } else if (function == ComplexFunction::kAbs) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCabsf(input_val);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCabs(input_val);
    }
  } else if (function == ComplexFunction::kSqabs) {
    if constexpr (std::is_same_v<hipFloatComplex, T>) {
      *output_val = hipCsqabsf(input_val);
    } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
      *output_val = hipCsqabs(input_val);
    }
  } else {
    *output_val = input_val.x;
  }
}

// Kernel that calls device function which performs complex functions with complex type result
template <typename T>
__global__ void ComplexResultKernel(ComplexFunction function, T* output_val, T input_val1,
                                    T input_val2, T input_val3) {
  PerformComplexResultFunction(function, output_val, input_val1, input_val2, input_val3);
}

// Kernel that calls device function which performs complex functions with scalar type result
template <typename T>
__global__ void ScalarResultKernel(ComplexFunction function, decltype(T().x)* output_val,
                                   T input_val) {
  PerformScalarResultFunction(function, output_val, input_val);
}

// Wrapper function for testing complex functions with one input parameter on device
template <typename T> void ComplexFunctionUnaryDeviceTest(ComplexFunction function, T input_val) {
  if (function == ComplexFunction::kConj) {
    LinearAllocGuard<T> result_d{LinearAllocs::hipMalloc, sizeof(T)};
    LinearAllocGuard<T> result_h{LinearAllocs::hipHostMalloc, sizeof(T)};

    ComplexResultKernel<<<1, 1>>>(function, result_d.ptr(), input_val, input_val, input_val);
    HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    ValidateComplexResultFunction(function, input_val, input_val, input_val, result_h.ptr()[0]);
  } else {
    LinearAllocGuard<decltype(T().x)> result_d{LinearAllocs::hipMalloc, sizeof(decltype(T().x))};
    LinearAllocGuard<decltype(T().x)> result_h{LinearAllocs::hipHostMalloc,
                                               sizeof(decltype(T().x))};

    ScalarResultKernel<<<1, 1>>>(function, result_d.ptr(), input_val);
    HIP_CHECK(
        hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(decltype(T().x)), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    ValidateScalarResultFunction(function, input_val, result_h.ptr()[0]);
  }
}

// Wrapper function for testing complex functions with one input parameter on host
template <typename T> void ComplexFunctionUnaryHostTest(ComplexFunction function, T input_val) {
  if (function == ComplexFunction::kConj) {
    T result;
    PerformComplexResultFunction(function, &result, input_val, input_val, input_val);
    ValidateComplexResultFunction(function, input_val, input_val, input_val, result);
  } else {
    decltype(T().x) result;
    PerformScalarResultFunction(function, &result, input_val);
    ValidateScalarResultFunction(function, input_val, result);
  }
}

// Wrapper function for testing complex functions with two input parameters on device
template <typename T>
void ComplexFunctionBinaryDeviceTest(ComplexFunction function, T input_val1, T input_val2) {
  LinearAllocGuard<T> result_d{LinearAllocs::hipMalloc, sizeof(T)};
  LinearAllocGuard<T> result_h{LinearAllocs::hipHostMalloc, sizeof(T)};

  ComplexResultKernel<<<1, 1>>>(function, result_d.ptr(), input_val1, input_val2, input_val2);
  HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  ValidateComplexResultFunction(function, input_val1, input_val2, input_val2, result_h.ptr()[0]);
}

// Wrapper function for testing complex functions with two input parameters on host
template <typename T>
void ComplexFunctionBinaryHostTest(ComplexFunction function, T input_val1, T input_val2) {
  T result;
  PerformComplexResultFunction(function, &result, input_val1, input_val2, input_val2);
  ValidateComplexResultFunction(function, input_val1, input_val2, input_val2, result);
}

// Wrapper function for testing complex functions with three input parameters on device
template <typename T>
void ComplexFunctionTernaryDeviceTest(ComplexFunction function, T input_val1, T input_val2,
                                      T input_val3) {
  LinearAllocGuard<T> result_d{LinearAllocs::hipMalloc, sizeof(T)};
  LinearAllocGuard<T> result_h{LinearAllocs::hipHostMalloc, sizeof(T)};

  ComplexResultKernel<<<1, 1>>>(function, result_d.ptr(), input_val1, input_val2, input_val3);
  HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  ValidateComplexResultFunction(function, input_val1, input_val2, input_val3, result_h.ptr()[0]);
}

// Wrapper function for testing complex functions with three input parameters on host
template <typename T>
void ComplexFunctionTernaryHostTest(ComplexFunction function, T input_val1, T input_val2,
                                    T input_val3) {
  T result;
  PerformComplexResultFunction(function, &result, input_val1, input_val2, input_val3);
  ValidateComplexResultFunction(function, input_val1, input_val2, input_val3, result);
}