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

#include "vector_types_common.hh"

enum class VectorOperation {
  kIncrementPrefix,
  kIncrementPostfix,
  kDecrementPrefix,
  kDecrementPostfix,
  kAddAssign,
  kSubtractAssign,
  kMultiplyAssign,
  kDivideAssign,
  kNegate,
  kBitwiseNot,
  kModuloAssign,
  kBitwiseXorAssign,
  kBitwiseOrAssign,
  kBitwiseAndAssign,
  kRightShiftAssign,
  kLeftShiftAssign,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kEqual,
  kNotEqual,
  kModulo,
  kBitwiseXor,
  kBitwiseOr,
  kBitwiseAnd,
  kRightShift,
  kLeftShift,
  kSubscript,
};

inline std::string to_string(VectorOperation operation) {
  switch (operation) {
    case VectorOperation::kIncrementPrefix:
      return "increment (prefix)";
    case VectorOperation::kIncrementPostfix:
      return "increment (postfix)";
    case VectorOperation::kDecrementPrefix:
      return "decrement (prefix)";
    case VectorOperation::kDecrementPostfix:
      return "decrement (postfix)";
    case VectorOperation::kAddAssign:
      return "add and assign";
    case VectorOperation::kSubtractAssign:
      return "subtract and assign";
    case VectorOperation::kMultiplyAssign:
      return "multiply and assign";
    case VectorOperation::kDivideAssign:
      return "divide and assign";
    case VectorOperation::kNegate:
      return "negate";
    case VectorOperation::kBitwiseNot:
      return "bitwise not";
    case VectorOperation::kModuloAssign:
      return "modulo and assign";
    case VectorOperation::kBitwiseXorAssign:
      return "bitwise XOR and assign";
    case VectorOperation::kBitwiseOrAssign:
      return "bitwise OR and assign";
    case VectorOperation::kBitwiseAndAssign:
      return "bitwise AND and assign";
    case VectorOperation::kRightShiftAssign:
      return "right shift and assign";
    case VectorOperation::kLeftShiftAssign:
      return "left shift and assign";
    case VectorOperation::kAdd:
      return "add";
    case VectorOperation::kSubtract:
      return "subtract";
    case VectorOperation::kMultiply:
      return "multiply";
    case VectorOperation::kDivide:
      return "divide";
    case VectorOperation::kEqual:
      return "equal";
    case VectorOperation::kNotEqual:
      return "not equal";
    case VectorOperation::kModulo:
      return "modulo";
    case VectorOperation::kBitwiseXor:
      return "bitwise XOR";
    case VectorOperation::kBitwiseOr:
      return "bitwise OR";
    case VectorOperation::kBitwiseAnd:
      return "bitwise AND";
    case VectorOperation::kRightShift:
      return "right shift";
    case VectorOperation::kLeftShift:
      return "left shift";
    case VectorOperation::kSubscript:
      return "subscript";
    default:
      return "Unknown";
  }
}

template <typename T>
void SanityCheck(VectorOperation operation, T vector, decltype(T().x) value1,
                 decltype(T().x) value2) {
  if (operation == VectorOperation::kIncrementPrefix) {
    ++value1;
  } else if (operation == VectorOperation::kIncrementPostfix) {
    value1++;
  } else if (operation == VectorOperation::kDecrementPrefix) {
    --value1;
  } else if (operation == VectorOperation::kDecrementPostfix) {
    value1--;
  } else if (operation == VectorOperation::kAddAssign) {
    value1 += value2;
  } else if (operation == VectorOperation::kSubtractAssign) {
    value1 -= value2;
  } else if (operation == VectorOperation::kMultiplyAssign) {
    value1 *= value2;
  } else if (operation == VectorOperation::kDivideAssign) {
    value1 /= value2;
  } else if (operation == VectorOperation::kAdd) {
    value1 = value1 + value2;
  } else if (operation == VectorOperation::kSubtract) {
    value1 = value1 - value2;
  } else if (operation == VectorOperation::kMultiply) {
    value1 = value1 * value2;
  } else if (operation == VectorOperation::kDivide) {
    value1 = value1 / value2;
  } else if (operation == VectorOperation::kEqual) {
    value1 = (value1 == value2) ? 2 * value1 : 3 * value1;
  } else if (operation == VectorOperation::kNotEqual) {
    value1 = (value1 != value2) ? 2 * value1 : 3 * value1;
  } else if (operation == VectorOperation::kSubscript) {
    value1 = value2;
  } else {
    if constexpr (std::is_signed_v<decltype(T().x)>) {
      if (operation == VectorOperation::kNegate) {
        value1 = -value1;
      }
    }
    if constexpr (std::is_integral_v<decltype(T().x)>) {
      if (operation == VectorOperation::kBitwiseNot) {
        value1 = ~value1;
      } else if (operation == VectorOperation::kModuloAssign) {
        value1 %= value2;
      } else if (operation == VectorOperation::kBitwiseXorAssign) {
        value1 ^= value2;
      } else if (operation == VectorOperation::kBitwiseOrAssign) {
        value1 |= value2;
      } else if (operation == VectorOperation::kBitwiseAndAssign) {
        value1 &= value2;
      } else if (operation == VectorOperation::kRightShiftAssign) {
        value1 >>= value2;
      } else if (operation == VectorOperation::kLeftShiftAssign) {
        value1 <<= value2;
      } else if (operation == VectorOperation::kModulo) {
        value1 = value1 % value2;
      } else if (operation == VectorOperation::kBitwiseXor) {
        value1 = value1 ^ value2;
      } else if (operation == VectorOperation::kBitwiseOr) {
        value1 = value1 | value2;
      } else if (operation == VectorOperation::kBitwiseAnd) {
        value1 = value1 & value2;
      } else if (operation == VectorOperation::kRightShift) {
        value1 = value1 >> value2;
      } else if (operation == VectorOperation::kLeftShift) {
        value1 = value1 << value2;
      }
    }
  }
  SanityCheck(vector, value1);
}

template <typename T>
__device__ __host__ void PerformVectorOperation(VectorOperation operation, T* vector1,
                                                const T& vector2) {
  if (operation == VectorOperation::kIncrementPrefix) {
    ++(*vector1);
  } else if (operation == VectorOperation::kIncrementPostfix) {
    (*vector1)++;
  } else if (operation == VectorOperation::kDecrementPrefix) {
    --(*vector1);
  } else if (operation == VectorOperation::kDecrementPostfix) {
    (*vector1)--;
  } else if (operation == VectorOperation::kAddAssign) {
    *vector1 += vector2;
  } else if (operation == VectorOperation::kSubtractAssign) {
    *vector1 -= vector2;
  } else if (operation == VectorOperation::kMultiplyAssign) {
    *vector1 *= vector2;
  } else if (operation == VectorOperation::kDivideAssign) {
    *vector1 /= vector2;
  } else if (operation == VectorOperation::kAdd) {
    *vector1 = *vector1 + vector2;
  } else if (operation == VectorOperation::kSubtract) {
    *vector1 = *vector1 - vector2;
  } else if (operation == VectorOperation::kMultiply) {
    *vector1 = *vector1 * vector2;
  } else if (operation == VectorOperation::kDivide) {
    *vector1 = *vector1 / vector2;
  } else if (operation == VectorOperation::kEqual) {
    *vector1 = (*vector1 == vector2) ? 2 * *vector1 : 3 * *vector1;
  } else if (operation == VectorOperation::kNotEqual) {
    *vector1 = (*vector1 != vector2) ? 2 * *vector1 : 3 * *vector1;
  } else if (operation == VectorOperation::kSubscript) {
    constexpr size_t n_elements = sizeof(T) / sizeof(decltype(T().x));
    for (int i = 0; i < n_elements; ++i) {
      (*vector1)[i] = vector2[i];
    }
  } else {
    if constexpr (std::is_signed_v<decltype(T().x)>) {
      if (operation == VectorOperation::kNegate) {
        *vector1 = -(*vector1);
      }
    }
    if constexpr (std::is_integral_v<decltype(T().x)>) {
      if (operation == VectorOperation::kBitwiseNot) {
        *vector1 = ~(*vector1);
      } else if (operation == VectorOperation::kModuloAssign) {
        *vector1 %= vector2;
      } else if (operation == VectorOperation::kBitwiseXorAssign) {
        *vector1 ^= vector2;
      } else if (operation == VectorOperation::kBitwiseOrAssign) {
        *vector1 |= vector2;
      } else if (operation == VectorOperation::kBitwiseAndAssign) {
        *vector1 &= vector2;
      } else if (operation == VectorOperation::kRightShiftAssign) {
        *vector1 >>= vector2;
      } else if (operation == VectorOperation::kLeftShiftAssign) {
        *vector1 <<= vector2;
      } else if (operation == VectorOperation::kModulo) {
        *vector1 = *vector1 % vector2;
      } else if (operation == VectorOperation::kBitwiseXor) {
        *vector1 = *vector1 ^ vector2;
      } else if (operation == VectorOperation::kBitwiseOr) {
        *vector1 = *vector1 | vector2;
      } else if (operation == VectorOperation::kBitwiseAnd) {
        *vector1 = *vector1 & vector2;
      } else if (operation == VectorOperation::kRightShift) {
        *vector1 = *vector1 >> vector2;
      } else if (operation == VectorOperation::kLeftShift) {
        *vector1 = *vector1 << vector2;
      }
    }
  }
}

template <typename T>
__device__ __host__ void PerformVectorOperation(VectorOperation operation, T* vector,
                                                decltype(T().x) value) {
  if (operation == VectorOperation::kAddAssign) {
    *vector += value;
  } else if (operation == VectorOperation::kSubtractAssign) {
    *vector -= value;
  } else if (operation == VectorOperation::kMultiplyAssign) {
    *vector *= value;
  } else if (operation == VectorOperation::kDivideAssign) {
    *vector /= value;
  } else if (operation == VectorOperation::kAdd) {
    *vector = *vector + value;
  } else if (operation == VectorOperation::kSubtract) {
    *vector = *vector - value;
  } else if (operation == VectorOperation::kMultiply) {
    *vector = *vector * value;
  } else if (operation == VectorOperation::kDivide) {
    *vector = *vector / value;
  } else if (operation == VectorOperation::kEqual) {
    *vector = (*vector == value) ? 2 * *vector : 3 * *vector;
  } else if (operation == VectorOperation::kNotEqual) {
    *vector = (*vector != value) ? 2 * *vector : 3 * *vector;
  } else if (operation == VectorOperation::kSubscript) {
    constexpr size_t n_elements = sizeof(T) / sizeof(decltype(T().x));
    for (int i = 0; i < n_elements; ++i) {
      (*vector)[i] = value;
    }
  } else {
    if constexpr (std::is_integral_v<decltype(T().x)>) {
      if (operation == VectorOperation::kModulo) {
        *vector = *vector % value;
      } else if (operation == VectorOperation::kBitwiseXor) {
        *vector = *vector ^ value;
      } else if (operation == VectorOperation::kBitwiseOr) {
        *vector = *vector | value;
      } else if (operation == VectorOperation::kBitwiseAnd) {
        *vector = *vector & value;
      } else if (operation == VectorOperation::kRightShift) {
        *vector = *vector >> value;
      } else if (operation == VectorOperation::kLeftShift) {
        *vector = *vector << value;
      }
    }
  }
}

template <typename T, bool two_vectors = true>
T PerformVectorOperationHost(VectorOperation operation, decltype(T().x) value1,
                             decltype(T().x) value2) {
  T vector1{};
  MakeVectorType(&vector1, value1);

  if constexpr (two_vectors) {
    T vector2{};
    MakeVectorType(&vector2, value2);
    PerformVectorOperation(operation, &vector1, vector2);
  } else {
    PerformVectorOperation(operation, &vector1, value2);
  }

  return vector1;
}

template <typename T, bool two_vectors>
__global__ void VectorOperationKernel(VectorOperation operation, T* vector1, decltype(T().x) value1,
                                      decltype(T().x) value2) {
  MakeVectorType(vector1, value1);
  if constexpr (two_vectors) {
    T vector2{};
    MakeVectorType(&vector2, value2);
    PerformVectorOperation(operation, vector1, vector2);
  } else {
    PerformVectorOperation(operation, vector1, value2);
  }
}

template <typename T, bool two_vectors = true>
T PerformVectorOperationDevice(VectorOperation operation, decltype(T().x) value1,
                               decltype(T().x) value2) {
  T vector_h{};
  T* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(T)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(T), hipMemcpyHostToDevice));
  VectorOperationKernel<T, two_vectors><<<1, 1, 0, 0>>>(operation, vector_d, value1, value2);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));
  return vector_h;
}
