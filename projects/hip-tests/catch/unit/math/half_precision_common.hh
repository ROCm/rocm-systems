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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "unary_common.hh"
#include "binary_common.hh"
#include "ternary_common.hh"


/********** Unary **********/

#define MATH_UNARY_HP_KERNEL_DEF(func_name)                                                        \
  __global__ void func_name##_kernel(Float16* const ys, const size_t num_xs, Float16* const xs) {  \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define MATH_UNARY_HP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                        \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    UnaryHalfPrecisionTest(func_name##_kernel, ref_func, validator_builder);                       \
  }

#define MATH_UNARY_HP_TEST_DEF(func_name, ref_func)                                                \
  MATH_UNARY_HP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_UNARY_HP_VALIDATOR_BUILDER_DEF(func_name)                                             \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x)


/********** Binary **********/

#define MATH_BINARY_HP_KERNEL_DEF(func_name)                                                       \
  __global__ void func_name##_kernel(Float16* const ys, const size_t num_xs, Float16* const x1s,   \
                                     Float16* const x2s) {                                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define MATH_BINARY_HP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                       \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    BinaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                      \
  }

#define MATH_BINARY_HP_TEST_DEF(func_name, ref_func)                                               \
  MATH_BINARY_HP_TEST_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_BINARY_HP_VALIDATOR_BUILDER_DEF(func_name)                                            \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2)


/********** Ternary **********/

#define MATH_TERNARY_HP_KERNEL_DEF(func_name)                                                      \
  __global__ void func_name##_kernel(Float16* const ys, const size_t num_xs, Float16* const x1s,   \
                                     Float16* const x2s, Float16* const x3s) {                     \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                   \
    }                                                                                              \
  }

#define MATH_TERNARY_HP_TEST_DEF_IMPL(func_name, ref_func, validator_builder)                      \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    TernaryFloatingPointTest(func_name##_kernel, ref_func, validator_builder);                     \
  }

#define MATH_TERNARY_HP_TEST_DEF(func_name, ref_func, validator_builder)                           \
  MATH_TERNARY_HP_TEST_DEF_IMPL(func_name, ref_func, func_name##_validator_builder)

#define MATH_TERNARY_HP_VALIDATOR_BUILDER_DEF(func_name)                                           \
  static std::unique_ptr<MatcherBase<float>> func_name##_validator_builder(float target, float x1, \
                                                                           float x2, float x3)