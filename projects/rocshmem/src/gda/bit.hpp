/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_BIT_HPP_
#define LIBRARY_SRC_GDA_BIT_HPP_

#include <limits>
#include <type_traits>
#include <hip/hip_runtime.h>

namespace rocshmem {
inline namespace _type_traits {
  template <typename T>
  struct is_unsigned_integer : std::bool_constant<std::is_integral_v<T> &&
                                                  std::is_same_v<T, std::make_unsigned_t<T>>> { };
  template <typename T>
  inline constexpr bool is_unsigned_integer_v = is_unsigned_integer<T>::value;
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
countl_zero(T val) noexcept {
  return __builtin_clzg(val, std::numeric_limits<T>::digits);
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
countl_one(T val) noexcept {
  return countl_zero(static_cast<T>(~val));
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
countr_zero(T val) noexcept {
  return __builtin_ctzg(val, std::numeric_limits<T>::digits);
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
countr_one(T val) noexcept {
  return countr_zero(static_cast<T>(~val));
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
bit_log2(T val) noexcept {
  return std::numeric_limits<T>::digits - 1 - countl_zero(val);
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, int>
bit_width(T val) noexcept {
  return val == 0 ? 0 : bit_log2(val) + 1;
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, T>
bit_ceil(T val) noexcept {
  if (val < 2)
    return 1;
  return static_cast<T>(T{1} << bit_width(static_cast<T>(val - 1)));
}

template <typename T>
constexpr inline __host__ __device__
std::enable_if_t<is_unsigned_integer_v<T>, T>
bit_floor(T val) noexcept {
  return val == 0 ? 0 : T{1} << bit_log2(val);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_BIT_HPP_
