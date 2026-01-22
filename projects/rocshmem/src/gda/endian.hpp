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

#ifndef LIBRARY_SRC_GDA_ENDIAN_HPP_
#define LIBRARY_SRC_GDA_ENDIAN_HPP_

#include <type_traits>
#include <hip/hip_runtime.h>

namespace rocshmem {

// this is essentially std::byteswap from C++23
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
constexpr inline __host__ __device__ T byteswap(T val) {
  if constexpr (sizeof(T) == 1) {
    return val;
  } else if constexpr (sizeof(T) == 2) {
    return __builtin_bswap16(val);
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bswap32(val);
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bswap64(val);
  } else {
    // sizeof(T) to force this to be instantiation-dependent
    static_assert(sizeof(T) == 0, "byteswap not implemented for this type");
  }
}

namespace endian {
  enum class Order {
    Big = __ORDER_BIG_ENDIAN__,
    Little = __ORDER_LITTLE_ENDIAN__,
    Native = __BYTE_ORDER__
  };

  template <Order To, Order From, typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
  __host__ __device__ constexpr inline T convert(T val) {
    if constexpr (To == From) {
      return val;
    } else {
      return byteswap(val);
    }
  }

  template <Order From, typename T>
  __host__ __device__ constexpr inline T to_native(T val) {
    return convert<Order::Native, From, T>(val);
  }

  template <Order To, typename T>
  __host__ __device__ constexpr inline T from_native(T val) {
    return convert<To, Order::Native, T>(val);
  }

  template <typename T>
  __host__ __device__ constexpr inline T to_be(T val) {
    return convert<Order::Big, Order::Native, T>(val);
  }

  template <typename T>
  __host__ __device__ constexpr inline T from_be(T val) {
    return convert<Order::Native, Order::Big, T>(val);
  }

  template <typename T>
  __host__ __device__ constexpr inline T to_le(T val) {
    return convert<Order::Little, Order::Native, T>(val);
  }

  template <typename T>
  __host__ __device__ constexpr inline T from_le(T val) {
    return convert<Order::Native, Order::Little, T>(val);
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_ENDIAN_HPP_
