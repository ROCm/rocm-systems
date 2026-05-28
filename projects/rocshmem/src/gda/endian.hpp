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

#include <hip/hip_runtime.h>

namespace rocshmem {

template <typename T>
__device__ void swap_endian_store(T *dst, const T val);

template <>
__device__ void swap_endian_store(uint64_t *dst, const uint64_t val);

template <>
__device__ void swap_endian_store(int64_t *dst, const int64_t val);

template <>
__device__ void swap_endian_store(uint32_t *dst, const uint32_t val);

template <>
__device__ void swap_endian_store(int32_t *dst, const int32_t val);

template <>
__device__ void swap_endian_store(uint16_t *dst, const uint16_t val);

template <>
__device__ void swap_endian_store(int16_t *dst, const int16_t val);

template <typename T>
__device__ T swap_endian_val(const T val) {
  T dst;
  swap_endian_store(&dst, val);
  return dst;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_ENDIAN_HPP_
