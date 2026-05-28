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

#include "endian.hpp"

namespace rocshmem {

template <typename T>
__device__ void swap_endian_store(T *dst, const T val) {
  typedef union U {
    T val;
    uint8_t bytes[sizeof(T)];
  } union_type;
  union_type src;
  union_type dst_tmp;

  src.val = val;
  std::reverse_copy(src.bytes, src.bytes + sizeof(T), dst_tmp.bytes);
  *dst = dst_tmp.val;
}

template <>
__device__ void swap_endian_store(uint64_t *dst, const uint64_t val) {
  uint64_t new_val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
                     ((val >> 8) & 0x00FF00FF00FF00FFULL);

  new_val = ((new_val << 16) & 0xFFFF0000FFFF0000ULL) |
            ((new_val >> 16) & 0x0000FFFF0000FFFFULL);

  *dst = (new_val << 32) | (new_val >> 32);
}

template <>
__device__ void swap_endian_store(int64_t *dst, const int64_t val) {
  swap_endian_store(reinterpret_cast<uint64_t*>(dst), (const uint64_t)val);
}

template <>
__device__ void swap_endian_store(uint32_t *dst, const uint32_t val) {
  uint32_t new_val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);

  *dst = (new_val << 16) | (new_val >> 16);
}

template <>
__device__ void swap_endian_store(int32_t *dst, const int32_t val) {
  swap_endian_store(reinterpret_cast<uint32_t*>(dst), (const uint32_t)val);
}

template <>
__device__ void swap_endian_store(uint16_t *dst, const uint16_t val) {
  *dst = ((val << 8) & 0xFF00) | ((val >> 8) & 0x00FF);
}

template <>
__device__ void swap_endian_store(int16_t *dst, const int16_t val) {
  swap_endian_store(reinterpret_cast<uint16_t*>(dst), (const uint16_t)val);
}

}  // namespace rocshmem
