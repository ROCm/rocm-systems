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

#ifndef LIBRARY_SRC_CONTAINERS_ARRAY_HPP_
#define LIBRARY_SRC_CONTAINERS_ARRAY_HPP_

#include <hip/hip_runtime.h>

#include "index_strategy.hpp"
#include "strategies.hpp"
#include "memory_allocator.hpp"

namespace rocshmem {

template <typename TYPE>
class Array {
 public:
  /**
   * @brief
   *
   * @param[in] index
   *
   * @return
   */
  __host__ __device__ TYPE& operator[](size_t index);

  /**
   * @brief
   *
   * @param[in] index
   *
   * @return
   */
  __host__ __device__ const TYPE& operator[](size_t index) const;

  /**
   * @brief
   *
   * @return
   */
  __host__ __device__ size_t size() const;

  /**
   * @brief
   *
   * @param[in] v
   * @param[in] start_index
   * @param[in] length
   *
   * @return void
   */
  __host__ void fill(TYPE v, size_t start_index, size_t length);

  /**
   * @brief
   *
   * @param[in] v
   * @param[in] start_index
   * @param[in] length
   *
   * @return void
   */
  __device__ void fill(TYPE v, size_t start_index, size_t length);

  /**
   * @brief
   *
   * @return void
   */
  __host__ __device__ void zero();

  /**
   * @brief
   *
   * @param[in] other
   * @param[in] start_index
   * @param[in] length
   *
   * @return void
   */
  __host__ void copy(const Array* other, size_t start_index,
                     size_t length);  // NOLINT

  /**
   * @brief
   *
   * @param[in] other
   * @param[in] start_index
   * @param[in] length
   *
   * @return void
   */
  __device__ void copy(const Array* other, size_t start_index,  // NOLINT
                       size_t length);

  /**
   * @brief
   */
  Array() = default;

  /**
   * @brief
   *
   * @param[in] array_size
   * @param[in] allocator
   */
  Array(size_t array_size, MemoryAllocator allocator,
        const ObjectStrategy* strategy);

  /**
   * @brief
   */
  ~Array();

  /**
   * @brief
   *
   * @param[in] other
   */
  Array(const Array& other);

  /**
   * @brief
   *
   * @param[in] rhs
   *
   * @return
   */
  Array& operator=(const Array& rhs);

  /**
   * @brief
   */
  __device__ void zero_thread_dump();

  /**
   * @brief
   */
  __device__ void any_thread_dump();

 private:
  /**
   * @brief
   */
  __device__ void _dump();

  /**
   * @brief
   *
   * @note _allocator is required to be declared before '_array' in this
   * class because _allocator deallocates _array in class destructor.
   */
  MemoryAllocator _allocator{};

 protected:
  /**
   * @brief
   */
  TYPE* _array{nullptr};

  /**
   * @brief
   */
  size_t _size{0};

 private:
  /**
   * @brief
   */
  IndexStrategy _dev_idx{};

  /**
   * @brief
   */
  const size_t dump_num_data_per_line{8};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_ARRAY_HPP_
