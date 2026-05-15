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

#ifndef LIBRARY_SRC_CONTAINERS_MATRIX_HPP_
#define LIBRARY_SRC_CONTAINERS_MATRIX_HPP_

#include "index_strategy.hpp"
#include "memory_allocator.hpp"

#include <vector>

#include <hip/hip_runtime.h>

namespace rocshmem {

template <typename T>
struct requires_internal_allocator {
  static const bool value{false};
};

template <typename T>
struct one_generic_constructor_parameter {
  static const bool value{false};
};

template <typename TYPE>
class Matrix {
 public:
  /**
   * @brief
   *
   * @param[in] row_index
   * @param[in] col_index
   *
   * @return
   */
  __host__ __device__ TYPE* access(size_t row_index, size_t col_index);

  /**
   * @brief
   *
   * @param[in] col_index
   *
   * @return
   */
  __device__ TYPE* access(size_t col_index);

  /**
   * @brief
   *
   * @param[in] col_index
   *
   * @return
   */
  __device__ TYPE* access();

  /**
   * @brief
   *
   * @return
   */
  __host__ __device__ size_t rows() const;

  /**
   * @brief
   *
   * @return
   */
  __host__ __device__ size_t columns() const;

 protected:
  /**
   * @brief
   */
  Matrix() = default;

  /**
   * @brief
   *
   * @param[in] number_rows
   * @param[in] number_columns
   * @param[in] allocator
   * @param[in] strategy
   */
  Matrix(size_t number_rows, size_t number_columns, MemoryAllocator allocator,
         const IndexStrategy index_strat);

  /**
   * @brief
   *
   * @param[in] number_rows
   * @param[in] number_columns
   * @param[in] TYPE_constructor_param
   * @param[in] allocator
   * @param[in] strategy
   */
  Matrix(size_t number_rows, size_t number_columns,
         size_t TYPE_constructor_param, MemoryAllocator allocator,
         const IndexStrategy index_strat);

  /**
   * @brief
   *
   * @param[in] number_rows
   * @param[in] number_columns
   * @param[in] TYPE_constructor_param
   * @param[in] allocator
   * @param[in] strategy
   */
  Matrix(size_t number_rows, size_t number_columns,
         std::vector<size_t> TYPE_constructor_param, MemoryAllocator allocator,
         const IndexStrategy index_strat);

  /**
   * @brief
   */
  ~Matrix();

 private:
  /**
   * @brief
   */
  size_t _number_rows{0};

  /**
   * @brief
   */
  size_t _number_columns{0};

  /**
   * @brief
   */
  MemoryAllocator _allocator{};

  /**
   * @brief
   */
  TYPE** _flat_c_array{nullptr};

  /**
   * @brief
   */
  IndexStrategy _dev_idx{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_MATRIX_HPP_
