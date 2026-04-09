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

#ifndef LIBRARY_SRC_CONTAINERS_MATRIX_IMPL_HPP_
#define LIBRARY_SRC_CONTAINERS_MATRIX_IMPL_HPP_

#include "matrix.hpp"

#include <vector>
#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

template <typename TYPE>
Matrix<TYPE>::Matrix(size_t number_rows, size_t number_columns,
                     MemoryAllocator allocator, const IndexStrategy strategy)
    : _number_rows(number_rows),
      _number_columns(number_columns),
      _allocator(allocator),
      _dev_idx(strategy) {
  /*
   * Allocate the flattened c-array which contains the top-level
   * TYPE pointers.
   */
  size_t flat_c_array_dimensions = _number_rows * _number_columns;
  size_t size_bytes = flat_c_array_dimensions * sizeof(TYPE*);
  _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array), size_bytes);

  /*
   * Iterate through the flattened c-array and initialize each pointer
   * with a valid TYPE.
   */
  for (size_t i = 0; i < flat_c_array_dimensions; i++) {
    _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array[i]),
                        sizeof(TYPE));

    if constexpr (one_generic_constructor_parameter<TYPE>::value &&
                  requires_internal_allocator<TYPE>::value) {
      /*
       * Do not invoke constructor since type traits do not match.
       */
    } else if constexpr (requires_internal_allocator<TYPE>::value) {
      /*
       * Construct the TYPE with placement new.
       */
      new (_flat_c_array[i]) TYPE(allocator, strategy);
    }
  }
}

template <typename TYPE>
Matrix<TYPE>::Matrix(size_t number_rows, size_t number_columns,
                     size_t TYPE_constructor_param, MemoryAllocator allocator,
                     const IndexStrategy strategy)
    : _number_rows(number_rows),
      _number_columns(number_columns),
      _allocator(allocator),
      _dev_idx(strategy) {
  /*
   * Allocate the flattened c-array which contains the top-level
   * TYPE pointers.
   */
  size_t flat_c_array_dimensions = _number_rows * _number_columns;
  size_t size_bytes = flat_c_array_dimensions * sizeof(TYPE*);
  _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array), size_bytes);

  /*
   * Iterate through the flattened c-array and initialize each pointer
   * with a valid TYPE.
   */
  for (size_t i = 0; i < flat_c_array_dimensions; i++) {
    _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array[i]),
                        sizeof(TYPE));

    /*
     * If type traits match, construct the TYPE with placement new.
     */
    if constexpr (one_generic_constructor_parameter<TYPE>::value &&
                  requires_internal_allocator<TYPE>::value) {
      new (_flat_c_array[i]) TYPE(TYPE_constructor_param, allocator, strategy);
    } else if constexpr (one_generic_constructor_parameter<TYPE>::value) {
      new (_flat_c_array[i]) TYPE(TYPE_constructor_param, strategy);
    }
  }
}

template <typename TYPE>
Matrix<TYPE>::Matrix(size_t number_rows, size_t number_columns,
                     std::vector<size_t> TYPE_constructor_param,
                     MemoryAllocator allocator, const IndexStrategy strategy)
    : _number_rows(number_rows),
      _number_columns(number_columns),
      _allocator(allocator),
      _dev_idx(strategy->index_strategy_three) {
  /*
   * Allocate the flattened c-array which contains the top-level
   * TYPE pointers.
   */
  size_t flat_c_array_dimensions{_number_rows * _number_columns};
  size_t size_bytes{flat_c_array_dimensions * sizeof(TYPE*)};
  _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array), size_bytes);

  /*
   * Check the TYPE_constructor_param vector to see if it has enough
   * entries to fully initialize the matrix.
   */
  assert(TYPE_constructor_param.size() == flat_c_array_dimensions);

  /*
   * Iterate through the flattened c-array and initialize each pointer
   * with a valid TYPE.
   */
  for (size_t i = 0; i < flat_c_array_dimensions; i++) {
    _allocator.allocate(reinterpret_cast<void**>(&_flat_c_array[i]),
                        sizeof(TYPE));

    /*
     * If type traits match, construct the TYPE with placement new.
     */
    if constexpr (one_generic_constructor_parameter<TYPE>::value &&
                  requires_internal_allocator<TYPE>::value) {
      new (_flat_c_array[i])
          TYPE(TYPE_constructor_param[i], allocator, strategy);
    } else if constexpr (one_generic_constructor_parameter<TYPE>::value) {
      new (_flat_c_array[i]) TYPE(TYPE_constructor_param[i], strategy);
    }
  }
}

template <typename TYPE>
Matrix<TYPE>::~Matrix() {
  if (_flat_c_array) {
    /*
     * Free internal TYPE instances.
     */
    size_t flat_c_array_dimensions = _number_rows * _number_columns;
    for (size_t i = 0; i < flat_c_array_dimensions; i++) {
      if (_flat_c_array[i]) {
        _flat_c_array[i]->~TYPE();
        _allocator.deallocate(_flat_c_array[i]);
      }
    }

    /*
     * Free top-level flat c-array.
     */
    _allocator.deallocate(_flat_c_array);
  }
}

template <typename TYPE>
__host__ __device__ TYPE* Matrix<TYPE>::access(size_t row_index,
                                               size_t col_index) {
  assert(row_index < _number_rows);
  assert(col_index < _number_columns);
  size_t offset = row_index * _number_columns + col_index;
  return _flat_c_array[offset];
}

template <typename TYPE>
__device__ TYPE* Matrix<TYPE>::access(size_t col_index) {
  auto row_index = _dev_idx.start();
  assert(row_index < _number_rows);
  assert(col_index < _number_columns);
  size_t offset = row_index * _number_columns + col_index;
  return _flat_c_array[offset];
}

template <typename TYPE>
__device__ TYPE* Matrix<TYPE>::access() {
  auto row_index = 0;
  auto col_index = _dev_idx.start();
  assert(row_index < _number_rows);
  assert(col_index < _number_columns);
  size_t offset = row_index * _number_columns + col_index;
  return _flat_c_array[offset];
}

template <typename TYPE>
__host__ __device__ size_t Matrix<TYPE>::rows() const {
  return _number_rows;
}

template <typename TYPE>
__host__ __device__ size_t Matrix<TYPE>::columns() const {
  return _number_columns;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_MATRIX_IMPL_HPP_
