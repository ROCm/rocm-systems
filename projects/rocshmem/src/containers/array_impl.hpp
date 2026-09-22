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

#ifndef LIBRARY_SRC_CONTAINERS_ARRAY_IMPL_HPP_
#define LIBRARY_SRC_CONTAINERS_ARRAY_IMPL_HPP_

#include "array.hpp"
#include "constants.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

extern __constant__ int* GLOBAL_DEVICE_PRINT_LOCK;

template <typename TYPE>
Array<TYPE>::Array(size_t array_size, MemoryAllocator allocator,
                   const ObjectStrategy* strategy)
    : _allocator(allocator),
      _size(array_size),
      _dev_idx(strategy->index_strategy_two) {
  _allocator.allocate(reinterpret_cast<void**>(&_array),
                      array_size * sizeof(TYPE));
}

template <typename TYPE>
Array<TYPE>::~Array() {
  if (_array) {
    _allocator.deallocate(_array);
  }
}

template <typename TYPE>
Array<TYPE>::Array(const Array<TYPE>& other) {
  _size = other._size;

  _allocator.allocate(reinterpret_cast<void**>(&_array), _size * sizeof(TYPE));

  memcpy(_array, other._array, _size * sizeof(TYPE));
}

template <typename TYPE>
Array<TYPE>& Array<TYPE>::operator=(const Array<TYPE>& rhs) {
  if (this == &rhs) {
    return *this;
  }

  if (_array) {
    _allocator.deallocate(_array);
  }

  _size = rhs._size;

  _allocator = rhs._allocator;

  _allocator.allocate(reinterpret_cast<void**>(&_array), _size * sizeof(TYPE));
  assert(_array);

  memcpy(_array, rhs._array, _size * sizeof(TYPE));

  return *this;
}

template <typename TYPE>
__host__ __device__ TYPE& Array<TYPE>::operator[](size_t index) {
  assert(index < _size);
  return _array[index];
}

template <typename TYPE>
__host__ __device__ const TYPE& Array<TYPE>::operator[](size_t index) const {
  assert(index < _size);
  return _array[index];
}

template <typename TYPE>
__host__ __device__ size_t Array<TYPE>::size() const {
  return _size;
}

template <typename TYPE>
__host__ void Array<TYPE>::fill(TYPE v, size_t start_index, size_t length) {
  for (size_t i = start_index; i < start_index + length; i++) {
    (*this)[i] = v;
  }
}

template <typename TYPE>
__device__ void Array<TYPE>::fill(TYPE v, size_t start_index, size_t length) {
  size_t i = start_index + _dev_idx.start();
  while ((i < start_index + length) && (i < _size)) {
    (*this)[i] = v;
    i = _dev_idx.next(i);
  }
}

template <typename TYPE>
__host__ __device__ void Array<TYPE>::zero() {
  fill({0, 0}, 0, size());
}

template <typename TYPE>
__host__ void Array<TYPE>::copy(const Array<TYPE>* other,
                                size_t start_index,  // NOLINT
                                size_t length) {
  assert(other);
  for (size_t i = start_index; i < start_index + length; i++) {
    (*this)[i] = (*other)[i];
  }
}

template <typename TYPE>
__device__ void Array<TYPE>::copy(const Array<TYPE>* other,  // NOLINT
                                  size_t start_index, size_t length) {
  assert(other);
  for (size_t i = start_index + _dev_idx.start(); i < start_index + length;
       i = _dev_idx.next(i)) {
    (*this)[i] = (*other)[i];
  }
}

template <typename TYPE>
__device__ void Array<TYPE>::zero_thread_dump() {
  Identity id{};
  if (id.global_thread_id() == 0) {
    _dump();
  }
}

template <typename TYPE>
__device__ void Array<TYPE>::any_thread_dump() {
  Identity id{};
  for (int i = 0; i < WF_SIZE; i++) {
    if ((id.local_thread_id() % WF_SIZE) == i) {
      while (atomicCAS(GLOBAL_DEVICE_PRINT_LOCK, 0, 1) == 1) {
      }

      printf("(thread %lu)\n", id.global_thread_id());
      _dump();

      *GLOBAL_DEVICE_PRINT_LOCK = 0;
    }
  }
}

template <typename TYPE>
__device__ void Array<TYPE>::_dump() {
  Thread_Contiguous_Block_Agnostic idx(_size);
  for (size_t i = idx.start(); i < idx.end(); i = idx.next(i)) {
    /*
     * Limit the number of printed elements per line.
     */
    if (i % dump_num_data_per_line == 0) {
      printf("\n");
      printf("%10lu ", i);
    }

    /*
     * Print the data for this index.
     */
    dump((*this)[i]);
  }
  printf("\n");
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_ARRAY_IMPL_HPP_
