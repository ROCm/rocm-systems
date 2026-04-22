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

#ifndef LIBRARY_SRC_MEMORY_SHMEM_ALLOCATOR_STRATEGY_HPP_
#define LIBRARY_SRC_MEMORY_SHMEM_ALLOCATOR_STRATEGY_HPP_

/**
 * @file shmem_allocator_strategy.hpp
 *
 * @brief Strategy design pattern for SHMEM symmetric heap allocator.
 */

namespace rocshmem {

class ShmemAllocatorStrategy {
 public:
  /**
   * @brief Default constructor
   */
  ShmemAllocatorStrategy() = default;

  /**
   * @brief Default destructor
   */
  virtual ~ShmemAllocatorStrategy() = default;

  /**
   * @brief Default copy construction
   */
  ShmemAllocatorStrategy(const ShmemAllocatorStrategy& other) = default;

  /**
   * @brief Disable copy assignment
   */
  ShmemAllocatorStrategy& operator=(const ShmemAllocatorStrategy& other) =
      default;

  /**
   * @brief Allocates memory from the symmetric heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  virtual void alloc(char** ptr, size_t request_size) = 0;

  /**
   * @brief Allocates memory from the symmetric heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  __device__ virtual void alloc(char** ptr, size_t request_size) = 0;

  /**
   * @brief Frees memory from the symmetric heap
   *
   * @param[in] Raw pointer to symmetric heap memory
   */
  virtual void free(char* ptr) = 0;

  /**
   * @brief Frees memory from the symmetric heap
   *
   * @param[in] Raw pointer to symmetric heap memory
   */
  __device__ virtual void free(char* ptr) = 0;

  /**
   * @brief Used heap memory
   *
   * @return memory size
   *
   * @note The used size may be larger than the sum of the user allocation sizes
   * (due to chunk tracking overhead and alignment).
   *
   */
  virtual size_t get_used() = 0;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_SHMEM_ALLOCATOR_STRATEGY_HPP_
