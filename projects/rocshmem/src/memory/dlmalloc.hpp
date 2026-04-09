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

#ifndef LIBRARY_SRC_MEMORY_DLMALLOC_HPP_
#define LIBRARY_SRC_MEMORY_DLMALLOC_HPP_

#include <cassert>
#include <map>

#include "constants.hpp"
#include "shmem_allocator_strategy.hpp"

/**
 * @file dlmalloc.hpp
 *
 * @brief Contains an allocator strategy for the heap using dlmalloc.
 *
 * This strategy uses DLMalloc to allocate/free from the symmetric heap
 *
 */

namespace rocshmem {


/*
 * @brief an encapsulation class for the C-static functions inherited from dlmalloc
 *
 * @note only a subset of dlmalloc is exposed, not intended for external use
 *
 * @rationale static functions cannot be defined in the templated class DLAllocatorStategy
 */
class DLMalloc {
 public:
  typedef void* mspace;
  static size_t destroy_mspace(mspace msp);
  static mspace create_mspace_with_base(void* base, size_t capacity, int locked);
  static void* mspace_malloc(mspace msp, size_t bytes);
  static void mspace_free(mspace msp, void* mem);
  static void* mspace_memalign(mspace msp, size_t alignment, size_t bytes);
  static size_t mspace_footprint(mspace msp);
  static size_t mspace_max_footprint(mspace msp);
  static size_t mspace_avail(mspace msp);
  static size_t mspace_used(mspace msp);
};

template <typename HM_T>
class DLAllocatorStrategy : public ShmemAllocatorStrategy {

public:
  /**
   * @brief Required for default construction of other objects
   *
   * @note Not intended for direct usage.
   */
  DLAllocatorStrategy() = default;

  /**
   * @brief Primary constructor type
   *
   * Construct the dlmalloc mspace
   *
   * @param[in] Raw pointer to heap memory type
   */
  explicit DLAllocatorStrategy(HM_T* heap_mem) {
    mspace_ = DLMalloc::create_mspace_with_base(heap_mem->get_ptr(), heap_mem->get_size(), false);
  }

  /**
   * @brief Primary destructor
   *
   * Destroy the dlmalloc mspace
   */
  ~DLAllocatorStrategy() {
    if(mspace_) {
      DLMalloc::destroy_mspace(mspace_);
    }
  }

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  void alloc(char** ptr, size_t request_size) override {
    assert(ptr);
    *ptr = nullptr;

    if (!request_size) {
      return;
    }
    *ptr = static_cast<char*>(DLMalloc::mspace_malloc(mspace_, request_size));
  }

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   *
   * @note Not implemented
   */
  __device__ void alloc([[maybe_unused]] char** ptr,
                        [[maybe_unused]] size_t request_size) override {}

  /**
   * @brief Frees memory from the heap
   *
   * Released memory is tracked by bookkeeping structures within this class.
   *
   * @param[in] Raw pointer to heap memory
   *
   */
  void free(char* ptr) override {
    DLMalloc::mspace_free(mspace_, ptr);
  }

  /**
   * @brief Frees memory from the heap
   *
   * Released memory is tracked by bookkeeping structures within this class.
   *
   * @param[in] Raw pointer to heap memory
   *
   * @note Not implemented
   */
  __device__ void free([[maybe_unused]] char* ptr) override {}

  /**
   * @brief Used heap memory
   *
   * @return memory size
   *
   * @note The used size may be larger than the sum of the user allocation sizes
   * (due to chunk tracking overhead and alignment).
   *
   */
  size_t get_used() override {
    size_t size{0};
    size = DLMalloc::mspace_used(mspace_);
    return size;
  }

  /**
   * @brief Available heap memory
   *
   * @return memory size
   *
   * @note The available size may be smaller than the total heap size minus the sum
   * of user allocation sizes (due to chunk tracking overhead and alignment).
   */
  size_t get_avail() {
    size_t size{0};
    size = DLMalloc::mspace_avail(mspace_);
    return size;
  }

 private:
  DLMalloc::mspace mspace_{nullptr};
};


}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_DLMALLOC_HPP_
