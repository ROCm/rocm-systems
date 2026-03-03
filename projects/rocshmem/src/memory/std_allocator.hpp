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

#ifndef LIBRARY_SRC_MEMORY_STD_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_STD_ALLOCATOR_HPP_

#include <hip/hip_runtime_api.h>

#include "default_allocator.hpp"

namespace rocshmem {
  template <class T>
  class StdAllocatorHIP {
  public:
    typedef T value_type;

    StdAllocatorHIP()
    {
      allocator_ = get_default_allocator();
    }

    template <class U>
    constexpr StdAllocatorHIP(const StdAllocatorHIP<U>&) noexcept
    {
      allocator_ = get_default_allocator();
    }

    [[nodiscard]] T* allocate(size_t n) {
      if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
	throw std::bad_array_new_length();
      }

      T* p{nullptr};
      allocator_->allocate(reinterpret_cast<void**>(&p), n * sizeof(T));
      if (p) {
	return p;
      }

      throw std::bad_alloc();
    }

    void deallocate(T* p, [[maybe_unused]] size_t n) noexcept {
      allocator_->deallocate(p);
    }

  private:
    HIPAllocator *allocator_{nullptr};
  };

  template <class T, class U>
  bool operator==(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
    return true;
  }

  template <class T, class U>
  bool operator!=(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
    return false;
  }
  
}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_STD_ALLOCATOR_HPP_
