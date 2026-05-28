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

#include "memory_allocator.hpp"

#include <cassert>

#include "util.hpp"

namespace rocshmem {

MemoryAllocator::MemoryAllocator(hipError_t (*hip_alloc_fn)(void**, size_t,
                                                            unsigned),
                                 hipError_t (*hip_free_fn)(void*),
                                 unsigned flags)
    : _hip_alloc_with_flags(hip_alloc_fn),
      _hip_free(hip_free_fn),
      _flags(flags) {}

MemoryAllocator::MemoryAllocator(hipError_t (*hip_alloc_fn)(void**, size_t),
                                 hipError_t (*hip_free_fn)(void*))
    : _hip_alloc(hip_alloc_fn), _hip_free(hip_free_fn) {}

MemoryAllocator::MemoryAllocator(std::function<void*(size_t)> alloc_fn,
                                 std::function<void(void*)> free_fn)
    : _alloc(alloc_fn), _free(free_fn) {}

MemoryAllocator::MemoryAllocator(
    std::function<int(void**, size_t, size_t)> posix_align_fn,
    std::function<void(void*)> free_fn, size_t alignment)
    : _alloc_posix_memalign(posix_align_fn),
      _free(free_fn),
      _alignment(alignment) {}

void MemoryAllocator::allocate(void** void_ptr, size_t size) {
  assert(void_ptr);

  if (_alloc) {
    *(reinterpret_cast<char**>(void_ptr)) =
        reinterpret_cast<char*>(_alloc(size));
    assert(*reinterpret_cast<char**>(void_ptr));
    return;
  }
  if (_alloc_posix_memalign) {
    assert(_alignment);
    _alloc_posix_memalign(void_ptr, _alignment, size);
    assert(*reinterpret_cast<char**>(void_ptr));
    return;
  }
  if (_hip_alloc) {
    CHECK_HIP(_hip_alloc(void_ptr, size));
    return;
  }
  if (_hip_alloc_with_flags) {
    CHECK_HIP(_hip_alloc_with_flags(void_ptr, size, _flags));
    return;
  }

  assert(false);
}

void MemoryAllocator::deallocate(void* ptr) {
  if (!ptr) {
    return;
  }
  if (_free) {
    _free(ptr);
    return;
  }
  if (_hip_free) {
    CHECK_HIP(_hip_free(ptr));
    return;
  }

  assert(false);
}

bool MemoryAllocator::is_managed() { return _managed; }

}  // namespace rocshmem
