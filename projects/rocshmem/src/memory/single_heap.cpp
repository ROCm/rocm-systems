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

#include "single_heap.hpp"

#include <sstream>

namespace rocshmem {

SingleHeap::SingleHeap() { }

void SingleHeap::malloc(void** ptr, size_t size) {
  strat_.alloc(reinterpret_cast<char**>(ptr), size);
}

__device__ void SingleHeap::malloc(void** ptr, size_t size) {}

void SingleHeap::free(void* ptr) {
  if (!ptr) {
    return;
  }
  strat_.free(reinterpret_cast<char*>(ptr));
}

__device__ void SingleHeap::free(void* ptr) {}

void* SingleHeap::realloc(void* ptr, size_t size) { return nullptr; }

void* SingleHeap::malign(size_t alignment, size_t size) { return nullptr; }

char* SingleHeap::get_base_ptr() { return heap_mem_.get_ptr(); }

size_t SingleHeap::get_size() { return heap_mem_.get_size(); }

size_t SingleHeap::get_used() { return strat_.get_used(); }

size_t SingleHeap::get_avail() { return get_size() - get_used(); }

}  // namespace rocshmem
