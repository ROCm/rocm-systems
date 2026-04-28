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
#include "util.hpp"

#include "dlmalloc.hpp"
#include "default_allocator.hpp"

namespace rocshmem {

HIPAllocator *default_allocator_{nullptr};

SingleHeap::SingleHeap() {

  HIPAllocator *allocator = get_default_allocator();
  if (allocator->type == AllocatorTypeCoarsegrained) {
    heap_mem_ = new HeapMemoryType<HIPAllocatorCoarsegrained>(envvar::heap_size.get_value());
  } else if (allocator->type == AllocatorTypeFinegrained) {
    heap_mem_ = new HeapMemoryType<HIPAllocatorFinegrained>(envvar::heap_size.get_value());
  } else if (allocator->type == AllocatorTypeUncached) {
    heap_mem_ = new HeapMemoryType<HIPAllocatorUncached>(envvar::heap_size.get_value());
  }
#if defined USE_HEAP_DEVICE_VMM_POSIX
  else if (allocator->type == AllocatorTypeVMM) {
    heap_mem_ = new HeapMemoryType<HIPAllocatorVMMPosixFd>(envvar::heap_size.get_value());
  }
#endif
  else {
    printf("Unknown allocator type\n");
    abort();
  }
  assert(heap_mem_ != nullptr);

  if (heap_mem_->type_ == AllocatorTypeCoarsegrained) {
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorCoarsegrained>>(reinterpret_cast<HeapMemoryType<HIPAllocatorCoarsegrained> *>(heap_mem_));
  } else if (heap_mem_->type_ == AllocatorTypeFinegrained){
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorFinegrained>>(reinterpret_cast<HeapMemoryType<HIPAllocatorFinegrained> *>(heap_mem_));
  } else if (heap_mem_->type_ == AllocatorTypeUncached){
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorUncached>>(reinterpret_cast<HeapMemoryType<HIPAllocatorUncached> *>(heap_mem_));
  }
#if defined USE_HEAP_DEVICE_VMM_POSIX
  else if (heap_mem_->type_ == AllocatorTypeVMM){
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorVMMPosixFd>>(reinterpret_cast<HeapMemoryType<HIPAllocatorVMMPosixFd> *>(heap_mem_));
  }
#endif
  else {
    printf("Unknown allocator type\n");
    abort();
  }
}

SingleHeap::~SingleHeap() {
  if (strat_) {
    delete strat_;
    strat_ = nullptr;
  }
  if (heap_mem_) {
    delete heap_mem_;
    heap_mem_ = nullptr;
  }
}
void SingleHeap::malloc(void** ptr, size_t size) {
  strat_->alloc(reinterpret_cast<char**>(ptr), size);
}

__device__ void SingleHeap::malloc([[maybe_unused]] void** ptr, [[maybe_unused]] size_t size) {}

void SingleHeap::free(void* ptr) {
  if (!ptr) {
    return;
  }
  strat_->free(reinterpret_cast<char*>(ptr));
}

__device__ void SingleHeap::free([[maybe_unused]] void* ptr) {}

void* SingleHeap::realloc([[maybe_unused]] void* ptr, [[maybe_unused]] size_t size) { return nullptr; }

void* SingleHeap::malign([[maybe_unused]] size_t alignment, [[maybe_unused]] size_t size) { return nullptr; }

char* SingleHeap::get_base_ptr() { return heap_mem_->get_ptr(); }

size_t SingleHeap::get_size() { return heap_mem_->get_size(); }

size_t SingleHeap::get_used() { return strat_->get_used(); }

size_t SingleHeap::get_avail() { return get_size() - get_used(); }

}  // namespace rocshmem
