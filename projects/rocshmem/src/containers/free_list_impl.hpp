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

#ifndef LIBRARY_SRC_CONTAINERS_FREE_LIST_IMPL_HPP_
#define LIBRARY_SRC_CONTAINERS_FREE_LIST_IMPL_HPP_

#include "free_list.hpp"

namespace rocshmem {

/*****************************************************************************
 ******************************* FREE LIST ***********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC>
FreeList<TYPE, ALLOC>::~FreeList() {}

template <typename TYPE, typename ALLOC>
void FreeList<TYPE, ALLOC>::deallocate_all_nodes() {
  // Deallocate any existing nodes
  while (head_ != nullptr) {
    auto temp = head_;
    head_ = temp->next;
    allocator_.deallocate(temp);
  }

  // The tail no longer points to any nodes
  tail_ = nullptr;

  // Deallocate all recycled nodes
  while (deallocated_nodes_ != nullptr) {
    auto temp = deallocated_nodes_;
    deallocated_nodes_ = temp->next;
    allocator_.deallocate(temp);
  }
}

template <typename TYPE, typename ALLOC>
FreeList<TYPE, ALLOC>::FreeList(const ALLOC& alloc)
    : allocator_{alloc},
      head_{nullptr},
      tail_{nullptr},
      deallocated_nodes_{nullptr} {}

template <typename TYPE, typename ALLOC>
template <class InputIt>
bool FreeList<TYPE, ALLOC>::push_back_range(InputIt first, InputIt last) {
  for (auto iter = first; iter != last; iter++) {
    auto key = *iter;
    const bool result = push_back(key);
    if (!result) {
      return false;
    }
  }
  return true;
}

template <typename TYPE, typename ALLOC>
__device__ bool FreeList<TYPE, ALLOC>::push_back(const TYPE& val) {
  TicketLockGuard<MutexType> guard(*mutex_.get());
  auto node = allocate_node();
  if (node == nullptr) {
    return false;
  }

  node->data = val;
  node->next = nullptr;

  insert_node_at_tail(node);

  return true;
}

template <typename TYPE, typename ALLOC>
__device__ bool FreeList<TYPE, ALLOC>::push_back(TYPE&& val) {
  return push_back(std::forward<const TYPE>(val));
}

template <typename TYPE, typename ALLOC>
__host__ bool FreeList<TYPE, ALLOC>::push_back(const TYPE& val) {
  auto node = allocate_node();
  if (node == nullptr) {
    return false;
  }

  node->data = val;
  node->next = nullptr;
  insert_node_at_tail(node);

  return true;
}
template <typename TYPE, typename ALLOC>
__host__ bool FreeList<TYPE, ALLOC>::push_back(TYPE&& val) {
  return push_back(std::forward<const TYPE>(val));
}

template <typename TYPE, typename ALLOC>
__device__ typename FreeList<TYPE, ALLOC>::PopBackResult
FreeList<TYPE, ALLOC>::pop_front() {
  TicketLockGuard<MutexType> guard(*mutex_.get());

  if (head_ == nullptr) {
    return {{}, false};
  }
  auto last_node = head_;
  head_ = head_->next;

  // if we removed all nodes, we should reset the tail
  if (head_ == nullptr) {
    tail_ = nullptr;
  }

  TYPE result{last_node->data};

  deallocate_node(last_node);

  return {result, true};
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_FREE_LIST_IMPL_HPP_
