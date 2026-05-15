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

#ifndef LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_
#define LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_

#include <hip/hip_runtime.h>

#include "memory/hip_allocator.hpp"
#include "sync/abql_block_mutex.hpp"

namespace rocshmem {

// Forward declaration of the proxy.
template <typename ALLOCATOR, typename TYPE>
class FreeListProxy;

/*****************************************************************************
 ******************************* FREE LIST ***********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC = HIPDefaultFinegrainedAllocator>
class FreeList {
  friend class FreeListProxy<ALLOC, TYPE>;

  using MutexProxyType = ABQLBlockMutexProxy<ALLOC>;
  using MutexType = ABQLBlockMutex;

  struct Node {
    TYPE data;
    Node* next{nullptr};
  };

  struct PopBackResult {
    TYPE value;
    bool success;
  };

  /**
   * @brief A lock guard for ticket-based locks that follows the design of
   * `std::lock_guard`.
   *
   * @tparam MUTEX The type of the ticket-based mutex to lock.
   */
  template <typename MUTEX>
  struct TicketLockGuard {
    /**
     * @brief Constructs the `TicketLockGuard` and locks the mutex.
     *
     * @param m Mutex to take ownership of.
     */
    __device__ explicit TicketLockGuard(MUTEX& m) : mutex_{m} {
      ticket_ = mutex_.lock();
      __threadfence();
    }

    /**
     * @brief Lock guards are not copyable
     */
    __device__ TicketLockGuard(const TicketLockGuard&) = delete;

    /**
     * @brief Lock guards are not moveable
     */
    __device__ TicketLockGuard(TicketLockGuard&&) = delete;

    /**
     * @brief Destructor the unlocks the mutex.
     */
    __device__ ~TicketLockGuard() {
      __threadfence();
      mutex_.unlock(ticket_);
    }

   private:
    using TicketT = uint64_t;
    MUTEX& mutex_;
    TicketT ticket_;
  };

 public:
  /**
   * @brief Construct a new Free List object
   *
   * @param alloc Allocator to use for free list nodes allocations.
   */
  explicit FreeList(const ALLOC& alloc = ALLOC());

  /**
   * @brief Constructs a FreeList object with contents from a range of elements
   * defined by [`first`, `last`).
   *
   * @tparam InputIt Iterator type of the elements to store in the free-list.
   * @param first First element in the range defining the input elements.
   * @param last Element after last that defines the input elements range.
   * @param alloc Allocator to use for allocating internal structures of the
   * free list.
   */
  template <class InputIt>
  FreeList(InputIt first, InputIt last, const ALLOC& alloc = ALLOC());

  /**
   * @brief Pushes a range of elements defined by [`first`, `last`).
   *
   * @tparam InputIt Iterator type of the elements to store in the free-list.
   * @param first First element in the range defining the input elements.
   * @param last Element after last that defines the input elements range.
   *
   * @return @c true if success, or @c false otherwise.
   */
  template <class InputIt>
  __host__ bool push_back_range(InputIt first, InputIt last);

  /**
   * @brief Destroy the Free List object
   */
  ~FreeList();

  /**
   * @brief  Inserts new element at the end of the FreeList.
   *
   * The element goes into the container right after its last
   * element. The content of val is copied (or moved) to the inserted
   * element.
   *
   * @note Host-side API is not thread safe.
   *
   * @param val The value to insert in the FreeList.
   * @return @c true if the operation succeed, and @c false otherwise.
   */
  __device__ bool push_back(const TYPE& val);

  /// @copydoc bool FreeList<TYPE, ALLOC>::push_back(const TYPE&)
  __device__ bool push_back(TYPE&& val);

  /// @copydoc bool FreeList<TYPE, ALLOC>::push_back(const TYPE&)
  __host__ bool push_back(const TYPE& val);

  /// @copydoc bool FreeList<TYPE, ALLOC>::push_back(const TYPE&)
  __host__ bool push_back(TYPE&& val);

  /**
   * @brief Removes the first element in FreeList, reducing its size by one.
   *
   * @return An object with two fields `value` and `success`. `success` is a
   * boolean indicating if the operation succeeded, and if the operation
   * succeeded, the `value` field contains the popped value.
   */
  __device__ PopBackResult pop_front();

 private:
  /**
   * @brief Deallocates all memory that was dynamically allocated using the free
   * list.
   */
  void deallocate_all_nodes();

  /**
   * @brief Allocates a node using the host-side allocator or the recycled
   * pointers on the device.
   *
   * @note The device-side API assumes the data structure is protected by a
   * lock.
   *
   * @return A pointer to the allocated node.
   */
  __host__ Node* allocate_node() {
    Node* node;
    allocator_.allocate((void**)(&node), sizeof(Node));
    return node;
  };

  /// @copydoc FreeList<TYPE, ALLOC>::Node* FreeList<TYPE,
  /// ALLOC>::allocate_node(const TYPE&)
  __device__ Node* allocate_node() {
    Node* node = deallocated_nodes_;
    if (node != nullptr) {
      deallocated_nodes_ = node->next;
    }
    return node;
  };

  /**
   * @brief Appends a node to the tail of the free list.
   *
   * @param node Node to append to the list.
   */
  __host__ __device__ void insert_node_at_tail(Node* node) {
    if (tail_ != nullptr) {
      tail_->next = node;
    }
    tail_ = node;

    // if the list is empty, set the head_ to the first node
    if (head_ == nullptr) {
      head_ = node;
    }
  }

  /**
   * @brief Device-side node deallocation that inserts the deallocated node into
   * a linked-list of pointers to reuse on the device.
   *
   * @note Assumes structure is protected by a lock.
   */
  __device__ void deallocate_node(Node* node) {
    // append the node to the head of the linked list
    node->next = deallocated_nodes_;
    deallocated_nodes_ = node;
  };

  /**
   * @brief Internal memory allocator used to create list nodes.
   */
  MemoryAllocator allocator_{};

  /**
   * @brief First element in the list.
   */
  Node* head_{};

  /**
   * @brief Last element in the list.
   */
  Node* tail_{};

  /**
   * @brief A linked-list of deallocated nodes.
   */
  Node* deallocated_nodes_{};

  /**
   * @brief Mutex protecting the free-list mutations.
   */
  MutexProxyType mutex_;
};

template <typename ALLOCATOR, typename TYPE>
class FreeListProxy {
  using FreeListT = FreeList<TYPE, ALLOCATOR>;
  using ProxyT = DeviceProxy<ALLOCATOR, FreeListT>;

 public:
  __host__ __device__ FreeListT* get() { return proxy_.get(); }

  FreeListProxy(size_t num_elems = 1) : proxy_{num_elems} {
    new (proxy_.get()) FreeListT();
  }

  FreeListProxy(const FreeListProxy& other) = delete;

  FreeListProxy& operator=(const FreeListProxy& other) = delete;

  FreeListProxy(FreeListProxy&& other) = default;

  FreeListProxy& operator=(FreeListProxy&& other) = default;

  ~FreeListProxy() {
    auto free_list = proxy_.get();
    free_list->deallocate_all_nodes();
    free_list->~FreeList();
  }

 private:
  ProxyT proxy_{};
};
}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_
