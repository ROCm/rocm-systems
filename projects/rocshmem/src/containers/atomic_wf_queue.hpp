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

#ifndef LIBRARY_SRC_CONTAINERS_ATOMIC_WF_QUEUE_HPP_
#define LIBRARY_SRC_CONTAINERS_ATOMIC_WF_QUEUE_HPP_

#include <hip/hip_runtime.h>

#include "memory/hip_allocator.hpp"
#include "sync/abql_block_mutex.hpp"
#include "util.hpp"

namespace rocshmem {

/*****************************************************************************
 ******************************* WAVE FREE LIST ******************************
 *****************************************************************************/

template <typename TYPE, typename ALLOCATOR = HIPDefaultFinegrainedAllocator>
class AtomicWFQueue {

  using MutexProxyType = ABQLBlockMutexProxy<ALLOCATOR>;
  using MutexType = ABQLBlockMutex;

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
   * @brief Construct a new AtomicWFQueue object
   *
   * @param allocator Allocator to use for allocating internal structures of the
   * AtomicWFQueue.
   */
  explicit AtomicWFQueue(const ALLOCATOR& allocator = ALLOCATOR());

  /**
   * @brief Destroy the AtomicWFQueue object
   */
  ~AtomicWFQueue();

  /**
   * @brief Enqueues an element into the AtomicWFQueue.
   *
   * This function inserts the specified value at the position indicated by 
   * the `tail_` of the AtomicWFQueue and increases the AtomicWFQueue size
   * by one. The enqueue operation follows a first-come, first-serve
   * execution order.
   *
   * @param val The value to be inserted into the AtomicWFQueue.
   */
  __device__ void enqueue(const TYPE& val);

  /**
   * @brief Dequeues an element from the AtomicWFQueue.
   *
   * This function dequeues the element pointed to by the `head_` of the
   * AtomicWFQueue and decreases the AtomicWFQueue size by one. If the
   * AtomicWFQueue is empty, the function waits until an element becomes
   * available. The dequeue operation follows a first-come, first-serve
   * execution order.
   *
   * @return The dequeued element from the AtomicWFQueue.
   */
  __device__ TYPE dequeue();

   /**
   * @brief Inserts a new element at the end of the AtomicWFQueue.
   *
   * This function adds the specified value to the end of the AtomicWFQueue,
   * updating the `tail_` and `curr_size_` accordingly. It is intended for
   * initializing the AtomicWFQueue with initial values.
   *
   * @note This function is not thread-safe and should only be used during 
   *       the AtomicWFQueue initialization phase or in scenarios where thread
   *       safety is not a concern.
   *
   * @param val The value to be inserted into the AtomicWFQueue.
   */
  __host__ void push(const TYPE& val);

  /**
   * @brief Allocates and initializes the AtomicWFQueue.
   *
   * This function allocates memory for the AtomicWFQueue with the specified
   * size and initializes the AtomicWFQueue's head, tail, current size, and
   * maximum size variables to their appropriate starting values.
   *
   * @param size The maximum number of elements the AtomicWFQueue can hold.
   */
  __host__ void allocate_queue(unsigned int size);

  /**
   * @brief Deallocates the AtomicWFQueue and resets its internal variables.
   *
   * This function frees the memory allocated for the AtomicWFQueue and resets
   * the AtomicWFQueue's internal variables such as head, tail, current size,
   * and maximum size to their default or zero-initialized values.
   */
  __host__ void deallocate_queue();

  /**
   * @brief Retrieves the logical lane ID of the calling thread.
   *
   * This function returns the active logical lane ID of the current thread
   * within the wavefront. The logical lane ID uniquely identifies
   * the thread's position among active threads in the wavefront.
   *
   * @return The logical lane ID of the active thread within the wavefront.
   */
  __device__ unsigned int active_logical_lane_id();

  /**
   * @brief Broadcasts a value to other threads in the wavefront.
   *
   * This function broadcasts the specified value to all active threads
   * in the wavefront. If `lowest_active` is true, the value is broadcasted
   * from the thread with the lowest active lane ID.
   *
   * @param lowest_active If true, broadcasting starts from the lowest 
   *                      active thread in the wavefront.
   * @param val The value to be broadcasted.
   *
   * @return The broadcasted value received by each thread in the wavefront.
   */
  __device__ TYPE broadcast_lds(bool lowest_active, TYPE val);

  /**
   * @brief Retrieves the maximum capacity of the AtomicWFQueue.
   *
   * This function returns the total size of the AtomicWFQueue, representing
   * the maximum number of elements it can hold.
   *
   * @return The maximum capacity of the AtomicWFQueue.
   */
  __host__ __device__ int get_queue_size() {
    return size_;
  }

  /**
   * @brief Retrieves the current number of elements in the AtomicWFQueue.
   *
   * This function returns the current size of the AtomicWFQueue, representing
   * the total number of elements currently stored.
   *
   * @return The current number of elements in the AtomicWFQueue.
   */
  __host__ __device__ int get_curr_size() {
    return curr_size_;
  }  

  /**
   * @brief Retrieves the tail index of the AtomicWFQueue.
   *
   * This function returns the current index of the tail in the
   * AtomicWFQueue, which represents the position where the next
   * element will be enqueued.
   *
   * @return The index of the tail in the AtomicWFQueue.
   */
  __host__ __device__ int get_tail() {
    return tail_;
  }

  /**
   * @brief Retrieves the head index of the AtomicWFQueue.
   *
   * This function returns the current index of the head in the
   * AtomicWFQueue, which represents the position of the next element
   * to be dequeued.
   *
   * @return The index of the head in the AtomicWFQueue.
   */
  __host__ __device__ int get_head() {
    return head_;
  }

 private:

  __device__ int atomic_load(const int* address) {
    return __hip_atomic_load(address, __ATOMIC_SEQ_CST,
                             __HIP_MEMORY_SCOPE_AGENT);
  }

  __device__ void atomic_store(int* address, const int val) {
    __hip_atomic_store(address, val, __ATOMIC_SEQ_CST,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  __device__ void atomic_add(int* address, const int val) {
    __hip_atomic_fetch_add(address, val, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT);
  }

  __device__ void atomic_sub(int* address, const int val) {
    __hip_atomic_fetch_sub(address, val, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT);
  }

  /**
   * @brief Checks if the AtomicWFQueue is full.
   *
   * This function determines whether the AtomicWFQueue has reached its
   * maximum capacity. It is used to prevent overflow conditions during
   * enqueue operations.
   *
   * @return true if the AtomicWFQueue is full, false otherwise.
   */
  __device__ bool is_full() {
    return atomic_load(&curr_size_) == size_;
  }

  /**
   * @brief Checks if the AtomicWFQueue is empty.
   *
   * This function determines whether the AtomicWFQueue has no elements
   * available for dequeue operations. It is used to prevent underflow
   * conditions.
   *
   * @return true if the AtomicWFQueue is empty, false otherwise.
   */
  __device__ bool is_empty() {
    return atomic_load(&curr_size_) == 0;
  }


  /**
   * @brief Internal memory allocator used to create internal structures of
   * the AtomicWFQueue.
   */
  ALLOCATOR allocator_{};

  /**
   * @brief Points to the index of first element in the AtomicWFQueue.
   */
  int head_{};

  /**
   * @brief Points to the next empty slot in the AtomicWFQueue.
   */
  int tail_{};

  /**
   * @brief Size of the AtomicWFQueue.
   */
  int size_{};

  /**
   * @brief Current size of the AtomicWFQueue.
   */
  int curr_size_{};

  /**
   * @brief Pointer to AtomicWFQueue memory
  */
  TYPE *queue_{nullptr};

  /**
   * @brief Mutex protecting the AtomicWFQueue mutations during dequeue.
   */
  MutexProxyType dequeue_mutex_;

  /**
   * @brief Mutex protecting the AtomicWFQueue mutations during enqueue_mutex.
   */
  MutexProxyType enqueue_mutex_;
};

template <typename ALLOCATOR, typename TYPE>
class AtomicWFQueueProxy {
  using AtomicWFQueueT = AtomicWFQueue<TYPE, ALLOCATOR>;
  using ProxyT = DeviceProxy<ALLOCATOR, AtomicWFQueueT>;

 public:
  __host__ __device__ AtomicWFQueueT* get() { return proxy_.get(); }

  AtomicWFQueueProxy(size_t num_elems = 1) : proxy_{num_elems} {
    new (proxy_.get()) AtomicWFQueueT();
  }

  AtomicWFQueueProxy(const AtomicWFQueueProxy& other) = delete;

  AtomicWFQueueProxy& operator=(const AtomicWFQueueProxy& other) = delete;

  AtomicWFQueueProxy(AtomicWFQueueProxy&& other) = default;

  AtomicWFQueueProxy& operator=(AtomicWFQueueProxy&& other) = default;

  ~AtomicWFQueueProxy() {
    auto atomic_wf_queue = proxy_.get();
    atomic_wf_queue->deallocate_queue();
    atomic_wf_queue->~AtomicWFQueue();
  }

 private:
  ProxyT proxy_{};
};
}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_ATOMIC_WF_QUEUE_HPP_
