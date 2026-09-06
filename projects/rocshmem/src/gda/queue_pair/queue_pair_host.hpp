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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_HOST_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_HOST_HPP_

/**
 * @file queue_pair_host.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <functional>
#include <iterator>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#include "constants.hpp"
#include "containers/free_list.hpp"
#include "memory/hip_allocator.hpp"

#include "gda/ibv_wrapper.hpp"
#include "queue_pair_common.hpp"

namespace rocshmem {

struct BufferHostInfo {
  struct ibv_mr* mr;
  BufferInfo*    info_ptr;

  BufferHostInfo(struct ibv_mr* mr, BufferInfo* info_ptr) : mr{mr}, info_ptr{info_ptr} { }
};

template <typename Iter>
class iterator_freelist {
private:
  std::vector<Iter> container{};

  /* Reserve space in container for std::distance(first, last) elements */
  void reserve(Iter first, Iter last) {
    container.reserve(std::distance(first, last));
  }

  void init(Iter first, Iter last) {
    /* Push back iterators in range [first, last) to container in reverse order
     * Sorted so that container.back() always has the smallest iterator */
    while (first != last--) {
      container.push_back(last);
    }
  }

public:
  using value_type      = Iter;
  using difference_type = typename std::iterator_traits<Iter>::difference_type;
  using size_type       = typename decltype(container)::size_type;

  iterator_freelist() = default;

  /* Initialize freelist with iterators in range [first, last) */
  iterator_freelist(Iter first, Iter last) {
    reserve(first, last);
    init(first, last);
  }

  /* Initialize freelist with iterators in range [first, std::next(first, count)) */
  iterator_freelist(Iter first, size_type count)
    : iterator_freelist{first, std::next(first, count)} { }

  bool empty() const {
    return container.empty();
  }

  size_type size() const {
    return container.size();
  }

  Iter top() {
    return container.back();
  }

  void push(Iter value) {
    std::greater<Iter> compare{};
    if (container.empty() || compare(top(), value)) {
      /* value < top(), so push_back maintains sorted property */
      container.push_back(value);
    } else {
      /* Use binary search to find insert position for value */
      auto pos = std::upper_bound(container.cbegin(), container.cend(), value, compare);
      container.insert(pos, value);
    }
  }

  void pop() {
    container.pop_back();
  }

  void reset(Iter first, Iter last) {
    container.clear();
    reserve(first, last);
    init(first, last);
  }

  void reset(Iter first, size_type count) {
    reset(first, std::next(first, count));
  }
};



/*
 * @brief QueuePairHost manages local buffer registration and
 * the atomic fetch buffers for a device QueuePair.
 */
class QueuePairHost {
/**
 * @name Constructors and Destructors
 *
 * @{
 */
public:
  /**
   * @brief Constructor.
   *
   * @param[in] pd IBVerbs Protection Domain for registering additional buffers and heaps.
   */
  __host__ explicit QueuePairHost(struct ibv_pd* pd);

  /*
   * @brief Copy constructor is deleted.
   */
  __host__ QueuePairHost(const QueuePairHost& other) = delete;

  /*
   * @brief Copy assignment operator is deleted.
   */
  __host__ QueuePairHost& operator=(const QueuePairHost& other) = delete;

  /*
   * @brief Move constructor.
   *
   * Performs a member-wise move of all data members from other to *this,
   * then resets allocated data members of other so that it can be safely reused or destroyed.
   *
   * @param[in,out] other QueuePairHost object to move from.
   */
  __host__ QueuePairHost(QueuePairHost&& other) noexcept;

  /*
   * @brief Move assignment operator.
   *
   * Cleans up all resources allocated by *this,
   * performs a member-wise move of all data members from other to *this,
   * then resets allocated data members of other so that it can be safely reused or destroyed.
   *
   * @param[in,out] other QueuePairHost object to move from.
   * @return *this
   */
  __host__ QueuePairHost& operator=(QueuePairHost&& other);

  /**
   * @brief Destructor.
   *
   * Cleans up all resources allocated by *this.
   */
  __host__ ~QueuePairHost();
/**@}*/



/**
 * @name Buffer Registration
 *
 * @{
 */
public:
  /**
   * @brief Register buffer for use as local address in rocSHMEM routines
   *
   * @param[in] addr Base address of buffer.
   * @param[in] length Length of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer registered successfully.
   * @retval ROCSHMEM_ERROR Buffer could not be registered.
   */
  __host__ int buffer_register(void *addr, size_t length);

  /**
   * @brief Unregister buffer.
   * Buffer must have previously been registered with buffer_register(void *, size_t).
   *
   * @param[in] addr Base address of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer unregistered successfully.
   */
  __host__ int buffer_unregister(void *addr);

  /**
   * @brief Unregister all registered buffers.
   *
   * @retval ROCSHMEM_SUCCESS Buffers unregistered successfully.
   */
  __host__ int buffer_unregister_all();
/**@}*/



/**
 * @name Internal
 *
 * @{
 */
private:
  template <typename T>
  __host__ std::tuple<T*, struct ibv_mr*> allocate_and_register(size_t size, int access);
/**@}*/



/**
 * @name Non-static Data Members
 *
 * @{
 */
public:
  // Used by get_laddr
  BufferInfo* buffer_info{nullptr};
  size_t num_user_buffers{0};

  // Used in atomic WQEs
  uint64_t* fetching_atomic{nullptr};
  uint64_t* nonfetching_atomic{nullptr};

  static constexpr size_t FETCHING_ATOMIC_CNT{1024};
  static_assert(FETCHING_ATOMIC_CNT % WF_SIZE == 0);
  using FreeListT = FreeList<uint64_t*>;
  FreeListT* fetching_atomic_freelist{nullptr};

  HIPAllocator allocator{};

  // Used by host in buffer_register, buffer_unregister
  std::unordered_map<void*, BufferHostInfo> buffer_info_map{};
  iterator_freelist<BufferInfo*> buffer_info_freelist{};
  struct ibv_pd* pd;
  struct ibv_mr* fetching_atomic_mr{nullptr};
  struct ibv_mr* nonfetching_atomic_mr{nullptr};
/**@}*/
};

template <typename T>
__host__ std::tuple<T*, struct ibv_mr*>
QueuePairHost::allocate_and_register(size_t count, int access) {
  void* ptr = nullptr;
  size_t size = sizeof(T) * count;
  allocator.allocate(&ptr, size);
  CHECK_HIP(hipMemset(ptr, 0, size));
  struct ibv_mr* mr = ibv.reg_mr(pd, ptr, size, access, &allocator);
  CHECK_NNULL(mr, "ibv_reg_mr");
  return {static_cast<T*>(ptr), mr};
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_HOST_HPP_
