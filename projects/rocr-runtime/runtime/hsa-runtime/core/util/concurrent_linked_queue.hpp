////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_UTIL_CONCURRENT_LINKED_QUEUE_HPP_
#define HSA_RUNTIME_CORE_UTIL_CONCURRENT_LINKED_QUEUE_HPP_

#include <atomic>
#include <new>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace rocr { /*@{*/

namespace details {

template <typename T, int N> struct TaggedPointerHelper {
  // Use upper 16 bits and lower 6 bits for tagging (22 bits total)
  // Upper 16 bits: non-canonical space (bits 48-63)
  // Lower 6 bits: alignment bits (bits 0-5) - always 0 for 64-byte aligned addresses
  static constexpr uintptr_t UpperTagShift = 48;  // Upper 16 bits
  static constexpr uintptr_t LowerTagShift = 0;   // Lower 6 bits
  static constexpr uintptr_t UpperTagMask = ((1ULL << 16) - 1) << UpperTagShift;
  static constexpr uintptr_t LowerTagMask = (1ULL << N) - 1;
  static constexpr uintptr_t TagMask = UpperTagMask | LowerTagMask;

 private:
  TaggedPointerHelper();        // Cannot instantiate
  void* operator new(size_t);   // allocate or
  void operator delete(void*);  // delete a TaggedPointerHelper.

 public:
  //! Create a tagged pointer.
  static TaggedPointerHelper* make(T* ptr, size_t tag) {
    // Split 22-bit tag: upper 16 bits go to upper bits, lower 6 bits go to lower bits
    uintptr_t upper_tag = (tag >> N) & 0xFFFF;            // Upper 16 bits of tag
    uintptr_t lower_tag = tag & ((1ULL << N) - 1);        // Lower 6 bits of tag

    return reinterpret_cast<TaggedPointerHelper*>((reinterpret_cast<uintptr_t>(ptr) & ~TagMask) |
                                                  ((upper_tag << UpperTagShift) & UpperTagMask) |
                                                  (lower_tag & LowerTagMask));
  }

  //! Return the pointer value.
  T* ptr() { return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this) & ~TagMask); }

  //! Return the tag value.
  size_t tag() const {
    uintptr_t upper_tag = (reinterpret_cast<uintptr_t>(this) & UpperTagMask) >> UpperTagShift;
    uintptr_t lower_tag = reinterpret_cast<uintptr_t>(this) & LowerTagMask;
    return (upper_tag << 6) | lower_tag;  // Combine upper 16 bits and lower 6 bits
  }
};

}  // namespace details

// Simple memory allocation functions
inline void* aligned_allocate(size_t size, size_t alignment) {
  return std::aligned_alloc(alignment, size);
}

inline void aligned_deallocate(void* ptr) {
  std::free(ptr);
}

// Thread-local node cache for efficient allocation/deallocation
template <typename T, int N> class ThreadLocalNodeCache {
private:
  static constexpr size_t kCacheSize = N;  // Number of nodes to cache per thread
  static constexpr size_t kAlignment = 1 << N;

private:
  std::vector<void*> free_list_;
  size_t free_count_;

public:
  ThreadLocalNodeCache() : free_count_(0) {
    free_list_.reserve(kCacheSize);
  }

  ~ThreadLocalNodeCache() {
    // Return all cached nodes to global allocator
    for (auto* node : free_list_) {
      aligned_deallocate(node);
    }
    free_list_.clear();
    free_count_ = 0;
  }

  // Allocate a node, prefer from cache
  void* allocate(size_t node_size) {
    if (!free_list_.empty()) {
      void* node = free_list_.back();
      free_list_.pop_back();
      --free_count_;
      return node;
    }
    return aligned_allocate(node_size, kAlignment);
  }

  // Deallocate a node, cache if possible
  void deallocate(void* node) {
    if (free_count_ < kCacheSize) {
      free_list_.push_back(node);
      ++free_count_;
    } else {
      aligned_deallocate(node);
    }
  }

  // Thread-local storage - each thread gets its own cache
  static thread_local ThreadLocalNodeCache<T, N> cache;
};

// Static member definition
template <typename T, int N>
thread_local ThreadLocalNodeCache<T, N> ThreadLocalNodeCache<T, N>::cache;

/*! \brief An unbounded thread-safe queue.
 *
 * This queue orders elements first-in-first-out. It is based on the algorithm
 * "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue
 * Algorithms by Maged M. Michael and Michael L. Scott.".
 */
template <typename T, int N = 6> class ConcurrentLinkedQueue {
  //! A simply-linked node
  struct Node {
    typedef details::TaggedPointerHelper<Node, N> TaggedPointerHelper;
    typedef TaggedPointerHelper* Ptr;

    T value_;                //!< The value stored in that node.
    std::atomic<Ptr> next_;  //!< Pointer to the next node

    //! Create a Node::Ptr
    static inline Ptr ptr(Node* ptr, size_t counter = 0) {
      return TaggedPointerHelper::make(ptr, counter);
    }
  };

 private:
  std::atomic<typename Node::Ptr> head_;  //! Pointer to the oldest element.
  std::atomic<typename Node::Ptr> tail_;  //! Pointer to the most recent element.

 private:
  //! \brief Allocate a free node using thread-local cache.
  static inline Node* allocNode() {
    return reinterpret_cast<Node*>(ThreadLocalNodeCache<T, N>::cache.allocate(sizeof(Node)));
  }

  //! \brief Return a node to the thread-local cache.
  static inline void reclaimNode(Node* node) {
    ThreadLocalNodeCache<T, N>::cache.deallocate(node);
  }

 public:
  //! \brief Initialize a new concurrent linked queue.
  ConcurrentLinkedQueue();

  //! \brief Destroy this concurrent linked queue.
  ~ConcurrentLinkedQueue();

  //! \brief Enqueue an element using perfect forwarding (handles both lvalues and rvalues).
  template<typename U>
  inline void enqueue(U&& elem);

  //! \brief Dequeue an element from this queue.
  inline T dequeue();

  //! \brief Check if queue is empty
  inline bool empty();
};

/*@}*/

template <typename T, int N> inline ConcurrentLinkedQueue<T, N>::ConcurrentLinkedQueue() {
  // Create the first "dummy" node.
  Node* dummy = allocNode();
  dummy->next_ = nullptr;

  // Head and tail should now point to it (empty list).
  head_ = tail_ = Node::ptr(dummy);

  // Make sure the instance is fully initialized before it becomes
  // globally visible.
  std::atomic_thread_fence(std::memory_order_release);
}

template <typename T, int N> inline ConcurrentLinkedQueue<T, N>::~ConcurrentLinkedQueue() {
  typename Node::Ptr head = head_;
  typename Node::Ptr tail = tail_;
  while (head->ptr() != tail->ptr()) {
    Node* node = head->ptr();
    head = head->ptr()->next_;
    reclaimNode(node);
  }
  reclaimNode(head->ptr());
}

// Implementation of perfect-forwarding enqueue
// Only this version should exist!
template <typename T, int N>
template <typename U>
inline void ConcurrentLinkedQueue<T, N>::enqueue(U&& elem) {
  Node* node = allocNode();
  node->value_ = std::forward<U>(elem); // Move or copy as appropriate
  node->next_ = nullptr;

  for (;;) {
    typename Node::Ptr tail = tail_.load(std::memory_order_acquire);
    typename Node::Ptr next = tail->ptr()->next_.load(std::memory_order_acquire);
    if (tail == tail_.load(std::memory_order_acquire)) {
      if (next->ptr() == nullptr) {
        if (tail->ptr()->next_.compare_exchange_weak(next, Node::ptr(node, next->tag() + 1),
                                                     std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
          tail_.compare_exchange_strong(tail, Node::ptr(node, tail->tag() + 1),
                                        std::memory_order_acq_rel, std::memory_order_acquire);
          return;
        }
      } else {
        tail_.compare_exchange_strong(tail, Node::ptr(next->ptr(), tail->tag() + 1),
                                      std::memory_order_acq_rel, std::memory_order_acquire);
      }
    }
  }
}

template <typename T, int N> inline T ConcurrentLinkedQueue<T, N>::dequeue() {
  for (;;) {
    typename Node::Ptr head = head_.load(std::memory_order_acquire);
    typename Node::Ptr tail = tail_.load(std::memory_order_acquire);
    typename Node::Ptr next = head->ptr()->next_.load(std::memory_order_acquire);
    if (head == head_.load(std::memory_order_acquire)) {
      if (head->ptr() == tail->ptr()) {
        if (next->ptr() == nullptr) {
          return T();
        }
        tail_.compare_exchange_strong(tail, Node::ptr(next->ptr(), tail->tag() + 1),
                                      std::memory_order_acq_rel, std::memory_order_acquire);
      } else {
        T value = next->ptr()->value_;
        if (head_.compare_exchange_weak(head, Node::ptr(next->ptr(), head->tag() + 1),
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
          // we can reclaim head now
          reclaimNode(head->ptr());
          return value;
        }
      }
    }
  }
}

template <typename T, int N> inline bool ConcurrentLinkedQueue<T, N>::empty() {
  for (;;) {
    typename Node::Ptr head = head_.load(std::memory_order_acquire);
    typename Node::Ptr tail = tail_.load(std::memory_order_acquire);
    typename Node::Ptr next = head->ptr()->next_.load(std::memory_order_acquire);
    if (head == head_.load(std::memory_order_acquire)) {
      if (head->ptr() == tail->ptr()) {
        if (next->ptr() == nullptr) {
          return true;
        }
      }
      return false;
    }
  }
}

}  // namespace rocr

#endif /*HSA_RUNTIME_CORE_UTIL_CONCURRENT_LINKED_QUEUE_HPP_*/