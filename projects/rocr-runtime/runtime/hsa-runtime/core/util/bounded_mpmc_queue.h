/*
Copyright (c) 2020 Erik Rigtorp <erik@rigtorp.se>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from https://github.com/rigtorp/MPMCQueue
- Removed noexcept static_asserts to allow types without noexcept constructors
- Kept all correctness-critical memory ordering and alignment
*/

#ifndef HSA_RUNTIME_CORE_UTIL_BOUNDED_MPMC_QUEUE_H_
#define HSA_RUNTIME_CORE_UTIL_BOUNDED_MPMC_QUEUE_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

#ifndef __cpp_aligned_new
#include <cstdlib> // posix_memalign
#endif

namespace rocr {
namespace mpmc {

static constexpr size_t kCacheLineSize = 64;

#if defined(__cpp_aligned_new)
template <typename T> using AlignedAllocator = std::allocator<T>;
#else
template <typename T> struct AlignedAllocator {
  using value_type = T;

  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    T *p;
    if (posix_memalign(reinterpret_cast<void **>(&p), alignof(T),
                       sizeof(T) * n) != 0) {
      throw std::bad_alloc();
    }
    return p;
  }

  void deallocate(T *p, std::size_t) { free(p); }
};
#endif

template <typename T> struct Slot {
  ~Slot() noexcept {
    if (turn & 1) {
      destroy();
    }
  }

  template <typename... Args> void construct(Args &&...args) {
    new (&storage) T(std::forward<Args>(args)...);
  }

  void destroy() noexcept {
    reinterpret_cast<T *>(&storage)->~T();
  }

  T &&move() noexcept { return reinterpret_cast<T &&>(storage); }

  alignas(kCacheLineSize) std::atomic<size_t> turn = {0};
  typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
};

template <typename T,
          typename Allocator = AlignedAllocator<Slot<T>>>
class BoundedQueue {
public:
  explicit BoundedQueue(const size_t capacity,
                        const Allocator &allocator = Allocator())
      : capacity_(capacity), allocator_(allocator), head_(0), tail_(0) {
    if (capacity_ < 1) {
      throw std::invalid_argument("capacity < 1");
    }
    slots_ = allocator_.allocate(capacity_ + 1);
    if (reinterpret_cast<size_t>(slots_) % alignof(Slot<T>) != 0) {
      allocator_.deallocate(slots_, capacity_ + 1);
      throw std::bad_alloc();
    }
    for (size_t i = 0; i < capacity_; ++i) {
      new (&slots_[i]) Slot<T>();
    }
  }

  ~BoundedQueue() noexcept {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].~Slot();
    }
    allocator_.deallocate(slots_, capacity_ + 1);
  }

  BoundedQueue(const BoundedQueue &) = delete;
  BoundedQueue &operator=(const BoundedQueue &) = delete;

  template <typename... Args> void emplace(Args &&...args) {
    auto const head = head_.fetch_add(1);
    auto &slot = slots_[idx(head)];
    while (turn(head) * 2 != slot.turn.load(std::memory_order_acquire))
      ;
    slot.construct(std::forward<Args>(args)...);
    slot.turn.store(turn(head) * 2 + 1, std::memory_order_release);
  }

  template <typename... Args> bool try_emplace(Args &&...args) {
    auto head = head_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx(head)];
      if (turn(head) * 2 == slot.turn.load(std::memory_order_acquire)) {
        if (head_.compare_exchange_strong(head, head + 1)) {
          slot.construct(std::forward<Args>(args)...);
          slot.turn.store(turn(head) * 2 + 1, std::memory_order_release);
          return true;
        }
      } else {
        auto const prevHead = head;
        head = head_.load(std::memory_order_acquire);
        if (head == prevHead) {
          return false;
        }
      }
    }
  }

  void push(const T &v) { emplace(v); }

  bool try_push(const T &v) { return try_emplace(v); }

  void pop(T &v) noexcept {
    auto const tail = tail_.fetch_add(1);
    auto &slot = slots_[idx(tail)];
    while (turn(tail) * 2 + 1 != slot.turn.load(std::memory_order_acquire))
      ;
    v = slot.move();
    slot.destroy();
    slot.turn.store(turn(tail) * 2 + 2, std::memory_order_release);
  }

  bool try_pop(T &v) noexcept {
    auto tail = tail_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx(tail)];
      if (turn(tail) * 2 + 1 == slot.turn.load(std::memory_order_acquire)) {
        if (tail_.compare_exchange_strong(tail, tail + 1)) {
          v = slot.move();
          slot.destroy();
          slot.turn.store(turn(tail) * 2 + 2, std::memory_order_release);
          return true;
        }
      } else {
        auto const prevTail = tail;
        tail = tail_.load(std::memory_order_acquire);
        if (tail == prevTail) {
          return false;
        }
      }
    }
  }

  ptrdiff_t size() const noexcept {
    return static_cast<ptrdiff_t>(head_.load(std::memory_order_relaxed) -
                                  tail_.load(std::memory_order_relaxed));
  }

  bool empty() const noexcept { return size() <= 0; }

private:
  constexpr size_t idx(size_t i) const noexcept { return i % capacity_; }
  constexpr size_t turn(size_t i) const noexcept { return i / capacity_; }

  const size_t capacity_;
  Slot<T> *slots_;
#if defined(__has_cpp_attribute) && __has_cpp_attribute(no_unique_address)
  Allocator allocator_ [[no_unique_address]];
#else
  Allocator allocator_;
#endif

  alignas(kCacheLineSize) std::atomic<size_t> head_;
  alignas(kCacheLineSize) std::atomic<size_t> tail_;
};

} // namespace mpmc
} // namespace rocr

#endif // HSA_RUNTIME_CORE_UTIL_BOUNDED_MPMC_QUEUE_H_
