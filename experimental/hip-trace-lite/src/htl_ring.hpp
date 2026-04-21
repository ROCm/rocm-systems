// htl_ring.hpp — bounded single-producer single-consumer ring.
// Capacity must be a power of two. One slot reserved for full/empty disambiguation.
#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>

namespace htl {

template <typename T, size_t Capacity>
class SpscRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two");
    static_assert(Capacity >= 2, "Capacity must be at least 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    SpscRing() : head_(0), tail_(0) {}

    bool try_push(const T& v) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & kMask;
        if (next == tail_.load(std::memory_order_acquire)) return false;  // full
        slots_[head] = v;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;  // empty
        out = slots_[tail];
        tail_.store((tail + 1) & kMask, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t kMask = Capacity - 1;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    T slots_[Capacity];
};

}  // namespace htl
