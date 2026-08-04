// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

// Bounded single-producer / single-consumer handoff of copied record batches,
// between the ring-copier (producer) and the record processor (consumer).
//
// The slots ARE the buffer pool: each holds a vector whose capacity is reused, so
// steady-state operation allocates nothing. A batch is only ever touched by one
// side at a time, and which side owns it is decided entirely by the two indices.
//
// The producer NEVER blocks. If the consumer has fallen behind and no slot is
// free, acquire() returns nullptr and the caller drops the batch and counts it --
// blocking would stall the ring read and cause exactly the overrun this split
// exists to avoid, and growing without bound would trade a bounded data loss for
// an unbounded memory one.
//
// MEMORY ORDERING (the whole correctness argument):
//   producer: fills slot[tail % N]  ->  m_tail.store(tail + 1, release)
//   consumer: m_tail.load(acquire)  ->  reads slot[head % N]
// The release/acquire pair on m_tail publishes the filled bytes: everything the
// producer wrote before the store is visible to the consumer after the load.
//   consumer: done with slot        ->  m_head.store(head + 1, release)
//   producer: m_head.load(acquire)  ->  reuses that slot
// The release/acquire pair on m_head publishes the slot's availability, so the
// producer cannot overwrite a batch the consumer is still reading.
// Each index has exactly one writer, so neither is a read-modify-write.

namespace rocprofiler
{
namespace kfd
{
// One batch of records copied out of the ring in a single drain.
struct record_batch
{
    std::vector<copied_record> records = {};
    uint64_t                   now_ns  = 0;  // host clock sampled once for the batch

    void clear()
    {
        records.clear();  // keeps capacity: steady state does not allocate
        now_ns = 0;
    }
};

template <size_t Capacity>
class record_pipe
{
    static_assert(Capacity >= 2, "a pipe needs at least one in-flight and one free slot");

public:
    record_pipe()  = default;
    ~record_pipe() = default;

    record_pipe(const record_pipe&) = delete;
    record_pipe& operator=(const record_pipe&) = delete;

    // PRODUCER. A cleared batch to fill, or nullptr when the consumer has not
    // kept up. Never blocks.
    record_batch* acquire()
    {
        const auto _tail = m_tail.load(std::memory_order_relaxed);
        if(_tail - m_head.load(std::memory_order_acquire) >= Capacity) return nullptr;
        auto& _slot = m_slots[_tail % Capacity];
        _slot.clear();
        return &_slot;
    }

    // PRODUCER. Publish the batch returned by acquire(). The release store is what
    // makes the copied bytes visible to the consumer.
    void publish()
    {
        m_tail.store(m_tail.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    // CONSUMER. The oldest unread batch, or nullptr when empty.
    record_batch* peek()
    {
        const auto _head = m_head.load(std::memory_order_relaxed);
        if(m_tail.load(std::memory_order_acquire) == _head) return nullptr;
        return &m_slots[_head % Capacity];
    }

    // CONSUMER. Return the batch from peek() to the pool.
    void pop()
    {
        m_head.store(m_head.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    bool empty() const
    {
        return m_tail.load(std::memory_order_acquire) == m_head.load(std::memory_order_acquire);
    }

    size_t size() const
    {
        return m_tail.load(std::memory_order_acquire) - m_head.load(std::memory_order_acquire);
    }

    static constexpr size_t capacity() { return Capacity; }

private:
    std::array<record_batch, Capacity> m_slots = {};
    // Written only by the consumer / only by the producer respectively.
    std::atomic<size_t> m_head = {0};
    std::atomic<size_t> m_tail = {0};
};
}  // namespace kfd
}  // namespace rocprofiler
