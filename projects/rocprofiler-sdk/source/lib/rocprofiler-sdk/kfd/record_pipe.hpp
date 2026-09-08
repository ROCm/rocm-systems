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
#include <deque>
#include <utility>
#include <vector>

// Bounded SPSC handoff of copied record batches, from the ring-copier to the
// record processor. The split is load-bearing: a single thread doing both the
// copy and the pairing fell behind and overran the firmware ring.
//
// The slots ARE the buffer pool: each holds a vector whose capacity is reused,
// so steady state allocates nothing.
//
// The producer NEVER blocks. With no free slot acquire() returns nullptr and the
// caller queues the batch in its own overflow: blocking would stall the ring
// read and cause the very overrun this split avoids.
//
// MEMORY ORDERING: the release/acquire pair on m_tail publishes the filled bytes
// to the consumer; the pair on m_head publishes the slot's availability, so the
// producer cannot overwrite a batch the consumer is still reading. Each index
// has exactly one writer, so neither is a read-modify-write.

namespace rocprofiler
{
namespace kfd
{
// One batch of records copied out of the ring in a single drain.
struct record_batch
{
    std::vector<copied_record> records = {};
    uint64_t                   now_ns  = 0;  // host clock sampled once for the batch
    // Which GPU's ring these came from. Pairing and correlation are keyed by it,
    // so records from different GPUs can never be matched to each other.
    uint32_t gpu_id = 0;

    void clear()
    {
        records.clear();  // keeps capacity: steady state does not allocate
        now_ns = 0;
        gpu_id = 0;
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

// PRODUCER-side helper: move as many FIFO batches from `overflow` into `pipe` as
// there are free slots, publishing each. Returns true once `overflow` is empty.
// This is the same handoff the copier performs when a slot frees; factoring it
// out lets the shutdown path pump the remaining overflow to the processor before
// declaring the reader finished, so a batch already taken from the ring is never
// dropped when the pipe was full at stop.
template <size_t Capacity>
bool
spill_overflow_to_pipe(record_pipe<Capacity>&    pipe,
                       std::deque<record_batch>& overflow,
                       std::atomic<uint64_t>*    inflight = nullptr)
{
    while(!overflow.empty())
    {
        auto* _slot = pipe.acquire();
        if(_slot == nullptr) return false;
        // Count captured BEFORE the move: std::move empties the source vector, so
        // reading its size after would read 0 and under-discharge the reservation.
        const auto _count = overflow.front().records.size();
        *_slot            = std::move(overflow.front());
        overflow.pop_front();
        pipe.publish();
        // Discharge after publish; release pairs with the GC gate's acquire load.
        if(inflight != nullptr) inflight->fetch_sub(_count, std::memory_order_release);
    }
    return true;
}

// D2 gate predicate: closed-window GC may run only when nothing copied is still
// invisible to the processor -- no reader-reserved-but-unpublished record AND an
// empty pipe. Pure and header-exposed so a unit test can drive all four
// combinations directly; the acquire load feeding `inflight` and its ordering
// ahead of pipe.empty() live at the call site in kfd_reader.cpp.
inline bool
gc_gate_open(uint64_t inflight, bool pipe_empty)
{
    return inflight == 0 && pipe_empty;
}
}  // namespace kfd
}  // namespace rocprofiler
