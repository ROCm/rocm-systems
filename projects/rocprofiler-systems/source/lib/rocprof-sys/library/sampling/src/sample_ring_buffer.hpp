// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_record.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <optional>

namespace rocprofsys::sampling
{

// Lock-free, fixed-capacity ring buffer for backtrace_record.
// Concurrency contract: SINGLE-PRODUCER (one signal handler per thread writes),
// SINGLE-CONSUMER (post_process reads). Multi-producer is NOT supported.
// Signal-handler safe: no allocation, no mutex, no format/log calls.
//
// Write protocol (C-4 fix): write payload to slot FIRST, then publish by
// incrementing write_count_ with release semantics. Reader acquires write_count_
// before reading the slot — no race on slots_.
//
// Capacity N = 2048 matches the legacy buffer_size trait (architecture.md § 2).
template <size_t N>
class sample_ring_buffer
{
    static_assert(N > 0, "sample_ring_buffer capacity must be > 0");

public:
    static constexpr size_t CAPACITY = N;

    sample_ring_buffer() noexcept
    {
        write_count_.store(0, std::memory_order_relaxed);
        read_count_.store(0, std::memory_order_relaxed);
    }

    sample_ring_buffer(sample_ring_buffer const&)            = delete;
    sample_ring_buffer& operator=(sample_ring_buffer const&) = delete;

    // Push a record. Returns false if the buffer was full (new record dropped,
    // existing records preserved — matches legacy spsc_queue::push semantics, C-8).
    // Async-signal-safe (SPSC): no allocation, no mutex, no logging.
    bool try_push(backtrace_record const& rec) noexcept
    {
        auto wc = write_count_.load(std::memory_order_relaxed);
        auto rc = read_count_.load(std::memory_order_acquire);
        if((wc - rc) >= N)
        {
            // Full — drop new sample, preserve oldest.
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        // Write payload to slot BEFORE publishing the incremented write_count_.
        std::memcpy(&slots_[wc % N], &rec, sizeof(backtrace_record));
        // Release: makes the slot write visible to the consumer before the count.
        write_count_.store(wc + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] size_t dropped_count() const noexcept
    {
        return dropped_count_.load(std::memory_order_relaxed);
    }

    // Non-destructively count available records; not signal-safe.
    [[nodiscard]] size_t count() const noexcept
    {
        auto wc = write_count_.load(std::memory_order_acquire);
        auto rc = read_count_.load(std::memory_order_acquire);
        return (wc >= rc) ? (wc - rc) : 0;
    }

    // Pop one record. Returns nullopt if empty.
    [[nodiscard]] std::optional<backtrace_record> pop() noexcept
    {
        auto rc = read_count_.load(std::memory_order_relaxed);
        // Acquire: pairs with the release store in try_push.
        auto wc = write_count_.load(std::memory_order_acquire);
        if(rc >= wc) return std::nullopt;
        backtrace_record out;
        std::memcpy(&out, &slots_[rc % N], sizeof(backtrace_record));
        read_count_.store(rc + 1, std::memory_order_release);
        return out;
    }

    // Reset for reuse (not signal-safe; call only when no handlers are active).
    void reset() noexcept
    {
        write_count_.store(0, std::memory_order_relaxed);
        read_count_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] size_t write_count() const noexcept
    {
        return write_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t read_count() const noexcept
    {
        return read_count_.load(std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<size_t> write_count_{ 0 };
    alignas(64) std::atomic<size_t> read_count_{ 0 };
    alignas(64) std::atomic<size_t> dropped_count_{ 0 };
    backtrace_record slots_[N] = {};
};

}  // namespace rocprofsys::sampling
