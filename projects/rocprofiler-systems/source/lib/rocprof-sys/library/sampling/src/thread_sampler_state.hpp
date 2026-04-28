// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Per-thread sampling state — DEC-3, DEC-4, DEC-15.
//
// thread_sampler_state<Policies>: holds the two optional triggers, the ring
// buffer, the re-entry guard flag, in-flight count, and configured signal set.
// One instance per active thread.
//
// thread_sampler_state_registry<Policies, MaxThreads>: owns an array of
// unique_ptr<thread_sampler_state<Policies>> indexed by logical tid.
// The MaxThreads template parameter defaults to 512 (ROCPROFSYS_MAX_THREADS)
// in production; unit tests use a small value (e.g. 8).

#include "sampling/src/sample_ring_buffer.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <thread>

namespace rocprofsys::sampling
{

// ─────────────────────────────────────────────────────────────────────────────
// thread_sampler_state<Policies>
// ─────────────────────────────────────────────────────────────────────────────

template <class Policies>
class thread_sampler_state
{
public:
    using timer_trigger_t    = typename Policies::timer_trigger;
    using overflow_trigger_t = typename Policies::overflow_trigger;

    static constexpr size_t RING_CAPACITY = 2048;

    thread_sampler_state() = default;

    thread_sampler_state(thread_sampler_state const&)            = delete;
    thread_sampler_state& operator=(thread_sampler_state const&) = delete;

    // ----- configured signal set -----
    void set_signal_types(std::set<int> sigs) { signal_types_ = std::move(sigs); }
    [[nodiscard]] std::set<int> const& signal_types() const noexcept
    {
        return signal_types_;
    }

    // ----- trigger accessors (DEC-3) -----
    [[nodiscard]] std::optional<timer_trigger_t>& timer_trigger() noexcept
    {
        return timer_trigger_;
    }
    [[nodiscard]] std::optional<timer_trigger_t> const& timer_trigger() const noexcept
    {
        return timer_trigger_;
    }
    [[nodiscard]] std::optional<overflow_trigger_t>& overflow_trigger() noexcept
    {
        return overflow_trigger_;
    }
    [[nodiscard]] std::optional<overflow_trigger_t> const& overflow_trigger()
        const noexcept
    {
        return overflow_trigger_;
    }

    // ----- ring buffer -----
    [[nodiscard]] sample_ring_buffer<RING_CAPACITY>& ring_buffer() noexcept
    {
        return ring_buffer_;
    }
    [[nodiscard]] sample_ring_buffer<RING_CAPACITY> const& ring_buffer() const noexcept
    {
        return ring_buffer_;
    }

    // ----- running flag -----
    [[nodiscard]] bool is_running() const noexcept
    {
        return running_.load(std::memory_order_acquire);
    }

    void start() noexcept { running_.store(true, std::memory_order_release); }
    void stop() noexcept { running_.store(false, std::memory_order_release); }

    // ----- in-flight count (DEC-4 § 7 shutdown busy-wait) -----
    // Incremented when signal handler enters, decremented on exit.
    void enter_in_flight() noexcept
    {
        in_flight_count_.fetch_add(1, std::memory_order_relaxed);
    }
    void exit_in_flight() noexcept
    {
        in_flight_count_.fetch_sub(1, std::memory_order_release);
    }

    [[nodiscard]] int in_flight_count() const noexcept
    {
        return in_flight_count_.load(std::memory_order_acquire);
    }

    // Busy-wait until in_flight_count drops to 0 or timeout_ms elapses.
    // Returns true if count reached 0, false if timed out.
    bool wait_for_in_flight_zero(int timeout_ms = 5000) noexcept
    {
        auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while(in_flight_count_.load(std::memory_order_acquire) != 0)
        {
            if(std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    }

    // ----- re-entry guard (DEC-15) -----
    // Returns true if the handler was already entered (must drop sample).
    // Returns false if not yet entered (safe to proceed; flag is now set).
    [[nodiscard]] bool try_enter_handler() noexcept
    {
        return in_handler_.test_and_set(std::memory_order_acquire);
    }

    void exit_handler() noexcept { in_handler_.clear(std::memory_order_release); }

    // ----- dropped sample counter -----
    void increment_dropped() noexcept
    {
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] size_t dropped_count() const noexcept
    {
        return dropped_count_.load(std::memory_order_relaxed);
    }

private:
    std::set<int>                     signal_types_;
    std::optional<timer_trigger_t>    timer_trigger_;
    std::optional<overflow_trigger_t> overflow_trigger_;
    sample_ring_buffer<RING_CAPACITY> ring_buffer_;
    std::atomic<bool>                 running_{ false };
    std::atomic<int>                  in_flight_count_{ 0 };
    std::atomic_flag                  in_handler_ = ATOMIC_FLAG_INIT;
    std::atomic<size_t>               dropped_count_{ 0 };
};

// ─────────────────────────────────────────────────────────────────────────────
// thread_sampler_state_registry<Policies, MaxThreads>
// ─────────────────────────────────────────────────────────────────────────────
//
// Owns std::array<std::unique_ptr<thread_sampler_state<Policies>>, MaxThreads>.
// Provides emplace(tid) / at(tid) / reset() / each().
// MaxThreads defaults to 512 (ROCPROFSYS_MAX_THREADS) in production.
// Unit tests parameterise to a smaller value via the template argument.

template <class Policies, size_t MaxThreads = 512>
class thread_sampler_state_registry
{
public:
    using state_t   = thread_sampler_state<Policies>;
    using state_ptr = std::unique_ptr<state_t>;
    using array_t   = std::array<state_ptr, MaxThreads>;

    thread_sampler_state_registry() noexcept = default;

    thread_sampler_state_registry(thread_sampler_state_registry const&) = delete;
    thread_sampler_state_registry& operator=(thread_sampler_state_registry const&) =
        delete;

    // Create a new state for tid if none exists.
    // OOB tids (tid >= MaxThreads or tid < 0) are silently ignored.
    // May throw std::bad_alloc on first emplace for a tid.
    void emplace(int64_t tid)
    {
        if(tid < 0 || static_cast<size_t>(tid) >= MaxThreads) return;
        auto& slot = slots_[static_cast<size_t>(tid)];
        if(!slot) slot = std::make_unique<state_t>();
    }

    // Return pointer to state for tid, or nullptr if absent / OOB.
    [[nodiscard]] state_t* at(int64_t tid) noexcept
    {
        if(tid < 0 || static_cast<size_t>(tid) >= MaxThreads) return nullptr;
        return slots_[static_cast<size_t>(tid)].get();
    }

    [[nodiscard]] state_t const* at(int64_t tid) const noexcept
    {
        if(tid < 0 || static_cast<size_t>(tid) >= MaxThreads) return nullptr;
        return slots_[static_cast<size_t>(tid)].get();
    }

    // Destroy the state for a single tid (called from shutdown(tid)).
    void erase(int64_t tid) noexcept
    {
        if(tid < 0 || static_cast<size_t>(tid) >= MaxThreads) return;
        slots_[static_cast<size_t>(tid)].reset();
    }

    // Destroy all states (called from global shutdown).
    void reset() noexcept
    {
        for(auto& slot : slots_)
            slot.reset();
    }

    // Apply a callable to every non-null state — mutable and const overloads.
    template <class Fn>
    void each(Fn&& fn) noexcept(noexcept(fn(int64_t{}, std::declval<state_t&>())))
    {
        for(size_t i = 0; i < MaxThreads; ++i)
        {
            if(slots_[i]) fn(static_cast<int64_t>(i), *slots_[i]);
        }
    }

    template <class Fn>
    void each(Fn&& fn) const
        noexcept(noexcept(fn(int64_t{}, std::declval<state_t const&>())))
    {
        for(size_t i = 0; i < MaxThreads; ++i)
        {
            if(slots_[i]) fn(static_cast<int64_t>(i), *slots_[i]);
        }
    }

    [[nodiscard]] static constexpr size_t capacity() noexcept { return MaxThreads; }

private:
    array_t slots_;
};

}  // namespace rocprofsys::sampling
