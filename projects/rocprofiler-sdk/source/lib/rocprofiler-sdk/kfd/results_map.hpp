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

#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

// ResultsMap: the handoff from the KFD reader thread to the completion path.
// The reader pairs a firmware start+eop record and deposits raw GPU ticks keyed
// by correlation_key; get_dispatch_time() takes them (converting ticks->ns
// there, so validation can compare against the HSA result on the same call).
// evict_stale() reclaims entries the completion path never took (dispatch
// completed via HSA fallback before its record arrived) to prevent leaks.
//
// deposit() is an event: it notifies wait_take(), so the completion path can wait
// for a record that is merely late instead of one-shot take()ing and silently
// substituting HSA timestamps. That rendezvous is the Phase 1 race-seal; it is
// only entered when kfd_selection_enabled() is true.

namespace rocprofiler
{
namespace kfd
{
// Monotonic clock backing the rendezvous deadline. Distinct from the
// CLOCK_BOOTTIME stamps stored in records: only used to bound the wait.
uint64_t
steady_now_ns();

// Raw firmware timing for one dispatch. Ticks are converted to CLOCK_BOOTTIME
// ns later (in get_dispatch_time); deposited_at_ns is a host CLOCK_BOOTTIME
// stamp used only for stale eviction.
struct kfd_timing_result
{
    uint64_t start_gpu_ticks = 0;
    uint64_t end_gpu_ticks   = 0;
    uint64_t deposited_at_ns = 0;
};

class ResultsMap
{
public:
    ResultsMap()  = default;
    ~ResultsMap() = default;

    ResultsMap(const ResultsMap&)     = delete;
    ResultsMap(ResultsMap&&) noexcept = delete;
    ResultsMap& operator=(const ResultsMap&) = delete;
    ResultsMap& operator=(ResultsMap&&) noexcept = delete;

    // Deposit a paired result (KFD reader thread). Insert-if-absent
    // (first-writer-wins): a duplicate key keeps the existing entry rather than
    // overwriting it. See the emplace note in deposit().
    void deposit(const correlation_key& key, const kfd_timing_result& result);

    // Atomically find + erase. nullopt if not present.
    std::optional<kfd_timing_result> take(const correlation_key& key);

    // Rendezvous form of take(): block until the reader deposits this key, the
    // waiters are abandoned, or the ABSOLUTE steady_now_ns() deadline passes, then
    // take. deadline_ns == 0 means do not block (plain take). The predicate is
    // re-evaluated under the lock on every wakeup, so a deposit racing the start of
    // the wait cannot be lost and a spurious wakeup cannot end the wait early.
    //
    // Blocks the calling completion thread: the caller must hold no other KFD lock.
    std::optional<kfd_timing_result> wait_take(const correlation_key& key, uint64_t deadline_ns);

    // Terminal: no further results can be deposited (reader dead or stopped). Wakes
    // every waiter and makes later waits return immediately instead of burning
    // their deadline on a reader that will never answer.
    void abandon_waiters();

    // Count an eligible dispatch that reported HSA timestamps anyway, so a source
    // substitution is never silent.
    void note_hsa_fallback();

    // Drop every result for a doorbell slot whose queue was destroyed, so a stale
    // result cannot be taken by whatever reuses the slot. Returns the count.
    size_t erase_slot(uint32_t gpu_id, uint32_t doorbell_off);

    // Remove entries whose deposited_at_ns is older than max_age_ns relative to
    // now_ns. now_ns is passed in (not sampled) so the function is deterministic
    // and unit-testable. Returns the number of entries evicted.
    size_t evict_stale(uint64_t now_ns, uint64_t max_age_ns);

    // Cumulative take() outcomes. A miss means a completed dispatch looked up a
    // valid correlation key before the reader had deposited its firmware record
    // (or the record never arrived) and silently fell back to HSA timestamps --
    // the only externally visible symptom of that race.
    struct take_stats
    {
        uint64_t hits      = 0;
        uint64_t misses    = 0;
        uint64_t fallbacks = 0;
    };
    take_stats stats() const;

private:
    using map_t = std::unordered_map<correlation_key, kfd_timing_result, correlation_key_hash>;

    // Caller must hold m_mutex.
    std::optional<kfd_timing_result> take_locked(const correlation_key& key);

    mutable std::mutex      m_mutex     = {};
    std::condition_variable m_cv        = {};
    map_t                   m_data      = {};
    bool                    m_abandoned = false;
    std::atomic<uint64_t>   m_hits      = {0};
    std::atomic<uint64_t>   m_misses    = {0};
    std::atomic<uint64_t>   m_fallbacks = {0};
};
}  // namespace kfd
}  // namespace rocprofiler
