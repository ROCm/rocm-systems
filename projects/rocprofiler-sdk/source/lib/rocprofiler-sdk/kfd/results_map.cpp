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

#include "lib/rocprofiler-sdk/kfd/results_map.hpp"

#include <chrono>

namespace rocprofiler
{
namespace kfd
{
namespace
{
std::chrono::steady_clock::time_point
to_steady(uint64_t ns)
{
    using namespace std::chrono;
    return steady_clock::time_point{duration_cast<steady_clock::duration>(nanoseconds{ns})};
}
}  // namespace

uint64_t
steady_now_ns()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

void
ResultsMap::deposit(const correlation_key& key, const kfd_timing_result& result)
{
    {
        auto lk = std::lock_guard<std::mutex>{m_mutex};
        // emplace = insert-if-absent (first-writer-wins). A correlation_key is
        // unique per in-flight dispatch, so a collision should not occur; if one does,
        // keep the first (real) pairing rather than overwrite with a later spurious one.
        m_data.emplace(key, result);
    }
    // Notified outside the lock. One deposit wakes every waiter because the map is
    // keyed per dispatch and waiters are few (one per in-flight completion batch);
    // each re-checks its own key and sleeps again.
    m_cv.notify_all();
}

std::optional<kfd_timing_result>
ResultsMap::take(const correlation_key& key)
{
    auto lk = std::lock_guard<std::mutex>{m_mutex};
    return take_locked(key);
}

std::optional<kfd_timing_result>
ResultsMap::wait_take(const correlation_key& key, uint64_t deadline_ns)
{
    auto lk    = std::unique_lock<std::mutex>{m_mutex};
    auto ready = [this, &key]() { return m_abandoned || m_data.count(key) != 0; };
    if(deadline_ns != 0 && !ready()) m_cv.wait_until(lk, to_steady(deadline_ns), ready);
    return take_locked(key);
}

void
ResultsMap::abandon_waiters()
{
    {
        auto lk     = std::lock_guard<std::mutex>{m_mutex};
        m_abandoned = true;
    }
    m_cv.notify_all();
}

void
ResultsMap::note_hsa_fallback()
{
    ++m_fallbacks;
}

std::optional<kfd_timing_result>
ResultsMap::take_locked(const correlation_key& key)
{
    auto it = m_data.find(key);
    if(it == m_data.end())
    {
        ++m_misses;
        return std::nullopt;
    }
    auto result = it->second;
    m_data.erase(it);
    ++m_hits;
    return result;
}

ResultsMap::take_stats
ResultsMap::stats() const
{
    return take_stats{m_hits.load(), m_misses.load(), m_fallbacks.load()};
}

size_t
ResultsMap::erase_slot(uint32_t gpu_id, uint32_t doorbell_off)
{
    auto   lk      = std::lock_guard<std::mutex>{m_mutex};
    size_t erased  = 0;
    for(auto it = m_data.begin(); it != m_data.end();)
    {
        if(it->first.gpu_id == gpu_id && it->first.doorbell_off == doorbell_off)
        {
            it = m_data.erase(it);
            ++erased;
        }
        else
        {
            ++it;
        }
    }
    return erased;
}

size_t
ResultsMap::evict_stale(uint64_t now_ns, uint64_t max_age_ns)
{
    auto   lk      = std::lock_guard<std::mutex>{m_mutex};
    size_t evicted = 0;
    for(auto it = m_data.begin(); it != m_data.end();)
    {
        // Guard against a deposit stamped slightly in the future relative to
        // now_ns (clock skew): only evict when now_ns is strictly ahead.
        if(now_ns > it->second.deposited_at_ns &&
           (now_ns - it->second.deposited_at_ns) > max_age_ns)
        {
            it = m_data.erase(it);
            ++evicted;
        }
        else
        {
            ++it;
        }
    }
    return evicted;
}
}  // namespace kfd
}  // namespace rocprofiler
