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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>

namespace rocprofiler
{
namespace hsa
{
/**
 * @brief Batches whose dispatch records are not out yet, by registration order.
 *
 * Batches complete in any order (different queues, short after long), so a count of finished
 * batches cannot tell whether a given one is done. This keeps the identity of each, so a buffer
 * flush can wait for exactly the batches added before it.
 */
class pending_batches
{
public:
    using clock_type    = std::chrono::steady_clock;
    using duration_type = clock_type::duration;

    // A batch still running on the device.
    uint64_t add()
    {
        auto lk = std::lock_guard<std::mutex>{m_mtx};
        auto id = m_next_id++;
        m_pending.emplace(id, false);
        return id;
    }

    // The device finished the batch and its records are being emitted. Id 0 is never tracked.
    void set_emitting(uint64_t id)
    {
        if(id == 0) return;
        {
            auto lk = std::lock_guard<std::mutex>{m_mtx};
            auto it = m_pending.find(id);
            if(it == m_pending.end() || it->second) return;
            it->second = true;
            ++m_transitions;
        }
        m_cv.notify_all();
    }

    // The batch's records are out.
    void remove(uint64_t id)
    {
        if(id == 0) return;
        {
            auto lk = std::lock_guard<std::mutex>{m_mtx};
            if(m_pending.erase(id) == 0) return;
            ++m_transitions;
        }
        m_cv.notify_all();
    }

    /**
     * @brief Wait until every batch added before this call is removed.
     *
     * Gives up once nothing has progressed for @p device_timeout while all of those batches are
     * still on the device (a dispatch that is still running), or for @p emit_timeout while some
     * are being emitted (a tool callback that does not return).
     *
     * @return whether every such batch was removed
     */
    bool wait_for_added(duration_type device_timeout, duration_type emit_timeout)
    {
        auto       lk            = std::unique_lock<std::mutex>{m_mtx};
        const auto target        = m_next_id;
        auto       seen          = m_transitions;
        auto       last_progress = clock_type::now();

        while(true)
        {
            const auto end = m_pending.lower_bound(target);
            if(m_pending.begin() == end) return true;

            const bool emitting =
                std::any_of(m_pending.begin(), end, [](const auto& itr) { return itr.second; });
            const auto deadline = last_progress + (emitting ? emit_timeout : device_timeout);

            m_cv.wait_until(lk, deadline, [&]() { return m_transitions != seen; });
            if(m_transitions != seen)
            {
                seen          = m_transitions;
                last_progress = clock_type::now();
            }
            else if(clock_type::now() >= deadline)
            {
                return false;
            }
        }
    }

    bool empty() const
    {
        auto lk = std::lock_guard<std::mutex>{m_mtx};
        return m_pending.empty();
    }

private:
    mutable std::mutex       m_mtx         = {};
    std::condition_variable  m_cv          = {};
    std::map<uint64_t, bool> m_pending     = {};
    uint64_t                 m_next_id     = 1;
    uint64_t                 m_transitions = 0;
};
}  // namespace hsa
}  // namespace rocprofiler
