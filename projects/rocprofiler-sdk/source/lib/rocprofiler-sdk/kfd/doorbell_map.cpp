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

#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

namespace rocprofiler
{
namespace kfd
{
open_result
DoorbellMap::open_window(uint32_t               gpu_id,
                         rocprofiler_queue_id_t queue_id,
                         uint32_t               slot,
                         uint64_t               tick_sample)
{
    // The process has an owner it never windowed; first_owner is untrustworthy,
    // so open nothing anywhere for the rest of the process.
    if(signal_less_disabled()) return open_result{};

    return m_data.wlock([&](map_data& data) -> open_result {
        auto&    live  = data.by_slot[{gpu_id, slot}];
        uint64_t floor = 0;
        if(!live.empty())
        {
            // A REFERENCE, no shared_ptr copy: a copy would hold use_count()==2
            // across the prune and the predecessor would never be reclaimed.
            owner_window& prev = *live.back();
            if(prev.t_close.load(std::memory_order_acquire) == kWindowOpen)
                return open_result{nullptr, true};  // two live owners: overlap, open nothing
            floor = prev.t_close.load(std::memory_order_acquire);
            prev.superseded.store(true, std::memory_order_release);
        }

        // live.empty() is exactly "this process never opened a window on this
        // slot" and stays that way: pruning runs only after the push below.
        const bool first_owner = live.empty();

        auto w         = std::make_shared<owner_window>();
        w->slot        = slot;
        w->first_owner = first_owner;
        // The max is load-bearing: both clock reads happen outside any
        // shared lock, so a create sample can be observed after a later-real-time
        // destroy sample. Flooring at prev.t_close makes disjointness a property
        // of the data, not of the thread schedule.
        w->t_open = std::max(tick_sample, floor);
        // prev is not touched after this point; push_back may reallocate.
        live.push_back(w);
        data.by_queue[queue_id.handle] = w;

        // Prune closed windows nobody but this list references (their entries have
        // been GC'd). Bounded, O(list); never reaches the just-pushed live window.
        for(auto it = live.begin(); it != live.end();)
        {
            if((*it)->t_close.load(std::memory_order_acquire) != kWindowOpen &&
               it->use_count() == 1)
                it = live.erase(it);
            else
                ++it;
        }

        return open_result{std::move(w), false};
    });
}

std::optional<window_ptr>
DoorbellMap::resolve(uint32_t /*gpu_id*/, rocprofiler_queue_id_t queue_id)
{
    // Pure read-lock lookup plus the process-wide latch check (one atomic load,
    // no lock). The enqueue path never takes the write lock.
    if(signal_less_disabled()) return std::nullopt;
    return m_data.rlock([&](const map_data& data) -> std::optional<window_ptr> {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end()) return std::nullopt;
        return it->second;
    });
}

window_ptr
DoorbellMap::close_window(rocprofiler_queue_id_t queue_id,
                          uint64_t               tick_sample,
                          uint64_t               gc_deadline_ns)
{
    return m_data.wlock([&](map_data& data) -> window_ptr {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end()) return nullptr;  // never opened a window
        auto w = it->second;
        data.by_queue.erase(it);
        // t_close is stored LAST, with release, so any thread that observes a
        // closed window also observes its deadline.
        w->gc_deadline_ns = gc_deadline_ns;
        w->t_close.store(tick_sample, std::memory_order_release);
        return w;
    });
}
}  // namespace kfd
}  // namespace rocprofiler
