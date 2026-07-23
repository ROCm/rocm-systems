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

namespace rocprofiler
{
namespace kfd
{
queue_doorbell_entry
DoorbellMap::bind_locked(map_data& data, rocprofiler_queue_id_t queue_id, uint32_t doorbell_off)
{
    // Preserve an existing generation for this doorbell; default to 0.
    auto gen_it = data.generations.find(doorbell_off);
    auto gen    = (gen_it != data.generations.end()) ? gen_it->second : 0u;

    auto entry                     = queue_doorbell_entry{doorbell_off, gen};
    data.by_queue[queue_id.handle] = entry;
    data.by_doorbell[doorbell_off] = queue_id.handle;
    data.generations[doorbell_off] = gen;
    data.uncertain.erase(doorbell_off);  // confirmed live
    return entry;
}

queue_doorbell_entry
DoorbellMap::bind_and_resolve(rocprofiler_queue_id_t queue_id, uint32_t doorbell_off)
{
    // Fast path: this queue is already bound to this exact doorbell_off and the
    // doorbell is certain (not awaiting a post-destroy rebind). Steady state for
    // every dispatch after the first -- a single read lock, no mutation.
    auto fast = m_data.rlock([&](const auto& data) -> std::optional<queue_doorbell_entry> {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end() || it->second.doorbell_off != doorbell_off)
            return std::nullopt;
        if(data.uncertain.find(doorbell_off) != data.uncertain.end()) return std::nullopt;
        return it->second;
    });
    if(fast) return *fast;

    // Slow path: first dispatch for this queue, or a rebind after doorbell reuse.
    // Take the write lock and bind (which also clears the uncertain mark).
    return m_data.wlock([&](auto& data) { return bind_locked(data, queue_id, doorbell_off); });
}

std::optional<queue_doorbell_entry>
DoorbellMap::get_by_queue(rocprofiler_queue_id_t queue_id) const
{
    return m_data.rlock([&](const auto& data) -> std::optional<queue_doorbell_entry> {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end()) return std::nullopt;
        return it->second;
    });
}

uint32_t
DoorbellMap::get_generation(uint32_t doorbell_off) const
{
    return m_data.rlock([&](const auto& data) -> uint32_t {
        auto it = data.generations.find(doorbell_off);
        return (it != data.generations.end()) ? it->second : 0u;
    });
}

void
DoorbellMap::on_queue_destroyed(rocprofiler_queue_id_t queue_id)
{
    m_data.wlock([&](auto& data) {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end()) return;

        const uint32_t doorbell_off = it->second.doorbell_off;

        data.by_queue.erase(it);
        data.by_doorbell.erase(doorbell_off);

        // Bump generation so records that arrive on this doorbell after the
        // queue is gone are not paired with the destroyed queue's dispatches.
        data.generations[doorbell_off] += 1;
        data.uncertain.insert(doorbell_off);
    });
}

bool
DoorbellMap::is_generation_certain(uint32_t doorbell_off) const
{
    return m_data.rlock([&](const auto& data) {
        return data.uncertain.find(doorbell_off) == data.uncertain.end();
    });
}

}  // namespace kfd
}  // namespace rocprofiler
