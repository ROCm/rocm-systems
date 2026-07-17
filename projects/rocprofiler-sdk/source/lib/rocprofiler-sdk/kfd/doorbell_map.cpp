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
void
DoorbellMap::bind(rocprofiler_queue_id_t queue_id, uint32_t doorbell_off)
{
    m_data.wlock([&](auto& data) {
        // Preserve an existing generation for this doorbell; default to 0.
        auto gen_it = data.generations.find(doorbell_off);
        auto gen    = (gen_it != data.generations.end()) ? gen_it->second : 0u;

        data.by_queue[queue_id.handle] = queue_doorbell_entry{doorbell_off, gen};
        data.by_doorbell[doorbell_off] = queue_id.handle;
        data.generations[doorbell_off] = gen;
        data.uncertain.erase(doorbell_off);  // confirmed live
    });
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

bool
DoorbellMap::note_pending_dispatch(rocprofiler_queue_id_t queue_id, uint32_t dispatch_idx_low32)
{
    return m_data.wlock([&](auto& data) {
        // Already bound? Caller can build the key directly; no hint needed.
        if(data.by_queue.find(queue_id.handle) != data.by_queue.end()) return true;

        // Unbound: record the hint so the reader can bind this queue's doorbell
        // when it sees the matching record.
        data.pending_index[dispatch_idx_low32] = queue_id.handle;
        return false;
    });
}

bool
DoorbellMap::bind_from_record(uint32_t doorbell_off, uint32_t dispatch_id)
{
    return m_data.wlock([&](auto& data) {
        // Already bound: nothing to do.
        if(data.by_doorbell.find(doorbell_off) != data.by_doorbell.end()) return true;

        // Find the queue that enqueued this dispatch_id (recorded by capture).
        auto it = data.pending_index.find(dispatch_id);
        if(it == data.pending_index.end()) return false;  // no hint yet

        const uint64_t queue_handle = it->second;

        // Bind, preserving any existing generation for this doorbell.
        auto gen_it = data.generations.find(doorbell_off);
        auto gen    = (gen_it != data.generations.end()) ? gen_it->second : 0u;

        data.by_queue[queue_handle]    = queue_doorbell_entry{doorbell_off, gen};
        data.by_doorbell[doorbell_off] = queue_handle;
        data.generations[doorbell_off] = gen;
        data.uncertain.erase(doorbell_off);
        data.pending_index.erase(it);
        return true;
    });
}

bool
DoorbellMap::is_bound(uint32_t doorbell_off) const
{
    return m_data.rlock([&](const auto& data) {
        return data.by_doorbell.find(doorbell_off) != data.by_doorbell.end();
    });
}
}  // namespace kfd
}  // namespace rocprofiler
