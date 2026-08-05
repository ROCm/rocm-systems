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

#include <optional>

namespace rocprofiler
{
namespace kfd
{
queue_doorbell_entry
DoorbellMap::bind_locked(map_data&              data,
                         uint32_t               gpu_id,
                         rocprofiler_queue_id_t queue_id,
                         uint32_t               doorbell_off)
{
    // Preserve an existing generation for this (gpu, doorbell); default to 0.
    auto gen_key = std::make_pair(gpu_id, doorbell_off);
    auto gen_it  = data.generations.find(gen_key);
    auto gen     = (gen_it != data.generations.end()) ? gen_it->second : 0u;

    auto entry                     = queue_doorbell_entry{doorbell_off, gen, gpu_id};
    data.by_queue[queue_id.handle] = entry;
    data.generations[gen_key]      = gen;
    return entry;
}

queue_doorbell_entry
DoorbellMap::bind_and_resolve(uint32_t               gpu_id,
                              rocprofiler_queue_id_t queue_id,
                              uint32_t               doorbell_off)
{
    // Fast path: this queue is already bound to this exact doorbell_off. Steady
    // state for every dispatch after the first -- a single read lock, no mutation.
    auto fast = m_data.rlock([&](const auto& data) -> std::optional<queue_doorbell_entry> {
        auto it = data.by_queue.find(queue_id.handle);
        if(it == data.by_queue.end() || it->second.doorbell_off != doorbell_off ||
           it->second.gpu_id != gpu_id)
            return std::nullopt;
        return it->second;
    });
    if(fast) return *fast;

    // Slow path: first dispatch for this queue, or a rebind after doorbell reuse.
    return m_data.wlock(
        [&](auto& data) { return bind_locked(data, gpu_id, queue_id, doorbell_off); });
}

uint32_t
DoorbellMap::get_generation(uint32_t gpu_id, uint32_t doorbell_off) const
{
    return m_data.rlock([&](const auto& data) -> uint32_t {
        auto it = data.generations.find({gpu_id, doorbell_off});
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
        const uint32_t gpu_id       = it->second.gpu_id;

        data.by_queue.erase(it);

        // Bump generation so records that arrive on this doorbell after the
        // queue is gone are not paired with the destroyed queue's dispatches.
        // Keyed per GPU: slot numbers repeat across GPUs, so a shared counter
        // would invalidate another GPU's live dispatches on the same slot.
        data.generations[{gpu_id, doorbell_off}] += 1;
    });
}
}  // namespace kfd
}  // namespace rocprofiler
