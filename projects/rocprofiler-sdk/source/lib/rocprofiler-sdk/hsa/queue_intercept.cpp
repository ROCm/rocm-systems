// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/common/static_object.hpp"

#include <cstring>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

doorbell_map_t&
get_doorbell_map()
{
    static auto*& _v = common::static_object<doorbell_map_t>::construct();
    return *_v;
}

QueueState*
lookup_queue_state(const hsa_queue_t* queue)
{
    QueueState* result = nullptr;
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second.get();
        }
    });
    return result;
}

QueueState*
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    QueueState* result = nullptr;
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second;
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    QueueState* state = lookup_queue_state(queue);
    if(state)
    {
        get_doorbell_map().wlock(
            [&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
    }
}

void
unregister_doorbell(hsa_signal_t doorbell)
{
    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map.erase(doorbell.handle); });
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value)
{
    return state->virtual_wptr.fetch_add(value, std::memory_order_relaxed);
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    state->virtual_wptr.store(value, std::memory_order_relaxed);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    return state->virtual_wptr.load(std::memory_order_relaxed);
}

void
sync_metadata_impl(QueueState* compute_state,
                   const hsa_kernel_dispatch_packet_t* /*pkt*/,
                   uint64_t /*dest_pos*/)
{
    auto* meta = compute_state->metadata_state;
    if(!meta) return;

    uint64_t meta_dest    = meta->next_submit_pos;
    meta->next_submit_pos = meta_dest + 1 + compute_state->k_factor;
    __atomic_store_n(meta->real_wdid, meta->next_submit_pos, __ATOMIC_RELEASE);
}

void
process_doorbell_impl(QueueState* state, hsa_signal_value_t value, doorbell_fn_t ring_doorbell)
{
    std::lock_guard<std::mutex> lock(state->gate_lock);

    uint64_t scan_end = state->virtual_wptr.load(std::memory_order_acquire);
    uint64_t scan_pos = state->next_scan_pos;

    for(uint64_t i = scan_pos; i < scan_end; i++)
    {
        auto* src = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(state->ring_buf) +
                    (i & state->ring_mask);

        uint64_t dest_idx = state->next_submit_pos;
        auto*    dst      = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(state->ring_buf) +
                    (dest_idx & state->ring_mask);

        if(dst != src)
        {
            memcpy(dst, src, 64);
        }

        state->next_submit_pos = dest_idx + 1 + state->k_factor;

        if(state->metadata_state)
        {
            sync_metadata_impl(state, src, dest_idx);
        }
    }

    state->next_scan_pos = scan_end;
    __atomic_store_n(state->real_wdid, state->next_submit_pos, __ATOMIC_RELEASE);
    ring_doorbell(state->doorbell_signal, value);
}

namespace
{
uint32_t
next_power_of_two(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
}  // namespace

uint32_t
compute_inflated_ring_size(uint32_t requested_size, uint64_t k_factor)
{
    if(k_factor == 0) return requested_size;
    uint64_t inflated = static_cast<uint64_t>(requested_size) * (1 + k_factor) * 2;
    if(inflated > 262144) inflated = 262144;
    return next_power_of_two(static_cast<uint32_t>(inflated));
}

void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   uint64_t           k_factor)
{
    auto state             = std::make_unique<QueueState>();
    state->ring_buf        = reinterpret_cast<void*>(queue->base_address);
    state->ring_size       = queue->size;
    state->ring_mask       = queue->size - 1;
    state->real_wdid       = wdid_addr;
    state->real_rdid       = rdid_addr;
    state->doorbell_signal = queue->doorbell_signal;
    state->k_factor        = k_factor;

    auto* raw_ptr = state.get();
    get_queue_registry().wlock([&](auto& map) { map[queue] = std::move(state); });
    get_doorbell_map().wlock([&](auto& map) { map[queue->doorbell_signal.handle] = raw_ptr; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    auto* state = lookup_queue_state(queue);
    if(!state) return;

    unregister_doorbell(state->doorbell_signal);
    get_queue_registry().wlock([&](auto& map) { map.erase(queue); });
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
