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

#include "lib/rocprofiler-sdk/hsa/queue_state_registry.hpp"
#include "lib/common/static_object.hpp"

#include <memory>

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

queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue)
{
    queue_state_ptr_t result = {};
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second;
        }
    });
    return result;
}

queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    queue_state_ptr_t result = {};
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second.lock();
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    auto state = lookup_queue_state(queue);
    if(!state) return;

    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
}

bool
unregister_doorbell(hsa_signal_t doorbell)
{
    bool erased = false;
    get_doorbell_map().wlock(
        [&](auto& doorbell_map) { erased = (doorbell_map.erase(doorbell.handle) > 0); });
    return erased;
}

void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   uint64_t           k_factor)
{
    auto     state        = std::make_shared<QueueState>();
    uint64_t current_wdid = __atomic_load_n(wdid_addr, __ATOMIC_ACQUIRE);
    state->ring_view      = RingView{queue->base_address,
                                queue->size,
                                queue->size - 1,
                                /*pkt_size=*/64};
    state->stride          = static_cast<uint32_t>(1 + k_factor);
    state->real_wdid       = wdid_addr;
    state->real_rdid       = rdid_addr;
    state->hsa_queue       = queue;
    state->doorbell_signal = queue->doorbell_signal;
    state->claim_pos.store(current_wdid, std::memory_order_relaxed);
    state->published_pos = current_wdid;

    // Capture queue type so process_doorbell_impl can enforce the
    // monotonic-doorbell spec requirement for HSA_QUEUE_TYPE_SINGLE (bug #9).
    state->is_single = (queue->type == HSA_QUEUE_TYPE_SINGLE);
    // Initialize last_doorbell_val to the hardware's current write-index so a
    // racing early-exit caller that sees nothing claimed can never forward a
    // value smaller than what the hardware has already observed.
    state->last_doorbell_val.store(
        (current_wdid == 0) ? 0 : (current_wdid - 1), std::memory_order_relaxed);

    get_queue_registry().wlock([&](auto& map) { map[queue] = state; });
    get_doorbell_map().wlock([&](auto& map) { map[queue->doorbell_signal.handle] = state; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    hsa_signal_t      doorbell = {0};
    queue_state_ptr_t doomed   = {};

    // Phase 1: remove the state from both maps. After this point, no NEW
    // process_doorbell_impl invocation can find this queue/doorbell, so no new
    // scan can start for the doomed state.
    get_queue_registry().wlock([&](auto& map) {
        auto it = map.find(queue);
        if(it == map.end())
        {
            return;
        }
        doomed = it->second;
        if(doomed) doorbell = doomed->doorbell_signal;
        map.erase(it);
    });

    if(!doomed) return;
    if(doorbell.handle != 0) unregister_doorbell(doorbell);

    // Phase 2: drain any in-flight scan. A concurrent wrap_signal_store_* that
    // already resolved the doorbell lookup BEFORE Phase 1 erased the maps may
    // still be inside process_doorbell_impl, holding doomed->gate_lock. Take
    // and immediately release gate_lock so we block until that scan returns.
    // Because of Phase 1, no new scan can enter after this point.
    {
        std::lock_guard<std::mutex> drain{doomed->gate_lock};
    }

    // Phase 3: doomed's shared_ptr goes out of scope. If this was the last
    // strong reference, QueueState is destroyed now that no scan is in flight.
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
