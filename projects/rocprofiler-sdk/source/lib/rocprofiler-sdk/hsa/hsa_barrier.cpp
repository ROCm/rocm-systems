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

#include "lib/rocprofiler-sdk/hsa/hsa_barrier.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <functional>

namespace rocprofiler
{
namespace hsa
{
hsa_barrier::hsa_barrier(std::function<void()>&& finished, CoreApiTable core_api)
: _barrier_finished(std::move(finished))
, _core_api(core_api)
{
    // Create the barrier signal
    _core_api.hsa_signal_create_fn(0, 0, nullptr, &_barrier_signal);
}

hsa_barrier::~hsa_barrier()
{
    if(registration::get_fini_status() > 0)
    {
        return;
    }

    bool outstanding = false;
    _barrier_enqueued.rlock(
        [&](const auto& barrier_enqueued) { outstanding = !barrier_enqueued.empty(); });

    // retirement is gated on safe_to_destroy(), so a barrier should never be destroyed while a
    // transition packet still references its signal; log an error (and still release) if it does.
    ROCP_ERROR_IF(outstanding) << "hsa_barrier (handle: " << _barrier_signal.handle
                               << ") destroyed with outstanding transition packets";

    // release the signal so any stale transition packet can pass
    clear_barrier();

    // Destroy the barrier signal
    _core_api.hsa_signal_destroy_fn(_barrier_signal);
}

void
hsa_barrier::set_barrier(const queue_map_ptr_t& q)
{
    _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 1);
    ROCP_TRACE << "Barrier (handle: " << _barrier_signal.handle << ") is now set.";
    _queue_waiting.wlock([&](auto& queue_waiting) {
        for(const auto& [_, queue] : q)
        {
            queue->lock_queue([ptr = queue, &queue_waiting]() {
                if(ptr->active_async_packets() > 0)
                {
                    queue_waiting[ptr->get_id().handle] = ptr->active_async_packets();
                }
            });
        }
        if(queue_waiting.empty())
        {
            clear_barrier();
        }
    });
}

std::optional<rocprofiler_packet>
hsa_barrier::enqueue_packet(int64_t queue_id, uint64_t dispatch_id)
{
    if(complete()) return std::nullopt;
    bool return_block = false;
    _barrier_enqueued.wlock([&](auto& barrier_enqueued) {
        if(barrier_enqueued.find(queue_id) == barrier_enqueued.end())
        {
            return_block = true;
            // record the dispatch id carrying this transition packet; it has executed once the
            // queue's completed count reaches this id
            barrier_enqueued.emplace(queue_id, dispatch_id);
        }
    });

    if(!return_block) return std::nullopt;

    rocprofiler_packet barrier{};
    barrier.barrier_and.header        = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.barrier_and.dep_signal[0] = _barrier_signal;
    ROCP_INFO << "Barrier (handle: " << _barrier_signal.handle
              << ") added to queue (handle: " << queue_id << ")";
    return barrier;
}

void
hsa_barrier::drain_queue(int64_t queue_id, uint64_t completed_id)
{
    // transition packet can't have executed until the barrier clears; skip the lock while armed
    if(!complete()) return;
    _barrier_enqueued.wlock([&](auto& barrier_enqueued) {
        auto it = barrier_enqueued.find(queue_id);
        if(it == barrier_enqueued.end()) return;
        // packet executed once completions reach the carrying dispatch id (queue is in-order)
        if(completed_id >= it->second)
        {
            barrier_enqueued.erase(it);
        }
    });
}

void
hsa_barrier::remove_queue(const Queue* queue)
{
    // a torn-down queue can't execute its packet, so release its pin too
    _barrier_enqueued.wlock(
        [&](auto& barrier_enqueued) { barrier_enqueued.erase(queue->get_id().handle); });

    _queue_waiting.wlock([&](auto& queue_waiting) {
        if(queue_waiting.find(queue->get_id().handle) == queue_waiting.end()) return;
        queue_waiting.erase(queue->get_id().handle);
        if(queue_waiting.empty())
        {
            clear_barrier();
        }
    });
}

bool
hsa_barrier::register_completion(const Queue* queue)
{
    bool found = false;
    _queue_waiting.wlock([&](auto& queue_waiting) {
        if(queue_waiting.find(queue->get_id().handle) == queue_waiting.end()) return;
        found = true;
        queue_waiting[queue->get_id().handle]--;
        if(queue_waiting[queue->get_id().handle] == 0)
        {
            queue_waiting.erase(queue->get_id().handle);
            if(queue_waiting.empty())
            {
                clear_barrier();
            }
        }
    });
    return found;
}

bool
hsa_barrier::complete() const
{
    return _core_api.hsa_signal_load_scacquire_fn(_barrier_signal) == 0;
}

bool
hsa_barrier::safe_to_destroy() const
{
    if(!complete()) return false;
    bool no_outstanding = false;
    _barrier_enqueued.rlock(
        [&](const auto& barrier_enqueued) { no_outstanding = barrier_enqueued.empty(); });
    return no_outstanding;
}

void
hsa_barrier::clear_barrier()
{
    if(complete())
    {
        return;
    }
    if(_barrier_finished)
    {
        _barrier_finished();
    }
    _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 0);
    ROCP_TRACE << "Barrier (handle: " << _barrier_signal.handle << ") is now cleared.";
}

}  // namespace hsa
}  // namespace rocprofiler
