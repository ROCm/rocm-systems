// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
: _barried_finished(std::move(finished))
, _core_api(core_api)
{
    // Create the barrier signal
    _core_api.hsa_signal_create_fn(0, 0, nullptr, &_barrier_signal);
}

hsa_barrier::~hsa_barrier()
{
    // Destroy the barrier signal
    if(registration::get_fini_status() < 1)
    {
        _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 0);
        _core_api.hsa_signal_destroy_fn(_barrier_signal);
    }
}

void
hsa_barrier::set_barrier(const queue_map_ptr_t& q)
{
    _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 1);
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
            _barried_finished();
            _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 0);
        }
    });
}

std::optional<rocprofiler_packet>
hsa_barrier::enqueue_packet(const Queue* queue)
{
    if(_complete) return std::nullopt;
    bool return_block = false;
    _barrier_enqueued.wlock([&](auto& barrier_enqueued) {
        if(barrier_enqueued.find(queue->get_id().handle) == barrier_enqueued.end())
        {
            return_block = true;
            barrier_enqueued.insert(queue->get_id().handle);
        }
    });

    if(!return_block) return std::nullopt;

    rocprofiler_packet barrier{};
    barrier.barrier_and.header        = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.barrier_and.dep_signal[0] = _barrier_signal;
    ROCP_INFO << "Barrier Added: " << _barrier_signal.handle;
    return barrier;
}

void
hsa_barrier::remove_queue(const Queue* queue)
{
    _queue_waiting.wlock([&](auto& queue_waiting) {
        if(queue_waiting.find(queue->get_id().handle) == queue_waiting.end()) return;
        queue_waiting.erase(queue->get_id().handle);
        if(queue_waiting.empty())
        {
            _barried_finished();
            _complete = true;
            _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 0);
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
                _barried_finished();
                // We are done, release the barrier
                _complete = true;
                _core_api.hsa_signal_store_screlease_fn(_barrier_signal, 0);
            }
        }
    });
    return found;
}
}  // namespace hsa
}  // namespace rocprofiler
