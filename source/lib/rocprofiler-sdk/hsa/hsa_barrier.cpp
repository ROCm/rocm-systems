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
#include <sstream>

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

    // If this barrier is destroyed and has queued packets, a future signal creation may
    // inadvertently block these packets if the same handle is re-used by HSA.
    if(!complete())
    {
        _barrier_enqueued.rlock([&](auto& barrier_enqueued) {
            if(!barrier_enqueued.empty())
            {
                ROCP_WARNING << "An incomplete hsa_barrier which was enqueued is being destroyed.";
            }
        });
    }

    // Clear and destroy the barrier signal
    clear_barrier();
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
hsa_barrier::enqueue_packet(const Queue* queue)
{
    if(complete()) return std::nullopt;
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
    ROCP_INFO << "Barrier (handle: " << _barrier_signal.handle
              << ") added to queue (handle: " << queue->get_id().handle << ")";
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

std::string
hsa_barrier::to_string() const
{
    std::ostringstream oss;
    oss << "hsa_barrier{";
    
    // _barrier_finished (function pointer - just indicate if it's set)
    oss << "_barrier_finished: " << (_barrier_finished ? "set" : "null") << ", ";
    
    // _barrier_signal
    oss << "_barrier_signal: {handle: " << _barrier_signal.handle << "}, ";
    
    // _queue_waiting
    oss << "_queue_waiting: {";
    _queue_waiting.rlock([&](const auto& queue_waiting) {
        bool first = true;
        for(const auto& [queue_id, count] : queue_waiting) {
            if(!first) oss << ", ";
            oss << queue_id << ": " << count;
            first = false;
        }
    });
    oss << "}, ";
    
    // _barrier_enqueued
    oss << "_barrier_enqueued: {";
    _barrier_enqueued.rlock([&](const auto& barrier_enqueued) {
        bool first = true;
        for(const auto& queue_id : barrier_enqueued) {
            if(!first) oss << ", ";
            oss << queue_id;
            first = false;
        }
    });
    oss << "}";
    
    oss << "}";
    return oss.str();
}

}  // namespace hsa
}  // namespace rocprofiler
