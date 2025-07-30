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

#include "lib/rocprofiler-sdk/hsa/profile_serializer.hpp"

#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <sstream>

namespace rocprofiler
{
namespace hsa
{
namespace
{
bool
profiler_serializer_ready_signal_handler(hsa_signal_value_t /* signal_value */, void* data)
{
    auto*       hsa_queue = static_cast<hsa_queue_t*>(data);
    const auto* queue     = CHECK_NOTNULL(get_queue_controller())->get_queue(*hsa_queue);
    CHECK(queue);
    CHECK_NOTNULL(get_queue_controller())->serializer(queue).wlock([&](auto& serializer) {
        serializer.queue_ready(hsa_queue, *queue);
    });
    return true;
}

void
clear_complete_barriers(std::deque<profiler_serializer::barrier_with_state>& barriers)
{
    while(!barriers.empty())
    {
        if(barriers.front().barrier->complete())
        {
            barriers.pop_front();
        }
        else
        {
            break;
        }
    }
}

}  // namespace

void
profiler_serializer::add_queue(hsa_queue_t** hsa_queues, const Queue& queue)
{
    hsa_signal_t signal = queue.ready_signal;
    hsa_status_t status =
        CHECK_NOTNULL(get_queue_controller())
            ->get_ext_table()
            .hsa_amd_signal_async_handler_fn(signal,
                                             HSA_SIGNAL_CONDITION_EQ,
                                             -1,
                                             profiler_serializer_ready_signal_handler,
                                             *hsa_queues);
    if(status != HSA_STATUS_SUCCESS) ROCP_FATAL << "hsa_amd_signal_async_handler failed";
}

void
profiler_serializer::kernel_completion_signal(const Queue& completed)
{
    // We do not want to track kernel compleiton signals before we have reached the barrier
    clear_complete_barriers(_barrier);
    _completed_packets++;
    // Find the state of this barrier
    auto state = _serializer_status.load();
    bool found = false;
    for(auto& barrier : _barrier)
    {
        // Register completion of the kernel. Each queue has a number of kernels it is
        // waiting on to complete for each barrier. If more than one barrier is present
        // that has this queue, then it will contain a count that is the sum of all previous
        // kernel packets in the queue. Thus we must register completion with every barrier.
        // The state of the queue at this time is the state of the first barrier (or the state
        // of the serializer if no barriers are present).
        if(barrier.barrier->register_completion(&completed) && !found)
        {
            state = barrier.state;
            found = true;
        }
    }

    if(state == Status::DISABLED) return;

    CHECK(_dispatch_queue);
    _dispatch_queue = nullptr;
    CHECK_NOTNULL(get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(completed.block_signal, 1);
    CHECK_NOTNULL(get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(completed.ready_signal, 0);
    if(!_dispatch_ready.empty())
    {
        const auto* queue = _dispatch_ready.front();
        _dispatch_ready.erase(_dispatch_ready.begin());
        CHECK_NOTNULL(get_queue_controller())
            ->get_core_table()
            .hsa_signal_store_screlease_fn(queue->block_signal, 0);
        _dispatch_queue = queue;
    }
}

void
profiler_serializer::queue_ready(hsa_queue_t* hsa_queue, const Queue& queue)
{
    {
        ROCP_TRACE << "Obtaining queue mutex lock...";
        std::lock_guard<std::mutex> cv_lock(queue.cv_mutex);
        ROCP_TRACE << "Queue mutex lock obtained";
        if(queue.get_state() == queue_state::to_destroy)
        {
            ROCP_TRACE << "Setting queue state to done_destroy...";
            CHECK_NOTNULL(get_queue_controller())
                ->set_queue_state(queue_state::done_destroy, hsa_queue);
            ROCP_TRACE << "Destroying ready signal...";
            CHECK_NOTNULL(get_queue_controller())
                ->get_core_table()
                .hsa_signal_destroy_fn(queue.ready_signal);
            ROCP_TRACE << "Notifying queue condition variable...";
            queue.cv_ready_signal.notify_one();
            return;
        }
    }

    ROCP_TRACE << "setting queue ready signal to 1...";

    CHECK_NOTNULL(get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(queue.ready_signal, 1);

    if(_dispatch_queue == nullptr)
    {
        CHECK_NOTNULL(get_queue_controller())
            ->get_core_table()
            .hsa_signal_store_screlease_fn(queue.block_signal, 0);
        _dispatch_queue = &queue;
    }
    else
    {
        _dispatch_ready.push_back(&queue);
    }
}

common::container::small_vector<hsa::rocprofiler_packet, 3>
profiler_serializer::kernel_dispatch(const Queue& queue) const
{
    common::container::small_vector<hsa::rocprofiler_packet, 3> ret;
    auto&& CreateBarrierPacket = [](hsa_signal_t* dependency_signal,
                                    hsa_signal_t* completion_signal) {
        hsa::rocprofiler_packet barrier{};
        barrier.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.barrier_and.header |= HSA_FENCE_SCOPE_SYSTEM
                                      << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        barrier.barrier_and.header |= HSA_FENCE_SCOPE_SYSTEM
                                      << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
        barrier.barrier_and.header |= 1 << HSA_PACKET_HEADER_BARRIER;
        if(dependency_signal != nullptr) barrier.barrier_and.dep_signal[0] = *dependency_signal;
        if(completion_signal != nullptr) barrier.barrier_and.completion_signal = *completion_signal;
        return barrier;
    };
    _enqueued_packets++;
    if(!_barrier.empty())
    {
        if(auto maybe_barrier = _barrier.back().barrier->enqueue_packet(&queue))
        {
            ret.push_back(*maybe_barrier);
        }
    }

    switch(_serializer_status)
    {
        case Status::DISABLED: return ret;
        case Status::ENABLED:
        {
            hsa_signal_t ready_signal = queue.ready_signal;
            hsa_signal_t block_signal = queue.block_signal;
            ret.push_back(CreateBarrierPacket(&ready_signal, &ready_signal));
            ret.push_back(CreateBarrierPacket(&block_signal, &block_signal));
            break;
        };
    }
    return ret;
}

void
profiler_serializer::destroy_queue(hsa_queue_t* id, const Queue& queue)
{
    ROCP_INFO << "destroying queue...";

    /*Deletes the queue to be destructed from the dispatch ready.*/
    for(auto& barriers : _barrier)
    {
        barriers.barrier->remove_queue(&queue);
    }

    _dispatch_ready.erase(
        std::remove_if(
            _dispatch_ready.begin(),
            _dispatch_ready.end(),
            [&](auto& it) {
                /*Deletes the queue to be destructed from the dispatch ready.*/
                if(it->get_id().handle == queue.get_id().handle)
                {
                    if(_dispatch_queue && _dispatch_queue->get_id().handle == queue.get_id().handle)
                    {
                        // insert fatal condition here
                        // ToDO [srnagara]: Need to find a solution rather than abort.
                        ROCP_FATAL
                            << "Queue is being destroyed while kernel launch is still active";
                    }
                    return true;
                }
                return false;
            }),
        _dispatch_ready.end());
    CHECK_NOTNULL(get_queue_controller())->set_queue_state(queue_state::to_destroy, id);
    CHECK_NOTNULL(get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(queue.ready_signal, 0);

    ROCP_INFO << "queue destroyed";
}

// Enable the serializer
void
profiler_serializer::enable(const hsa_barrier::queue_map_ptr_t& queues)
{
    if(_serializer_status == Status::ENABLED) return;

    ROCP_INFO << "Enabling profiler serialization...";

    _serializer_status = Status::ENABLED;
    if(queues.empty()) return;

    clear_complete_barriers(_barrier);

    _barrier.emplace_back(Status::DISABLED,
                          std::make_unique<hsa_barrier>(
                              [] {}, CHECK_NOTNULL(get_queue_controller())->get_core_table()));
    _serializer_status = Status::ENABLED;

    _barrier.back().barrier->set_barrier(queues);
    ROCP_INFO << "Profiler serialization enabled";
}

// Disable the serializer
void
profiler_serializer::disable(const hsa_barrier::queue_map_ptr_t& queues)
{
    if(_serializer_status == Status::DISABLED) return;

    ROCP_INFO << "Disabling profiler serialization...";

    _serializer_status = Status::DISABLED;
    if(queues.empty()) return;

    clear_complete_barriers(_barrier);

    _barrier.emplace_back(Status::ENABLED,
                          std::make_unique<hsa_barrier>(
                              [] {}, CHECK_NOTNULL(get_queue_controller())->get_core_table()));
    _serializer_status = Status::DISABLED;

    _barrier.back().barrier->set_barrier(queues);
    ROCP_INFO << "Profiler serialization disabled";
}

std::string
profiler_serializer::to_string() const
{
    std::ostringstream oss;
    oss << "profiler_serializer{";

    // _dispatch_queue
    oss << "_dispatch_queue: ";
    if(_dispatch_queue)
    {
        oss << "{id: " << _dispatch_queue->get_id().handle << "}";
    }
    else
    {
        oss << "null";
    }
    oss << ", ";

    // _dispatch_ready
    oss << "_dispatch_ready: {";
    oss << "count: " << _dispatch_ready.size();
    if(!_dispatch_ready.empty())
    {
        oss << ", queue_ids: [";
        bool first = true;
        for(const auto* queue : _dispatch_ready)
        {
            if(!first) oss << ", ";
            oss << (queue ? queue->get_id().handle : 0);
            first = false;
        }
        oss << "]";
    }
    oss << "}, ";

    // _serializer_status
    oss << "_serializer_status: ";
    switch(_serializer_status.load())
    {
        case Status::ENABLED: oss << "ENABLED"; break;
        case Status::DISABLED: oss << "DISABLED"; break;
        default: oss << "UNKNOWN"; break;
    }
    oss << ", ";

    // _barrier
    oss << "_barrier: {";
    oss << "count: " << _barrier.size();
    if(!_barrier.empty())
    {
        oss << ", barriers: [";
        bool first = true;
        for(const auto& barrier : _barrier)
        {
            if(!first) oss << ", ";
            oss << "{state: ";
            switch(barrier.state)
            {
                case Status::ENABLED: oss << "ENABLED"; break;
                case Status::DISABLED: oss << "DISABLED"; break;
                default: oss << "UNKNOWN"; break;
            }
            oss << ", complete: " << (barrier.barrier ? barrier.barrier->complete() : false) << "}";
            first = false;
        }
        oss << "]";
    }
    oss << "}";
    oss << " Completed Packets: " << _completed_packets.load() << ", ";
    oss << "Enqueued Packets: " << _enqueued_packets.load();
    oss << "}";
    return oss.str();
}

}  // namespace hsa
}  // namespace rocprofiler
