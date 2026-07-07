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

#pragma once

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
class Queue;

class hsa_barrier
{
public:
    using queue_map_ptr_t = std::unordered_map<hsa_queue_t*, Queue*>;
    hsa_barrier(std::function<void()>&& finished, CoreApiTable core_api);
    ~hsa_barrier();

    // If a queue in q currently has active async packets, sets the barrier and saves how many
    // packets are currently in each queue to track barrier completion. If there are no queues with
    // packets, clears the barrier and marks it as complete.
    void set_barrier(const queue_map_ptr_t& q);

    // Returns the transition packet to insert in the queue identified by queue_id, stamping it with
    // dispatch_id (that queue's in-order serialized-dispatch id carrying the packet). Returns
    // nothing if the barrier is already complete or the queue was already handed a packet.
    std::optional<rocprofiler_packet> enqueue_packet(int64_t queue_id, uint64_t dispatch_id);

    // To be called when 1 async packet is finished in the given queue to decrement the remaining
    // packets counter. If all packets in all queues given in set_barrier are completed, clears the
    // barrier and marks it as complete. Returns true if the given queue had async packets waiting.
    bool register_completion(const Queue* queue);

    // Drop the queue from the outstanding set once its completed count (completed_id) reaches the
    // dispatch id that carried this barrier's transition packet (called on each serialized
    // completion).
    void drain_queue(int64_t queue_id, uint64_t completed_id);

    // Checks if this barrier is complete
    bool complete() const;

    // complete() AND no queue still references the signal -> safe to destroy (no handle reuse).
    bool safe_to_destroy() const;

    // Removes a queue from the barrier dependency list (waiting + outstanding transition packets).
    // If this is the last queue waiting, clears the barrier and marks it as complete.
    void remove_queue(const Queue* queue);

private:
    std::function<void()>                                      _barrier_finished = {};
    CoreApiTable                                               _core_api         = {};
    common::Synchronized<std::unordered_map<int64_t, int64_t>> _queue_waiting    = {};
    // queue id -> serialized-dispatch id carrying this barrier's transition packet
    common::Synchronized<std::unordered_map<int64_t, uint64_t>> _barrier_enqueued = {};

    void clear_barrier();

    // Blocks all queues from executing until the barrier is lifted
    hsa_signal_t _barrier_signal = {};
};

}  // namespace hsa
}  // namespace rocprofiler
