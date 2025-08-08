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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa_barrier.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <fmt/format.h>

#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rocprofiler
{
namespace hsa
{
/**
 * HOW THE SERIALIZATION MECHANISM WORKS:
 * =====================================
 *
 * OVERVIEW:
 * The profiler_serializer ensures that only ONE kernel executes at a time across
 * ALL profiled queues on a device. This is critical for accurate hardware counter measurements.
 *
 * KEY DATA STRUCTURES:
 * - _dispatch_queue: Pointer to the currently executing queue (nullptr = none executing)
 * - _enqueued_kernels: FIFO deque of queues with kernels enqueued for execution
 * - _ready_queues: Multiset of queues that have reached their blocking barrier and are ready
 * - _seen_queues: Set tracking which queues have ready_signal handlers registered
 * - _barrier: List of active synchronization barriers
 * - block_signal: Per-queue HSA signal controlling kernel execution permission
 * - ready_signal: Per-queue HSA signal indicating queue has reached blocking barrier
 *
 * EXECUTION FLOW:
 *
 * 1. KERNEL SUBMISSION (kernel_dispatch):
 *    - Generate barrier packets that depend on queue.block_signal
 *    - If no queue executing: Grant immediate permission (block_signal = 0)
 *    - If queue executing: Add to _enqueued_kernels list (block_signal remains 1)
 *    - Register ready_signal handler for new queues (triggers at -1)
 *
 * 2. BARRIER PACKETS INJECTED INTO QUEUE:
 *    - HSA barrier packets are inserted before the actual kernel
 *    - When barrier is reached, ready_signal is set to -1 (triggers queue_ready)
 *    - Barriers wait for block_signal == 0 before proceeding
 *    - If block_signal == 1, kernels are blocked at hardware level
 *
 * 3. READY SIGNAL NOTIFICATION (queue_ready):
 *    - Called when queue reaches blocking barrier (ready_signal == -1)
 *    - Reset ready_signal to 0
 *    - Add queue to _ready_queues multiset
 *    - If no queue executing: Find first queue in _enqueued_kernels that's also in _ready_queues
 *    - Grant execution to selected queue if found
 *
 * 4. KERNEL COMPLETION (kernel_completion_signal):
 *    - Current queue completes: Set its block_signal = 1 (block further kernels)
 *    - Find first queue in _enqueued_kernels that's also in _ready_queues
 *    - If found: Remove from both lists, set block_signal = 0, update _dispatch_queue
 *    - If not found: Set _dispatch_queue = nullptr
 *
 * 5. HARDWARE SYNCHRONIZATION:
 *    - HSA barrier packets provide hardware-level synchronization
 *    - Ready signal provides notification when barriers are reached
 *    - No software polling or busy-waiting required
 *    - Automatic wake-up when signal conditions are met
 *
 * SERIALIZATION GUARANTEES:
 * - Only one kernel executes across all profiled queues
 * - Only queues that have reached their barrier can execute
 * - FIFO ordering with ready-state verification
 * - Hardware-level blocking prevents race conditions
 * - Automatic cleanup when queues are destroyed
 *
 * EXAMPLE EXECUTION SEQUENCE:
 * 1. Queue A submits kernel → Gets immediate execution (_dispatch_queue = A, A.block_signal = 0)
 * 2. Queue B submits kernel → Added to _enqueued_kernels (B.block_signal = 1)
 * 3. Queue B reaches barrier → Added to _ready_queues (B.ready_signal triggers)
 * 4. Queue C submits kernel → Added to _enqueued_kernels (C.block_signal = 1)
 * 5. Queue A completes → Checks _enqueued_kernels for ready queues
 * 6. Queue B selected → B.block_signal = 0, _dispatch_queue = B (C still waiting)
 * 7. Queue C reaches barrier → Added to _ready_queues
 * 8. Queue B completes → Queue C selected and executes
 */
class profiler_serializer
{
public:
    enum class Status
    {
        ENABLED,
        DISABLED,
    };

    struct barrier_with_state
    {
        barrier_with_state(Status _state, std::unique_ptr<hsa_barrier> _barrier)
        : state(_state)
        , barrier(std::move(_barrier))
        {}
        Status                       state;
        std::unique_ptr<hsa_barrier> barrier;
    };

    void kernel_completion_signal(const Queue&);
    // Signal a kernel dispatch is taking place, generates packets needed to be
    // inserted to support kernel dispatch
    common::container::small_vector<hsa::rocprofiler_packet, 3> kernel_dispatch(const Queue&);

    // Signal that a queue has reached the blocking barrier and is ready to execute
    void queue_ready(hsa_queue_t* hsa_queue, const Queue& queue);

    // Check if queue has been seen before and set up signal handler if new
    void register_queue_ready_handler(const Queue& queue);

    // Enable the serializer
    void enable(const hsa_barrier::queue_map_ptr_t& queues);
    // Disable the serializer
    void disable(const hsa_barrier::queue_map_ptr_t& queues);

    void destroy_queue(hsa_queue_t* id, const Queue& queue);

    // Returns a string containing all class variable data
    std::string to_string() const;

private:
    // Extract common queue selection and execution logic
    const Queue* select_and_grant_ready_queue();

    const Queue*             _dispatch_queue{nullptr};
    std::deque<const Queue*> _enqueued_kernels;  // FIFO queue of kernels enqueued for execution
    std::unordered_multiset<const Queue*> _ready_queues;  // Queues that have signaled ready
    std::unordered_set<uint64_t>   _seen_queues;  // Track queue IDs we've registered handlers for
    std::atomic<Status>            _serializer_status{Status::DISABLED};
    std::deque<barrier_with_state> _barrier;
    mutable std::atomic<int64_t>   _enqueued_packets{0};
    mutable std::atomic<int64_t>   _completed_packets{0};
};

}  // namespace hsa
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for profiler_serializer
template <>
struct formatter<rocprofiler::hsa::profiler_serializer>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::hsa::profiler_serializer const& serializer, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", serializer.to_string());
    }
};
}  // namespace fmt
