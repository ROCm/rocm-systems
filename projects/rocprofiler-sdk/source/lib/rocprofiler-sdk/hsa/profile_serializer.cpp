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

/**
 * @file profile_serializer.cpp
 * @brief HSA Kernel Execution Serialization for Profiling
 *
 * This file implements kernel execution serialization to ensure only one kernel
 * executes at a time across all profiled queues. This is necessary for accurate
 * profiling measurements, especially for hardware performance counters.
 *
 * SERIALIZATION MECHANISM OVERVIEW:
 * ================================
 *
 * The profiler_serializer implements a two-stage queue-based serialization system:
 *
 * 1. **Single Executor Rule**: Only one queue can execute kernels at any given time
 * 2. **Ready-State Verification**: Queues must signal ready before they can execute
 * 3. **FIFO Ordering with Ready Check**: Queues are granted execution in FIFO order,
 *    but only if they have reached their blocking barrier
 * 4. **Dual-Signal Control**: Uses block_signal for execution control and ready_signal
 *    for barrier notification
 *
 * KEY COMPONENTS:
 * - _dispatch_queue: Currently executing queue (nullptr = no queue executing)
 * - _enqueued_kernels: FIFO queue of kernels enqueued for execution
 * - _ready_queues: Multiset of queues that have reached their blocking barrier
 * - _seen_queues: Tracks which queues have ready_signal handlers registered
 * - _barrier: List of active barriers (transitions to/from serialized execution)
 * - block_signal: HSA signal controlling kernel execution (0=execute, 1=blocked)
 * - ready_signal: HSA signal indicating queue reached barrier (-1 triggers handler)
 *
 * EXECUTION FLOW:
 * 1. Kernel submitted → Queue added to _enqueued_kernels
 * 2. Barrier reached → ready_signal triggers, queue added to _ready_queues
 * 3. If no queue executing → Select first queue in _enqueued_kernels that's also ready
 * 4. Kernel completes → Select next ready queue from _enqueued_kernels
 * 5. Repeat until no ready queues available
 *
 * SIGNAL VALUES:
 * - block_signal: RELEASE_BARRIER (0) allows execution, STOP_BARRIER (1) blocks
 * - ready_signal: -1 triggers handler, reset to 0 after processing
 */
#include "lib/rocprofiler-sdk/hsa/profile_serializer.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <sstream>

namespace rocprofiler
{
namespace hsa
{
namespace
{
/**
 * @brief Signal value to release/unblock barrier packets
 *
 * When a queue's block_signal is set to RELEASE_BARRIER (0), the HSA barrier
 * packets depending on this signal will proceed, allowing kernels to execute.
 */
constexpr int64_t RELEASE_BARRIER = 0;

/**
 * @brief Signal value to stop/block barrier packets
 *
 * When a queue's block_signal is set to STOP_BARRIER (1), the HSA barrier
 * packets depending on this signal will be blocked, preventing kernel execution.
 */
constexpr int64_t STOP_BARRIER = 1;

/**
 * @brief Remove completed barriers from the barrier queue
 *
 * This function cleans up barriers that have finished their synchronization.
 * Completed barriers are removed from the front of the deque to maintain
 * proper barrier lifecycle management.
 *
 * @param barriers Reference to the barrier deque to clean up
 */
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

}  // namespace

/**
 * @brief Select the first ready queue and grant it execution permission
 *
 * This function encapsulates the common logic for finding the first queue
 * in _dispatch_ready that is also in _ready_queues, removing it from both
 * containers, and granting it execution permission.
 *
 * @return Pointer to the selected queue, or nullptr if no ready queue found
 */
const Queue*
profiler_serializer::select_and_grant_ready_queue()
{
    ROCP_INFO << "[select_and_grant] Starting - _enqueued_kernels.size=" << _enqueued_kernels.size()
              << ", _ready_queues.size=" << _ready_queues.size();

    if(_enqueued_kernels.empty())
    {
        ROCP_INFO << "[select_and_grant] No enqueued kernels, returning nullptr";
        return nullptr;
    }

    const auto& controller = *CHECK_NOTNULL(get_queue_controller());

    // Find the first queue in _enqueued_kernels that is also ready
    for(auto it = _enqueued_kernels.begin(); it != _enqueued_kernels.end(); ++it)
    {
        auto queue_id    = (*it)->get_id().handle;
        auto ready_count = _ready_queues.count(*it);

        ROCP_INFO << "[select_and_grant] Checking queue " << queue_id
                  << " - ready_count=" << ready_count;

        if(ready_count > 0)
        {
            const Queue* queue_to_run = *it;

            // Remove from both containers
            _enqueued_kernels.erase(it);
            _ready_queues.erase(_ready_queues.find(queue_to_run));

            // Grant execution permission
            _dispatch_queue = queue_to_run;
            controller.get_core_table().hsa_signal_store_screlease_fn(queue_to_run->block_signal,
                                                                      RELEASE_BARRIER);

            ROCP_INFO << "[select_and_grant] Queue " << queue_to_run->get_id().handle
                      << " SELECTED AND GRANTED EXECUTION"
                      << ", block_signal set to 0"
                      << ", _enqueued_kernels.size now=" << _enqueued_kernels.size()
                      << ", _ready_queues.size now=" << _ready_queues.size();

            return queue_to_run;
        }
    }

    ROCP_INFO << "[select_and_grant] No ready queue found in _enqueued_kernels";
    return nullptr;
}

/**
 * @brief Handle kernel completion and manage dispatch queue transitions
 *
 * This is the core function that handles the serialization state machine when
 * a kernel completes execution. It performs the following critical operations:
 *
 * 1. Clean up completed barriers
 * 2. Update packet completion counters
 * 3. Determine the current serialization state
 * 4. Block the completed queue from further execution
 * 5. Find the first queue in _enqueued_kernels that is also in _ready_queues
 * 6. Grant execution permission to the selected ready queue (if found)
 * 7. Update the dispatch queue pointer
 *
 * SERIALIZATION STATE MACHINE:
 * Current queue completes → Block current queue → Find ready queue → Grant permission
 *
 * The key difference from simple FIFO is that we only grant execution to queues
 * that have signaled they are ready (reached their blocking barrier).
 *
 * @param completed The queue that just finished executing a kernel
 */
void
profiler_serializer::kernel_completion_signal(const Queue& completed)
{
    ROCP_INFO << "[kernel_completion] Queue " << completed.get_id().handle
              << " completed, _dispatch_queue="
              << (_dispatch_queue ? _dispatch_queue->get_id().handle : 0)
              << ", _enqueued_kernels.size=" << _enqueued_kernels.size()
              << ", _ready_queues.size=" << _ready_queues.size();

    // Clean up any barriers that have completed their synchronization
    clear_complete_barriers(_barrier);

    // Track the number of completed packets for debugging/monitoring
    _completed_packets++;

    // Determine the current serialization state from active barriers
    // If no barriers are active, use the serializer's global state
    auto state = _serializer_status.load();
    bool found = false;
    // Check all active barriers to see if this completion affects them
    for(auto& barrier : _barrier)
    {
        // Register completion of the kernel with each barrier
        // Each barrier tracks how many kernels from each queue are still pending
        // If multiple barriers exist, they accumulate the counts from previous barriers
        // The state of the first barrier that recognizes this queue determines our action
        if(barrier.barrier->register_completion(&completed) && !found)
        {
            // Use the state from the first barrier that recognized this completion
            state = barrier.state;
            found = true;
        }
    }

    // If serialization is disabled, don't manage dispatch queue transitions
    if(state == Status::DISABLED)
    {
        ROCP_INFO << "[kernel_completion] Serialization DISABLED, returning";
        return;
    }

    const auto& controller = *CHECK_NOTNULL(get_queue_controller());

    // Verify we have a currently executing queue (should be the completed one)
    CHECK(_dispatch_queue);

    ROCP_INFO << "[kernel_completion] Blocking queue " << completed.get_id().handle
              << " (setting block_signal to 1)";

    // STEP 1: Block the completed queue from executing more kernels
    controller.get_core_table().hsa_signal_store_screlease_fn(completed.block_signal, STOP_BARRIER);

    // STEP 2: Handle dispatch queue transition - use extracted function
    ROCP_INFO << "[kernel_completion] Looking for next ready queue to execute";
    if(!select_and_grant_ready_queue())
    {
        // No ready queue found or no queues waiting
        _dispatch_queue = nullptr;
        ROCP_INFO
            << "[kernel_completion] NO READY QUEUE AVAILABLE - _dispatch_queue set to nullptr";
    }
    else
    {
        ROCP_INFO << "[kernel_completion] Next queue selected: "
                  << _dispatch_queue->get_id().handle;
    }
}

/**
 * @brief Generate HSA barrier packets for kernel dispatch serialization
 *
 * This function is called when a kernel is about to be dispatched. It generates
 * the necessary HSA barrier packets that will be inserted into the queue to
 * control execution timing based on the serialization state.
 *
 * PACKET GENERATION LOGIC:
 * 1. If serialization is DISABLED: Return empty packet list (no serialization)
 * 2. If serialization is ENABLED: Generate barrier packets that depend on block_signal
 * 3. Handle dispatch queue assignment (immediate execution vs. queuing for later)
 *
 * The generated barrier packets will block kernel execution until the queue's
 * block_signal is set to RELEASE_BARRIER (0).
 *
 * @param queue The queue that is about to dispatch a kernel
 * @return Vector of HSA barrier packets to insert before the kernel
 */
common::container::small_vector<hsa::rocprofiler_packet, 3>
profiler_serializer::kernel_dispatch(const Queue& queue)
{
    common::container::small_vector<hsa::rocprofiler_packet, 3> ret;

    // Helper lambda to create properly configured HSA barrier packets
    auto&& CreateBarrierPacket = [](hsa_signal_t* dependency_signal,
                                    hsa_signal_t* completion_signal) {
        hsa::rocprofiler_packet barrier{};

        // Set packet type to barrier with AND operation
        barrier.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;

        // Configure memory fence scopes for system-wide synchronization
        barrier.barrier_and.header |= HSA_FENCE_SCOPE_SYSTEM
                                      << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        barrier.barrier_and.header |= HSA_FENCE_SCOPE_SYSTEM
                                      << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

        // Mark as barrier packet
        barrier.barrier_and.header |= 1 << HSA_PACKET_HEADER_BARRIER;

        // Set dependency signal (packet waits for this signal)
        if(dependency_signal != nullptr) barrier.barrier_and.dep_signal[0] = *dependency_signal;

        // Set completion signal (packet signals this when done)
        if(completion_signal != nullptr) barrier.barrier_and.completion_signal = *completion_signal;

        return barrier;
    };

    // Track the number of packets being enqueued for monitoring
    _enqueued_packets++;

    // If there are active barriers, add any barrier-specific packets
    if(!_barrier.empty())
    {
        // Check if the most recent barrier needs to add packets for this queue
        if(auto maybe_barrier = _barrier.back().barrier->enqueue_packet(&queue))
        {
            ret.push_back(*maybe_barrier);
        }
    }

    // Generate serialization packets based on current state
    switch(_serializer_status)
    {
        case Status::DISABLED:
            // Serialization disabled: return any barrier packets but no serialization control
            ROCP_INFO << "[kernel_dispatch] Queue " << queue.get_id().handle
                      << " - serialization DISABLED, returning " << ret.size()
                      << " barrier packets";
            return ret;

        case Status::ENABLED:
        {
            // SERIALIZATION ENABLED: Create barrier packets to control execution timing
            ROCP_INFO << "[kernel_dispatch] Queue " << queue.get_id().handle
                      << " - serialization ENABLED, _dispatch_queue="
                      << (_dispatch_queue ? _dispatch_queue->get_id().handle : 0)
                      << ", _enqueued_kernels.size=" << _enqueued_kernels.size()
                      << ", _ready_queues.size=" << _ready_queues.size();

            hsa_signal_t block_signal = queue.block_signal;
            hsa_signal_t ready_signal = queue.ready_signal;

            // Register ready signal handler if not already done
            register_queue_ready_handler(queue);

            ret.push_back(CreateBarrierPacket(nullptr, &ready_signal));

            // Create barrier packet that depends on the queue's block_signal
            // This packet will block until block_signal == 0 (RELEASE_BARRIER)
            ret.push_back(CreateBarrierPacket(&block_signal, &block_signal));

            // Add a second barrier packet for memory synchronization so that the
            // block_signal completion update is visible to CP
            ret.push_back(CreateBarrierPacket(nullptr, nullptr));

            ROCP_INFO << "[kernel_dispatch] Queue " << queue.get_id().handle
                      << " adding to _enqueued_kernels";
            _enqueued_kernels.push_back(&queue);
            break;
        }
    }
    ROCP_INFO << "[kernel_dispatch] Queue " << queue.get_id().handle << " - returning "
              << ret.size() << " total packets";
    return ret;
}

/**
 * @brief Handle notification that a queue has reached the blocking barrier
 *
 * This function is called when a queue's ready_signal is triggered, indicating
 * that the queue has reached the blocking barrier and is ready to execute once
 * the block_signal is released. This provides visibility into the serialization
 * state and confirms that releasing the block signal will immediately trigger
 * kernel execution.
 *
 * @param hsa_queue The HSA queue handle
 * @param queue The Queue object that has reached the ready state
 */
/**
 * @brief Check if queue has been seen before and set up signal handler if new
 *
 * This function checks if we've encountered this queue before by its ID.
 * If it's a new queue, it sets up an async signal handler for the ready_signal
 * that will be triggered when the signal value reaches -1.
 *
 * @param queue The Queue object to check and potentially register
 */
void
profiler_serializer::register_queue_ready_handler(const Queue& queue)
{
    uint64_t queue_id = queue.get_id().handle;

    // Check if we've already seen this queue
    if(_seen_queues.find(queue_id) == _seen_queues.end())
    {
        ROCP_INFO << "[register_ready_handler] Queue " << queue_id
                  << " - NEW QUEUE, registering ready_signal handler";

        // New queue - register the signal handler
        _seen_queues.insert(queue_id);

        const auto& controller = *CHECK_NOTNULL(get_queue_controller());

        // Set up async signal handler for ready_signal when it equals -1
        // The handler will be called when ready_signal transitions to -1
        hsa_status_t status = controller.get_ext_table().hsa_amd_signal_async_handler_fn(
            queue.ready_signal,
            HSA_SIGNAL_CONDITION_EQ,
            -1,  // Trigger value
            profiler_serializer_ready_signal_handler,
            const_cast<hsa_queue_t*>(queue.intercept_queue()));

        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_ERROR
                << "[register_ready_handler] FAILED to register ready signal handler for queue "
                << queue_id << ", status: " << status;
        }
        else
        {
            ROCP_INFO << "[register_ready_handler] Successfully registered ready signal handler "
                         "for queue "
                      << queue_id << " (signal handle: " << queue.ready_signal.handle << ")";
        }
    }
    else
    {
        ROCP_INFO << "[register_ready_handler] Queue " << queue_id
                  << " - already has ready_signal handler registered";
    }
}

void
profiler_serializer::queue_ready(hsa_queue_t* /* hsa_queue */, const Queue& queue)
{
    ROCP_INFO << "[queue_ready] Queue " << queue.get_id().handle
              << " SIGNALED READY - _dispatch_queue="
              << (_dispatch_queue ? _dispatch_queue->get_id().handle : 0)
              << ", _enqueued_kernels.size=" << _enqueued_kernels.size()
              << ", _ready_queues.size=" << _ready_queues.size();

    // Reset the ready_signal back to 0
    const auto& controller = *CHECK_NOTNULL(get_queue_controller());
    controller.get_core_table().hsa_signal_store_screlease_fn(queue.ready_signal, 0);

    ROCP_INFO << "[queue_ready] Queue " << queue.get_id().handle << " - ready_signal reset to 0";

    // If serialization is disabled, nothing to do
    if(_serializer_status == Status::DISABLED)
    {
        ROCP_INFO << "[queue_ready] Serialization DISABLED, returning";
        return;
    }

    // Always mark this queue as ready
    _ready_queues.insert(&queue);
    ROCP_INFO << "[queue_ready] Queue " << queue.get_id().handle
              << " added to _ready_queues (count now: " << _ready_queues.count(&queue)
              << ", total ready: " << _ready_queues.size() << ")";

    // Check if there is a dispatch currently executing
    if(_dispatch_queue == nullptr)
    {
        ROCP_INFO << "[queue_ready] NO QUEUE EXECUTING - checking for ready queue to grant";
        // No queue is currently executing - use extracted function to find and grant ready queue
        if(!select_and_grant_ready_queue())
        {
            ROCP_INFO
                << "[queue_ready] No queue in _enqueued_kernels is ready to execute"
                << " (this can happen if queue_ready fired before kernel_dispatch added to list)";
        }
        else
        {
            ROCP_INFO << "[queue_ready] Granted execution to queue "
                      << _dispatch_queue->get_id().handle;
        }
    }
    else
    {
        ROCP_INFO << "[queue_ready] Queue " << _dispatch_queue->get_id().handle
                  << " is currently executing - this queue will wait";
    }

    // Notify any waiters on the ready condition variable
    {
        std::lock_guard<std::mutex> lock(queue.cv_mutex);
        queue.cv_ready_signal.notify_all();
    }
}

void
profiler_serializer::destroy_queue(hsa_queue_t* id, const Queue& queue)
{
    ROCP_INFO << "destroying queue...";

    uint64_t     queue_id  = queue.get_id().handle;
    const Queue* queue_ptr = &queue;

    // Remove from barriers
    for(auto& barriers : _barrier)
    {
        barriers.barrier->remove_queue(&queue);
    }

    // Check if queue is currently executing
    if(_dispatch_queue && _dispatch_queue->get_id().handle == queue_id)
    {
        ROCP_FATAL << "Queue is being destroyed while kernel launch is still active";
    }

    // Check if queue is in _enqueued_kernels and report error
    auto dispatch_it = std::find_if(
        _enqueued_kernels.begin(), _enqueued_kernels.end(), [queue_id](const Queue* q) {
            return q->get_id().handle == queue_id;
        });
    if(dispatch_it != _enqueued_kernels.end())
    {
        ROCP_ERROR << "Queue " << queue_id
                   << " found in enqueued_kernels during destruction - removing";
        _enqueued_kernels.erase(dispatch_it);
    }

    // Check if queue is in _ready_queues and report error
    if(_ready_queues.count(queue_ptr) > 0)
    {
        size_t count = _ready_queues.count(queue_ptr);
        ROCP_ERROR << "Queue " << queue_id << " found " << count
                   << " times in ready_queues during destruction - removing all";
        auto ready_range = _ready_queues.equal_range(queue_ptr);
        _ready_queues.erase(ready_range.first, ready_range.second);
    }

    // Remove from _seen_queues
    _seen_queues.erase(queue_id);

    // Finalize queue destruction
    auto* controller = CHECK_NOTNULL(get_queue_controller());
    controller->set_queue_state(queue_state::to_destroy, id);
    controller->get_core_table().hsa_signal_store_screlease_fn(queue.block_signal, 0);

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

    // _enqueued_kernels
    oss << "_enqueued_kernels: {";
    oss << "count: " << _enqueued_kernels.size();
    if(!_enqueued_kernels.empty())
    {
        oss << ", queue_ids: [";
        bool first = true;
        for(const auto* queue : _enqueued_kernels)
        {
            if(!first) oss << ", ";
            oss << (queue ? queue->get_id().handle : 0);
            first = false;
        }
        oss << "]";
    }
    oss << "}, ";

    // _ready_queues
    oss << "_ready_queues: {";
    oss << "count: " << _ready_queues.size();
    if(!_ready_queues.empty())
    {
        oss << ", queue_ids: [";
        bool first = true;
        for(const auto* queue : _ready_queues)
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
