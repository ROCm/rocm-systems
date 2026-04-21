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
#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include <hsa/hsa.h>

#include <cstdint>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief Layer 5 of the queue_intercept refactor: per-queue state lifecycle
 *        and lookup.
 *
 * Owns the two global maps (queue pointer -> QueueState, doorbell handle ->
 * weak QueueState) and exposes the create/destroy/lookup entry points used by
 * the inline-intercept wrappers and by the queue_controller's
 * create_queue/destroy_queue callbacks.
 *
 * The QueueState type itself is shared with other layers and remains in
 * queue_intercept.hpp for now.
 */

/// Thread-safe map from HSA queue pointer to its QueueState
using queue_registry_t =
    common::Synchronized<std::unordered_map<const hsa_queue_t*, queue_state_ptr_t>>;

/// Thread-safe map from doorbell signal handle to weak QueueState reference
using doorbell_map_t = common::Synchronized<std::unordered_map<uint64_t, queue_state_weak_ptr_t>>;

/**
 * @brief Get the global queue registry singleton
 *
 * The registry maps HSA queue pointers to their corresponding QueueState.
 * This is the primary lookup mechanism for queue state.
 *
 * @return Reference to the queue registry
 */
queue_registry_t&
get_queue_registry();

/**
 * @brief Get the global doorbell map singleton
 *
 * The doorbell map allows looking up QueueState by doorbell signal handle,
 * which is needed when intercepting signal store operations.
 *
 * @return Reference to the doorbell map
 */
doorbell_map_t&
get_doorbell_map();

/**
 * @brief Look up QueueState by HSA queue pointer
 *
 * @param queue The HSA queue to look up
 * @return Strong QueueState reference if found, empty otherwise
 */
queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue);

/**
 * @brief Look up QueueState by doorbell signal
 *
 * @param signal The doorbell signal to look up
 * @return Strong QueueState reference if found and still alive, empty otherwise
 */
queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal);

/**
 * @brief Register a doorbell signal for a queue
 *
 * This creates an association between the queue's doorbell signal and its
 * QueueState, enabling lookup by signal handle in intercepted signal stores.
 *
 * @param queue The HSA queue whose doorbell to register
 * @param doorbell The doorbell signal to register
 */
void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell);

/**
 * @brief Unregister a doorbell signal
 *
 * Removes the doorbell-to-QueueState mapping when a queue is destroyed.
 *
 * @param doorbell The doorbell signal to unregister
 * @return true if an entry was erased, false if nothing was registered
 */
bool
unregister_doorbell(hsa_signal_t doorbell);

/**
 * @brief Create and register queue state
 *
 * Allocates a QueueState for the given queue and registers it in both the
 * queue registry and doorbell map. This should be called when a queue is
 * created by the application.
 *
 * @param queue The HSA queue to create state for
 * @param wdid_addr Pointer to the queue's real write doorbell index
 * @param rdid_addr Pointer to the queue's real read doorbell index
 * @param k_factor K-factor for metadata queue synchronization
 */
void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   uint64_t           k_factor);

/**
 * @brief Destroy and unregister queue state
 *
 * Removes the queue's state from the registry and doorbell map.
 * This should be called when a queue is destroyed by the application.
 *
 * @param queue The HSA queue to destroy state for
 */
void
destroy_queue_state(const hsa_queue_t* queue);

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
