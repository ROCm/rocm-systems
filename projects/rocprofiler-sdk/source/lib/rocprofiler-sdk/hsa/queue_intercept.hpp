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

#include <hsa/hsa.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief Per-queue state for SDK-level write pointer virtualization
 *
 * This structure maintains the state needed to intercept and virtualize
 * write-index updates and doorbell signals for an HSA queue. It enables
 * the SDK to scan and potentially modify packets before they are submitted
 * to the GPU.
 */
struct QueueState
{
    void*    ring_buf  = nullptr;  ///< Pointer to the queue's packet ring buffer
    uint32_t ring_size = 0;        ///< Number of packets the ring can hold
    uint32_t ring_mask = 0;        ///< Mask for ring index wrapping (ring_size - 1)

    std::atomic<uint64_t> virtual_wptr{0};            ///< SDK-visible write index (virtualized)
    volatile uint64_t*    real_wdid       = nullptr;  ///< Pointer to actual queue write index
    volatile uint64_t*    real_rdid       = nullptr;  ///< Pointer to actual queue read index
    uint64_t              next_scan_pos   = 0;        ///< Next packet index to scan
    uint64_t              next_submit_pos = 0;        ///< Next packet index to submit

    hsa_signal_t doorbell_signal = {0};      ///< The queue's doorbell signal
    uint64_t     k_factor        = 0;        ///< K-factor for metadata queue sync
    QueueState*  metadata_state  = nullptr;  ///< Pointer to metadata queue state if present
    std::mutex   gate_lock;                  ///< Lock for packet submission gating
};

/// Thread-safe map from HSA queue pointer to its QueueState
using queue_registry_t =
    common::Synchronized<std::unordered_map<const hsa_queue_t*, std::unique_ptr<QueueState>>>;

/// Thread-safe map from doorbell signal handle to QueueState pointer
using doorbell_map_t = common::Synchronized<std::unordered_map<uint64_t, QueueState*>>;

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
 * @return Pointer to QueueState if found, nullptr otherwise
 */
QueueState*
lookup_queue_state(const hsa_queue_t* queue);

/**
 * @brief Look up QueueState by doorbell signal
 *
 * @param signal The doorbell signal to look up
 * @return Pointer to QueueState if found, nullptr otherwise
 */
QueueState*
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
 */
void
unregister_doorbell(hsa_signal_t doorbell);

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
