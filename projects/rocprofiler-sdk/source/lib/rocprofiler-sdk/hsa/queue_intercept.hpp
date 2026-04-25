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
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
// Per-dispatch correlation snapshot captured at doorbell-store time
// in mode_tracing_only. Read by the firmware-ring drainer when the
// dispatch's END timestamp arrives.
//
// Slot lifecycle invariants are documented in
// PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md §3.5. Briefly:
//   - gen != 0           -> slot is occupied (not yet consumed/cleared)
//   - captured_wdid      -> identifies WHICH dispatch occupies this slot
//                           (drainer compares low 32 bits to the firmware
//                           record's dispatch_idx as alias guard)
//   - slot_publish_mu    -> serializes all field access (capture,
//                           consume, cleanup)
struct CorrEntry
{
    context::correlation_id*               corr_id    = nullptr;
    rocprofiler_thread_id_t                tid        = 0;
    uint64_t                               enqueue_ts = 0;
    tracing::external_correlation_id_map_t external_corr_ids;
    uint64_t                               seq        = 0;
    // Full 64-bit dispatch index (NOT modulo'd). Drainer compares
    // (captured_wdid & 0xFFFFFFFFu) == record.dispatch_idx.
    uint64_t                               captured_wdid = 0;
    // 0 == empty/consumed; non-zero == occupied. Release-store on publish.
    std::atomic<uint64_t>                  gen{0};
};

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
    uint32_t pkt_size  = 64;       ///< Packet size in bytes (64 for AQL, 256 for metadata)

    std::atomic<uint64_t> virtual_wptr{0};            ///< SDK-visible write index (virtualized)
    volatile uint64_t*    real_wdid       = nullptr;  ///< Pointer to actual queue write index
    volatile uint64_t*    real_rdid       = nullptr;  ///< Pointer to actual queue read index
    uint64_t              next_scan_pos   = 0;        ///< Next packet index to scan
    uint64_t              next_submit_pos = 0;        ///< Next packet index to submit

    const hsa_queue_t* hsa_queue       = nullptr;  ///< HSA queue pointer for Queue* lookup
    hsa_signal_t       doorbell_signal = {0};      ///< The queue's doorbell signal
    std::mutex         gate_lock;                  ///< Lock for packet submission gating

    // ---- mode_tracing_only fields (PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md §3.2) ----
    //
    // Two values are stored: legacy_passthrough is a process-wide install-
    // time decision, not a per-QueueState flag (no QueueState exists for
    // those queues).
    enum class Mode : uint8_t { tracing_only, full_intercept };
    Mode                       mode              = Mode::full_intercept;

    // Side-table for correlation capture in mode_tracing_only. Sized to
    // queue->size at create_queue_state time. Unused (size 0) in
    // mode_full_intercept.
    std::vector<CorrEntry>     corr_slots;
    uint32_t                   corr_ring_mask    = 0;
    std::atomic<uint64_t>      last_observed_wdid{0};

    // Serializes ALL access to corr_slots fields (capture, consume,
    // cleanup). See §3.5 Invariant 3.
    std::mutex                 slot_publish_mu;

    // One-shot guard for the throttled overwrite warning (§4 capture path).
    std::atomic<bool>          overwrite_warning_logged{false};
};

using queue_state_ptr_t      = std::shared_ptr<QueueState>;
using queue_state_weak_ptr_t = std::weak_ptr<QueueState>;

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
 */
bool
unregister_doorbell(hsa_signal_t doorbell);

/**
 * @brief Atomically add to virtual write pointer
 *
 * Increments the virtual write pointer by the given value and returns
 * the previous value. This is used to claim packet slots in the queue.
 *
 * @param state Queue state
 * @param value Amount to add
 * @return Previous value of virtual_wptr
 */
uint64_t
add_write_index_impl(QueueState* state, uint64_t value);

/**
 * @brief Store a new value to virtual write pointer
 *
 * Sets the virtual write pointer to the given value. This is typically
 * used for queue resets or initialization.
 *
 * @param state Queue state
 * @param value New value to store
 */
void
store_write_index_impl(QueueState* state, uint64_t value);

/**
 * @brief Compare-and-swap on virtual write pointer
 *
 * Atomically compares the virtual write pointer to expected and, if equal,
 * replaces it with value. Returns the previous value.
 *
 * @param state Queue state
 * @param expected Expected current value
 * @param value New value to store if comparison succeeds
 * @return Previous value of virtual_wptr
 */
uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value);

/**
 * @brief Load virtual write pointer
 *
 * Returns the current value of the virtual write pointer.
 *
 * @param state Queue state
 * @return Current value of virtual_wptr
 */
uint64_t
load_write_index_impl(const QueueState* state);

/// Type alias for doorbell function callback
using doorbell_fn_t = std::function<void(hsa_signal_t, hsa_signal_value_t)>;

/**
 * @brief Process doorbell ring for inline queue interposition
 *
 * This function scans the write-ahead zone (from next_scan_pos to virtual_wptr),
 * snapshots source packets in application-visible order, applies the queue
 * WriteInterceptor chain, advances the real write doorbell index, and calls the
 * provided doorbell function.
 *
 * The callback chain is invoked over the full batch of packets (1:1 forwarding).
 *
 * @param state Strong queue-state reference for call lifetime
 * @param value Signal value to pass to doorbell
 * @param ring_doorbell Callback to ring the actual hardware doorbell
 */
void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell);

// Capture-only doorbell processor for Mode::tracing_only.
// Captures correlation/tid/external-corr into state->corr_slots and
// returns; does NOT call invoke_write_interceptor and does NOT
// publish_submitted_packets. The caller chains through to the real
// hsa_signal_store_* after this returns.
//
// See PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md §4 for full rationale.
void
process_doorbell_tracing_only(const queue_state_ptr_t& state, hsa_signal_value_t value);

/**
 * @brief Create and register queue state
 *
 * Allocates a QueueState for the given queue and registers it in both the
 * queue registry and doorbell map. This should be called when a queue is
 * created by the application.
 *
 * @param queue The HSA queue to create state for (real, not intercept queue,
 *              in tracing-only mode)
 * @param wdid_addr Pointer to the queue's real write doorbell index (into the
 *                  queue's amd_queue_t mqd fields)
 * @param rdid_addr Pointer to the queue's real read doorbell index (into the
 *                  queue's amd_queue_t mqd fields)
 * @param mode Which Mode this QueueState should operate in. Tracing-only
 *             allocates corr_slots sized to queue->size; full_intercept
 *             leaves them empty.
 */
void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   QueueState::Mode   mode);

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

/**
 * @brief Install interposition wrappers into the HSA core API table
 *
 * Saves original function pointers and replaces them with wrappers that
 * route through the SDK's write-pointer virtualization when the queue is
 * tracked, or fall through to the original HSA implementation otherwise.
 *
 * @param core_table The HSA core API table to intercept
 */
void
install_intercept(CoreApiTable& core_table);

bool
is_intercepting_inline();

/**
 * @brief Disable inline queue interception and clear tracked state
 *
 * This leaves the wrapped function pointers installed but removes all tracked
 * queue state so wrappers always pass through to the next function table.
 * Intended for finalization to avoid teardown-order hazards in static objects.
 */
void
shutdown_intercept();

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
