// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rocprofiler
{
namespace context
{
struct context;
}
namespace hsa
{
namespace queue_interposition
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
    uint32_t pkt_size  = 64;       ///< AQL packet size in bytes

    std::atomic<uint64_t> virtual_wptr{0};            ///< SDK-visible write index (virtualized)
    volatile uint64_t*    real_wdid       = nullptr;  ///< Pointer to actual queue write index
    volatile uint64_t*    real_rdid       = nullptr;  ///< Pointer to actual queue read index
    uint64_t              next_scan_pos   = 0;        ///< Next packet index to scan
    uint64_t              next_submit_pos = 0;        ///< Next packet index to submit

    const hsa_queue_t* hsa_queue       = nullptr;  ///< HSA queue pointer for Queue* lookup
    hsa_signal_t       doorbell_signal = {0};      ///< The queue's doorbell signal
    std::mutex         gate_lock       = {};       ///< Lock for packet submission gating

    /// Signal-less admission latch (§1.5.4). Plain bool, guarded by gate_lock: set
    /// by the destroy path's close_admission_and_snapshot_locked() so no later batch can
    /// register against this queue's window, read in signal_less_batch_eligible()
    /// which already holds gate_lock. Never an atomic -- one lock orders both.
    bool admission_closed = false;

    /// F1 lifetime interlock. drain_mu is held by whoever dereferences real_rdid for
    /// a bounded HW drain; destroy_queue() holds it across its whole sequence and
    /// clears rdid_valid+real_rdid under it BEFORE the runtime queue can be freed, so
    /// finalization's "skip if it disappeared" test (rdid_valid) is race-free. Leaf:
    /// acquired only at the TOP of the two drain sequences, never under gate_lock /
    /// doorbell map / hub / owner registry / the queue registry lock.
    std::mutex drain_mu   = {};
    bool       rdid_valid = false;  // guarded by drain_mu; set true at publication
};

using queue_state_ptr_t = std::shared_ptr<QueueState>;

/// Thread-safe map from HSA queue pointer to its QueueState
using queue_registry_t =
    common::Synchronized<std::unordered_map<const hsa_queue_t*, queue_state_ptr_t>>;

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
 * @brief Look up QueueState by HSA queue pointer
 *
 * @param queue The HSA queue to look up
 * @return Strong QueueState reference if found, empty otherwise
 */
queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue, bool create_if_missing = false);

/**
 * @brief Look up QueueState by doorbell signal
 *
 * @param signal The doorbell signal to look up
 * @return Strong QueueState reference if found and still alive, empty otherwise
 */
queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal, bool create_if_missing = false);

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
add_write_index_impl(QueueState*       state,
                     uint64_t          value,
                     std::memory_order order = std::memory_order_relaxed);

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
store_write_index_impl(QueueState*       state,
                       uint64_t          value,
                       std::memory_order order = std::memory_order_relaxed);

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
cas_write_index_impl(QueueState*       state,
                     uint64_t          expected,
                     uint64_t          value,
                     std::memory_order order = std::memory_order_relaxed);

/**
 * @brief Load virtual write pointer
 *
 * Returns the current value of the virtual write pointer.
 *
 * @param state Queue state
 * @return Current value of virtual_wptr
 */
uint64_t
load_write_index_impl(const QueueState* state, std::memory_order order = std::memory_order_relaxed);

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

/**
 * @brief Create and register queue state
 *
 * Allocates a QueueState for the given queue and inserts it into the queue
 * registry. Should be called when a queue is created by the application.
 *
 * @param queue The HSA queue to create state for
 * @param overwrite If true, replace any existing entry for queue; if false,
 *                  return the existing entry unchanged.
 */
std::shared_ptr<QueueState>
create_queue_state(const hsa_queue_t* queue, bool overwrite = false);

/**
 * @brief Destroy and unregister queue state
 *
 * Removes the queue's entry from the registry. Should be called when a queue
 * is destroyed by the application.
 *
 * @param queue The HSA queue to destroy state for
 */
void
destroy_queue_state(const hsa_queue_t* queue);

// The fence across EVERY live queue: teardown step 2 and the (a) fence of a
// hub-aware sync. The caller must hold no hub, registry or gate lock.
void
fence_all_queue_gates();

// F1 teardown HW drain across every live queue, race-free against a concurrent
// hsa_queue_destroy (the drain_mu/rdid_valid interlock). One shared absolute
// deadline. Caller holds no hub, registry, gate, or drain lock.
void
drain_all_queues_hw(uint64_t deadline_ns);

/// The hardware queue has consumed every packet we submitted. The inline path's
/// analogue of `_active_kernels == 0`, which only the legacy path maintains.
inline bool
hw_queue_drained(uint64_t real_rdid, uint64_t next_submit_pos)
{
    return real_rdid >= next_submit_pos;
}

// Close signal-less admission and snapshot the submit position in ONE gate_lock
// section (§1.5.4, Path-4 step 0): set admission_closed (SW-2, no later batch can
// register against this queue's window) AND read next_submit_pos, ordered by the
// same lock that serialises every publishing critical section. Returns that
// snapshot P. No separate fence and no atomic are needed.
//
// _locked: the caller holds state.drain_mu (F1). State-based, not pointer-based, so
// no relookup races the queue's destruction. The pointer-taking overloads are gone.
uint64_t
close_admission_and_snapshot_locked(QueueState& state);

/// Wait, bounded by an absolute kfd::steady_now_ns() deadline, until this queue's
/// hardware read index has caught up with the sampled submit position `P`, so
/// every packet in that snapshot -- hence every registered START -- has been
/// consumed by the CP before the caller decides anything was lost (§1.3 Path 4).
///
/// False on deadline; true immediately when real_rdid is null or P is already
/// drained. _locked: the caller holds state.drain_mu, so real_rdid cannot be freed
/// under the load (F1). State-based; the pointer-taking overload is gone.
bool
wait_queue_hw_drained_locked(QueueState& state, uint64_t submit_pos, uint64_t deadline_ns);

/**
 * @brief Check if queue interposition has been installed
 *
 * Returns true once interposition_init() has run, regardless of whether
 * interception is currently active. Used at queue construction time to choose
 * between the legacy WriteInterceptor path and the interposition path.
 *
 * @return True if interposition_init() has been called, false otherwise
 */
bool
supports_queue_interposition();

void
notify_queue_interposition_consumer_context_started(const context::context* ctx);

void
notify_queue_interposition_consumer_context_stopped(const context::context* ctx);

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
interposition_init(CoreApiTable* core_table, bool enabled);

/**
 * @brief Disable inline queue interception and clear tracked state
 *
 * This leaves the wrapped function pointers installed but removes all tracked
 * queue state so wrappers always pass through to the next function table.
 * Intended for finalization to avoid teardown-order hazards in static objects.
 */
void
interposition_fini();

/**
 * @brief Wait, bounded, for in-flight completions to retire
 *
 * Leaves the monitor running, so it is safe on paths where the SDK keeps operating
 * (context stop, code-object unload). Returns without waiting if the monitor is not
 * running or if called from a completion callback, which runs on the monitor thread and
 * so cannot wait for itself. The wait is bounded in every case and warns on expiry
 * rather than hanging.
 */
void
interposition_sync();

/**
 * @brief Stop and join the completion-monitor thread, retiring any stragglers
 *
 * Closes the producer admission gate, wakes the monitor, joins it, then waits (bounded)
 * for any producer still in the doorbell path to leave it and retires whatever they left
 * behind so pooled signals and correlation-id references are released. Must be called
 * before the signal pool is torn down so those releases target a live pool. A batch whose
 * completion signal never dropped is released without emitting a dispatch record: its
 * timestamps live in that signal and were never written.
 */
void
stop_completion_monitor();

/**
 * @brief Signal that global finalization has begun
 *
 * Moves the monitor to a finalizing state, distinct from the transient per-client
 * finalizer state. The monitor keeps running and keeps retiring completions; only the
 * grace period a drain is willing to wait shortens, because at global shutdown an
 * in-flight dependency may never resolve and stop_completion_monitor's forced retirement
 * is the authoritative flush. Must be called before the finalize-path drain, and only on
 * the true global-teardown path.
 */
void
request_completion_monitor_shutdown();
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
