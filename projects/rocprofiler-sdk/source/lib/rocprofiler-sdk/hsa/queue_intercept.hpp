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

#include "lib/rocprofiler-sdk/hsa/ring_buffer.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief Explicit per-claim record (Step 12' — Option A).
 *
 * Each `add_write_index` call records one `PendingClaim`. The consumer
 * (`process_doorbell_impl`) drains these in `claim_index` order and only
 * processes a claim once every slot it owns holds a non-INVALID header.
 * This replaces the Step 9' ring-scan + layout-heuristic approach and
 * structurally fixes bug #1 (INVALID-header race) and bug #2 (strided vs
 * contiguous layout ambiguity).
 */
struct PendingClaim
{
    uint64_t claim_index  = 0;  ///< What add_write_index returned — the start ring slot
    uint32_t packet_count = 0;  ///< Value the caller passed to add_write_index
    uint32_t stride       = 1;  ///< Stride at claim time (1 + k_factor when claimed)
};

/**
 * @brief Per-queue state for SDK-level write pointer virtualization
 *
 * This structure maintains the state needed to intercept and virtualize
 * write-index updates and doorbell signals for an HSA queue. It enables
 * the SDK to scan and potentially modify packets before they are submitted
 * to the GPU.
 *
 * Cursor model (Shape B):
 *   - `claim_pos` is the SDK's authoritative atomic cursor that advances
 *     whenever the application claims packet slots (via add_write_index).
 *     It is measured in ring-slot (stride-scaled) units.
 *   - `published_pos` is the cursor advanced inside `process_doorbell_impl`
 *     as packets are scanned from the write-ahead zone and published to
 *     the ring. It is guarded by `gate_lock`. After each doorbell it must
 *     equal `claim_pos` (modulo partial/strided writes).
 *   - `*real_wdid` is the HSA-side hardware shadow; the SDK writes through
 *     to it after rewriting packets.
 *
 * Explicit pending-claim tracking (Step 12'):
 *   - Every `add_write_index` inserts a `PendingClaim` keyed by the returned
 *     `claim_index`. `process_doorbell_impl` pops ready claims in order.
 *   - `store_write_index`/`cas_write_index` (on success) clear `pending`
 *     because the caller has re-synchronized the wptr directly.
 */
struct QueueState
{
    RingView ring_view = {};  ///< Ring geometry (set at creation, never changes)
    uint32_t stride    = 1;   ///< Per-packet ring advance = 1 + k_factor

    std::atomic<uint64_t>    claim_pos{0};         ///< Authoritative SDK cursor (claim side)
    volatile uint64_t*       real_wdid = nullptr;  ///< Pointer to actual queue write index
    volatile const uint64_t* real_rdid = nullptr;  ///< Pointer to actual queue read index

    const hsa_queue_t* hsa_queue       = nullptr;  ///< HSA queue pointer for Queue* lookup
    hsa_signal_t       doorbell_signal = {0};      ///< The queue's doorbell signal

    std::mutex gate_lock;          ///< Lock for packet submission gating
    uint64_t   published_pos = 0;  ///< Cursor up to which packets have been published

    /// HSA_QUEUE_TYPE_SINGLE flag. When true, forwarded doorbell values must be
    /// monotonically non-decreasing (spec requirement). See bug #9.
    bool is_single = false;

    /// Highest doorbell value we have rung on this queue so far. Used only
    /// when `is_single` to clamp a smaller forwarded value up to the last
    /// published high-water mark. Stored relaxed because gate_lock serializes
    /// the only writer (process_doorbell_impl).
    std::atomic<uint64_t> last_doorbell_val{0};

    /// Ordered map of in-flight claims, keyed by `claim_index`. Producers
    /// race on `claim_pos.fetch_add`, so insertion order into this map does
    /// not match claim-index order; std::map gives us O(log N) sorted-by-key
    /// insert/iterate. N is small (bounded by concurrent in-flight claims).
    std::mutex                       pending_lock;
    std::map<uint64_t, PendingClaim> pending;  ///< key = claim_index
};

using queue_state_ptr_t      = std::shared_ptr<QueueState>;
using queue_state_weak_ptr_t = std::weak_ptr<QueueState>;

/**
 * @brief Atomically add to claim_pos
 *
 * Advances claim_pos by (value * stride) so callers that pass
 * packet-count units advance the cursor in ring-slot units.
 *
 * @param state Queue state
 * @param value Amount to add (in packet-count units)
 * @param mo    Memory order corresponding to the HSA variant called
 *              (relaxed/acquire/release/acq_rel)
 * @return Previous value of claim_pos
 */
uint64_t
add_write_index_impl(QueueState* state, uint64_t value, std::memory_order mo);

/**
 * @brief Store a new value to claim_pos
 *
 * Sets claim_pos to the given value verbatim. No stride scaling is
 * performed because the caller's argument is already in claim_pos units
 * (e.g., a value previously loaded/CASed via this same interface).
 *
 * HSA's store API only exposes `_relaxed` and `_screlease`; `mo` must be
 * either `std::memory_order_relaxed` or `std::memory_order_release`.
 *
 * @param state Queue state
 * @param value New value to store
 * @param mo    Memory order (relaxed or release)
 */
void
store_write_index_impl(QueueState* state, uint64_t value, std::memory_order mo);

/**
 * @brief Compare-and-swap on claim_pos
 *
 * Atomically compares claim_pos to expected and, if equal, replaces it
 * with value. Both expected and value are in claim_pos units; no stride
 * scaling is applied.
 *
 * The success memory order is `mo`; the failure order is derived to be no
 * stronger than success and to exclude release/acq_rel.
 *
 * @param state Queue state
 * @param expected Expected current value
 * @param value New value to store if comparison succeeds
 * @param mo    Success memory order corresponding to the HSA variant called
 * @return Previous value of claim_pos
 */
uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value, std::memory_order mo);

/**
 * @brief Load claim_pos
 *
 * HSA's load API only exposes `_relaxed` and `_scacquire`; `mo` is expected
 * to be either `std::memory_order_relaxed` or `std::memory_order_acquire`.
 *
 * @param state Queue state
 * @param mo    Memory order (relaxed or acquire)
 * @return Current value of claim_pos
 */
uint64_t
load_write_index_impl(const QueueState* state, std::memory_order mo);

/// Type alias for doorbell function callback
using doorbell_fn_t = std::function<void(hsa_signal_t, hsa_signal_value_t)>;

/**
 * @brief Process doorbell ring for inline queue interposition
 *
 * This function scans the write-ahead zone (from published_pos to claim_pos),
 * snapshots source packets in application-visible order, applies the queue
 * WriteInterceptor chain, advances the real write doorbell index, and calls the
 * provided doorbell function.
 *
 * For stride==1 (k_factor=0), the callback chain is invoked over the full batch.
 * For stride>1, packets are transformed one-by-one and padded to stride as needed.
 *
 * @param state Strong queue-state reference for call lifetime
 * @param value Signal value to pass to doorbell
 * @param ring_doorbell Callback to ring the actual hardware doorbell
 */
void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell);

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
