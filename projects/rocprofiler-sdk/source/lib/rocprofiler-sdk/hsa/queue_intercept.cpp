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

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/packet_transformer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/ring_buffer.hpp"

#include <cassert>
#include <cstring>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
uint64_t
add_write_index_impl(QueueState* state, uint64_t value, std::memory_order mo)
{
    // stride baked into delta; claim_pos is authoritative
    const uint64_t prev = state->claim_pos.fetch_add(value * state->stride, mo);
    {
        std::lock_guard<std::mutex> lk{state->pending_lock};
        state->pending.emplace(prev,
                               PendingClaim{prev, static_cast<uint32_t>(value), state->stride});
    }
    return prev;
}

void
store_write_index_impl(QueueState* state, uint64_t value, std::memory_order mo)
{
    // HSA only exposes _relaxed and _screlease variants for store_write_index.
    // Any other order would be invalid for std::atomic::store.
    assert(mo == std::memory_order_relaxed || mo == std::memory_order_release);
    // The caller is resynchronizing the wptr directly, which invalidates any
    // in-flight bookkeeping we had for previously-claimed slots. Drop all
    // pending claims. `published_pos` is intentionally NOT touched here —
    // it is guarded by `gate_lock` which we do not hold. The next
    // `process_doorbell_impl` call will observe an empty pending list and
    // forward the doorbell verbatim; `published_pos` catches up naturally on
    // the next real claim.
    {
        std::lock_guard<std::mutex> lk{state->pending_lock};
        state->pending.clear();
    }
    // no rescaling; value is already in claim_pos units
    state->claim_pos.store(value, mo);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value, std::memory_order mo)
{
    // Failure order must be no stronger than success and must not be release/acq_rel.
    auto fail_mo = (mo == std::memory_order_release)   ? std::memory_order_relaxed
                   : (mo == std::memory_order_acq_rel) ? std::memory_order_acquire
                                                       : mo;
    // no rescaling; value is already in claim_pos units
    uint64_t   prev = expected;
    const bool ok   = state->claim_pos.compare_exchange_strong(prev, value, mo, fail_mo);
    if(ok)
    {
        // Successful CAS means the caller resynchronized the wptr.
        // Drop pending bookkeeping (see store_write_index_impl for the same
        // reasoning about published_pos).
        std::lock_guard<std::mutex> lk{state->pending_lock};
        state->pending.clear();
    }
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state, std::memory_order mo)
{
    // HSA only exposes _relaxed and _scacquire variants for load_write_index.
    assert(mo == std::memory_order_relaxed || mo == std::memory_order_acquire);
    return state->claim_pos.load(mo);
}

namespace
{
/// A claim is ready to publish only when every slot it owns has been
/// written by the application (i.e. no slot is still INVALID). Acquire-load
/// the 16-bit header to pair with the application's release-store of its
/// packet header (the standard AQL ordering pattern).
///
/// Within a single claim, packets are always laid out contiguously starting
/// at `claim_index`: a caller that passes `packet_count=N` to
/// `add_write_index` writes slots `claim_index + 0..N-1`. The `stride` field
/// only affects the claim's reservation_end (where the next claim begins),
/// not the intra-claim slot layout. This is what structurally fixes bug #2.
bool
claim_is_ready(const RingView& view, const PendingClaim& c)
{
    for(uint32_t i = 0; i < c.packet_count; ++i)
    {
        const auto* slot = view.read_slot(c.claim_index + i);
        const auto hdr = __atomic_load_n(reinterpret_cast<const uint16_t*>(slot), __ATOMIC_ACQUIRE);
        const uint16_t pt =
            (hdr >> HSA_PACKET_HEADER_TYPE) & ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
        if(pt == HSA_PACKET_TYPE_INVALID) return false;
    }
    return true;
}
}  // namespace

void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell)
{
    if(!state) return;
    auto* state_ptr = state.get();

    std::unique_lock<std::mutex> lock{state_ptr->gate_lock};

    const RingView& view     = state_ptr->ring_view;
    const uint32_t  pkt_size = view.pkt_size;

    // Bug #9 — HSA_QUEUE_TYPE_SINGLE queues require monotonically
    // non-decreasing doorbell values. On a multi-claim race an early-exit
    // caller can otherwise forward its own (smaller) claim index on top of a
    // later caller's higher value. Clamp against last_doorbell_val.
    auto clamp_for_single = [state_ptr](hsa_signal_value_t v) -> hsa_signal_value_t {
        if(!state_ptr->is_single) return v;
        auto prev = state_ptr->last_doorbell_val.load(std::memory_order_relaxed);
        return (static_cast<int64_t>(v) < static_cast<int64_t>(prev))
                   ? static_cast<hsa_signal_value_t>(prev)
                   : v;
    };

    // Shared commit/publish lambda — one place that advances real_wdid and rings.
    auto publish_and_ring = [&](uint64_t final_pos, hsa_signal_value_t db_val) {
        auto real_rdid = __atomic_load_n(state_ptr->real_rdid, __ATOMIC_ACQUIRE);
        auto ring_used = final_pos - real_rdid;
        if(ring_used > view.size)
        {
            ROCP_WARNING << "Queue-intercept observed ring usage beyond ring size. queue="
                         << state_ptr->hsa_queue << ", ring_used=" << ring_used
                         << ", ring_size=" << view.size
                         << ", published_pos=" << state_ptr->published_pos;
        }
        __atomic_store_n(state_ptr->real_wdid, final_pos, __ATOMIC_RELEASE);
        const auto clamped = clamp_for_single(db_val);
        state_ptr->last_doorbell_val.store(static_cast<uint64_t>(clamped),
                                           std::memory_order_relaxed);
        ring_doorbell(state_ptr->doorbell_signal, clamped);
    };

    // Resolve queue once.
    auto*        qc = get_queue_controller();
    const Queue* queue =
        (qc && state_ptr->hsa_queue) ? qc->get_queue(*state_ptr->hsa_queue) : nullptr;

    RingCursor cursor{view, state_ptr->published_pos, state_ptr->real_rdid};

    // Drain ready claims in claim_index order.
    while(true)
    {
        PendingClaim head;
        {
            std::lock_guard<std::mutex> pend{state_ptr->pending_lock};
            if(state_ptr->pending.empty()) break;
            head = state_ptr->pending.begin()->second;
        }

        // Bug #1 fix — if any slot the claim owns is still INVALID, the
        // application hasn't finished writing. Stop here; a later doorbell
        // will catch the same claim again once its headers land.
        if(!claim_is_ready(view, head)) break;

        const uint64_t reservation_end =
            head.claim_index + static_cast<uint64_t>(head.packet_count) * head.stride;
        cursor.set_reservation_end(reservation_end);

        if(queue)
        {
            // Snapshot source packets into a side buffer so the cursor can
            // safely overwrite the same ring slots. Packets from one claim
            // are always contiguous in `claim_index`-space (the producer
            // wrote them back-to-back); when stride > 1 the tail of the
            // reservation is reservation-only and gets gap-padded below.
            std::vector<char> snapshot(static_cast<size_t>(head.packet_count) * pkt_size);
            for(uint32_t i = 0; i < head.packet_count; ++i)
            {
                memcpy(
                    snapshot.data() + i * pkt_size, view.read_slot(head.claim_index + i), pkt_size);
            }

            ScopedWriter guard{[&cursor, pkt_size](const void* pkts, uint64_t n) {
                const auto* src = static_cast<const char*>(pkts);
                for(uint64_t i = 0; i < n; ++i)
                {
                    if(!cursor.write(src + i * pkt_size)) return;
                }
            }};

            queue->invoke_write_interceptor(
                snapshot.data(), head.packet_count, packet_writer_trampoline);
        }
        else
        {
            // No interceptor — pass packets through in the order claimed.
            for(uint32_t i = 0; i < head.packet_count; ++i)
            {
                if(!cursor.write(view.read_slot(head.claim_index + i))) break;
            }
        }
        cursor.pad_to_reservation();

        // Pop this claim now that it's processed.
        {
            std::lock_guard<std::mutex> pend{state_ptr->pending_lock};
            if(!state_ptr->pending.empty()) state_ptr->pending.erase(state_ptr->pending.begin());
        }
    }

    if(cursor.submit_pos() == state_ptr->published_pos)
    {
        // Nothing processed. Forward user's value (clamped for SINGLE queues).
        const auto clamped = clamp_for_single(value);
        state_ptr->last_doorbell_val.store(static_cast<uint64_t>(clamped),
                                           std::memory_order_relaxed);
        ring_doorbell(state_ptr->doorbell_signal, clamped);
        return;
    }

    state_ptr->published_pos = cursor.submit_pos();
    publish_and_ring(state_ptr->published_pos,
                     static_cast<hsa_signal_value_t>(state_ptr->published_pos - 1));
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
