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
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <hsa/hsa.h>

#include <cstring>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

doorbell_map_t&
get_doorbell_map()
{
    static auto*& _v = common::static_object<doorbell_map_t>::construct();
    return *_v;
}

queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue)
{
    queue_state_ptr_t result = {};
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second;
        }
    });
    return result;
}

queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    queue_state_ptr_t result = {};
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second.lock();
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    auto state = lookup_queue_state(queue);
    if(!state) return;

    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
}

bool
unregister_doorbell(hsa_signal_t doorbell)
{
    bool erased = false;
    get_doorbell_map().wlock(
        [&](auto& doorbell_map) { erased = (doorbell_map.erase(doorbell.handle) > 0); });
    return erased;
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value)
{
    return state->virtual_wptr.fetch_add(value, std::memory_order_relaxed);
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    state->virtual_wptr.store(value, std::memory_order_relaxed);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    return state->virtual_wptr.load(std::memory_order_relaxed);
}

namespace
{
thread_local QueueState*          tls_state                     = nullptr;
thread_local uint64_t             tls_submit_pos                = 0;
thread_local uint32_t             tls_pkt_size                  = 64;
thread_local const doorbell_fn_t* tls_ring_doorbell             = nullptr;
thread_local uint64_t             tls_last_published_submit_pos = 0;

inline void
log_overwrite_warning_once(QueueState* state)
{
    bool already = state->overwrite_warning_logged.exchange(true, std::memory_order_relaxed);
    if(!already)
    {
        ROCP_WARNING << "queue_intercept: tracing-only slot wraparound on queue="
                     << state->hsa_queue
                     << " — drainer cadence is slower than enqueue rate; "
                        "one or more dispatches will lose correlation. This warning "
                        "is logged once per queue.";
    }
}

inline void
publish_submitted_packets(QueueState* state, uint64_t submit_pos)
{
    if(!tls_ring_doorbell || submit_pos <= tls_last_published_submit_pos || submit_pos == 0) return;

    __atomic_store_n(state->real_wdid, submit_pos, __ATOMIC_RELEASE);
    (*tls_ring_doorbell)(state->doorbell_signal, static_cast<hsa_signal_value_t>(submit_pos - 1));
    tls_last_published_submit_pos = submit_pos;
}

inline void
wait_for_free_slot(QueueState* state, uint64_t submit_pos)
{
    while(true)
    {
        auto real_rdid = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
        auto ring_used = submit_pos - real_rdid;
        if(ring_used < state->ring_size)
        {
            return;
        }

        // If the producer is blocked on a full ring and has already written
        // packets beyond the last visible write index, publish progress so the
        // consumer can observe and drain them.
        publish_submitted_packets(state, submit_pos);
        std::this_thread::yield();
    }
}

void
ring_buffer_writer(const void* pkts, uint64_t pkt_count)
{
    auto*       state    = tls_state;
    auto        pkt_size = tls_pkt_size;
    const auto* src      = static_cast<const char*>(pkts);
    for(uint64_t i = 0; i < pkt_count; i++)
    {
        wait_for_free_slot(state, tls_submit_pos);
        auto        slot = tls_submit_pos & state->ring_mask;
        auto*       dst  = static_cast<char*>(state->ring_buf) + (slot * pkt_size);
        const auto* s    = src + i * pkt_size;
        if(dst != s) memcpy(dst, s, pkt_size);
        tls_submit_pos++;
    }
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

    const uint64_t scan_pos = state_ptr->next_scan_pos;
    const uint64_t scan_end = state_ptr->virtual_wptr.load(std::memory_order_acquire);

    if(scan_pos >= scan_end)
    {
        ring_doorbell(state_ptr->doorbell_signal, value);
        return;
    }

    const uint64_t pkt_count = scan_end - scan_pos;

    std::vector<char> source_snapshot(pkt_count * state_ptr->pkt_size);
    for(uint64_t i = 0; i < pkt_count; ++i)
    {
        const auto* src = static_cast<const char*>(state_ptr->ring_buf) +
                          (((scan_pos + i) & state_ptr->ring_mask) * state_ptr->pkt_size);
        memcpy(source_snapshot.data() + (i * state_ptr->pkt_size), src, state_ptr->pkt_size);
    }

    tls_state                     = state_ptr;
    tls_submit_pos                = state_ptr->next_submit_pos;
    tls_pkt_size                  = state_ptr->pkt_size;
    tls_ring_doorbell             = &ring_doorbell;
    tls_last_published_submit_pos = state_ptr->next_submit_pos;
    uint64_t start_submit_pos     = tls_submit_pos;

    auto*        qc = get_queue_controller();
    const Queue* queue =
        (qc && state_ptr->hsa_queue) ? qc->get_queue(*state_ptr->hsa_queue) : nullptr;

    if(queue)
    {
        queue->invoke_write_interceptor(source_snapshot.data(), pkt_count, ring_buffer_writer);
    }
    else
    {
        ring_buffer_writer(source_snapshot.data(), pkt_count);
    }

    uint64_t written = tls_submit_pos - start_submit_pos;
    if(written != pkt_count)
    {
        ROCP_WARNING << "Write-interceptor changed packet count. "
                     << "queue=" << state_ptr->hsa_queue << ", input_pkt_count=" << pkt_count
                     << ", written_pkt_count=" << written;
    }

    state_ptr->next_scan_pos   = scan_end;
    state_ptr->next_submit_pos = tls_submit_pos;

    auto real_rdid = __atomic_load_n(state_ptr->real_rdid, __ATOMIC_ACQUIRE);
    auto ring_used = (state_ptr->next_submit_pos - real_rdid);
    if(ring_used > state_ptr->ring_size)
    {
        ROCP_WARNING << "Queue-intercept observed ring usage beyond ring size. queue="
                     << state_ptr->hsa_queue << ", ring_used=" << ring_used
                     << ", ring_size=" << state_ptr->ring_size << ", scan_pos=" << scan_pos
                     << ", scan_end=" << scan_end
                     << ", next_submit_pos=" << state_ptr->next_submit_pos;
    }

    publish_submitted_packets(state_ptr, state_ptr->next_submit_pos);

    tls_ring_doorbell             = nullptr;
    tls_last_published_submit_pos = 0;
    tls_state                     = nullptr;
}

void
process_doorbell_tracing_only(const queue_state_ptr_t& state, hsa_signal_value_t value)
{
    if(!state) return;
    auto* state_ptr = state.get();

    // §3.5 Invariant 3: take the slot mutex; serializes capture vs
    // consume vs cleanup.
    std::lock_guard<std::mutex> g(state_ptr->slot_publish_mu);

    const uint64_t prev = state_ptr->last_observed_wdid.load(std::memory_order_relaxed);
    // HIP convention: hsa_signal_store_screlease(doorbell, new_wdid - 1).
    const uint64_t new_wdid = static_cast<uint64_t>(value) + 1;

    // Stale/sentinel doorbell — no advance.
    if(new_wdid <= prev) return;

    auto* corr_id      = context::get_latest_correlation_id();
    bool  popping_corr = false;
    if(!corr_id)
    {
        corr_id      = context::correlation_tracing_service::construct(1);
        popping_corr = true;
    }
    if(!corr_id)
    {
        // Finalization race — service is tearing down. Advance the
        // observed wdid so we don't reprocess this range; drainer will
        // see no entry and emit zero-corr fallback.
        state_ptr->last_observed_wdid.store(new_wdid, std::memory_order_release);
        return;
    }

    // Balance construct's initial +1 at scope exit; per-slot adds are
    // released to the drainer.
    auto _corr_id_dtor = common::scope_destructor{[&] {
        if(popping_corr)
        {
            context::pop_latest_correlation_id(corr_id);
            corr_id->sub_ref_count();
        }
    }};

    const auto tid =
        corr_id->thread_idx ? corr_id->thread_idx : common::get_tid();
    const auto enqueue_ts = common::timestamp_ns();

    // Snapshot external correlation IDs once for the whole batch.
    tracing::tracing_data td{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               td);
    tracing::populate_external_correlation_ids(
        td.external_correlation_ids,
        tid,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
        corr_id->internal);

    static std::atomic<uint64_t> s_seq{0};

    const auto* pkts =
        static_cast<const hsa_kernel_dispatch_packet_t*>(state_ptr->ring_buf);

    for(uint64_t d = prev; d < new_wdid; ++d)
    {
        const uint32_t slot_idx = static_cast<uint32_t>(d & state_ptr->corr_ring_mask);

        // Skip non-kernel-dispatch packets (barrier_and/or, agent_dispatch).
        // The firmware ring only emits records for kernel dispatches, so a
        // captured slot for a non-kernel-dispatch packet would never be
        // consumed and would slowly leak.
        const uint16_t hdr =
            __atomic_load_n(&pkts[slot_idx].header, __ATOMIC_ACQUIRE);
        const uint8_t ptype = (hdr >> HSA_PACKET_HEADER_TYPE) &
                              ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
        if(ptype != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

        auto& entry = state_ptr->corr_slots[slot_idx];

        // §3.5 Invariant 2: detect wraparound overwrite. If the slot is
        // still occupied by a previous capture that the drainer has not
        // consumed, retire its refcounts so we don't leak. The drainer
        // will receive the prior dispatch's END record later, fail the
        // alias guard, and emit it as a zero-corr fallback.
        if(entry.gen.load(std::memory_order_relaxed) != 0 && entry.corr_id)
        {
            entry.corr_id->sub_kern_count();
            entry.corr_id->sub_ref_count();
            log_overwrite_warning_once(state_ptr);
        }

        corr_id->add_ref_count();
        corr_id->add_kern_count();

        entry.corr_id           = corr_id;
        entry.tid               = tid;
        entry.enqueue_ts        = enqueue_ts;
        entry.external_corr_ids = td.external_correlation_ids;
        entry.seq               = s_seq.fetch_add(1, std::memory_order_relaxed);
        entry.captured_wdid     = d;  // §3.5 Invariant 1
        entry.gen.fetch_add(1, std::memory_order_release);  // publish
    }

    state_ptr->last_observed_wdid.store(new_wdid, std::memory_order_release);
    // _corr_id_dtor fires here, balancing the construct's initial +1 if
    // popping_corr.
}

void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   QueueState::Mode   mode)
{
    // Idempotency guard: if a QueueState already exists for this
    // queue (e.g., late-attach raced with normal queue creation),
    // return without overwriting.
    if(lookup_queue_state(queue)) return;

    auto     state         = std::make_shared<QueueState>();
    uint64_t current_wdid  = __atomic_load_n(wdid_addr, __ATOMIC_ACQUIRE);
    state->ring_buf        = queue->base_address;
    state->ring_size       = queue->size;
    state->ring_mask       = queue->size - 1;
    state->real_wdid       = wdid_addr;
    state->real_rdid       = rdid_addr;
    state->hsa_queue       = queue;
    state->doorbell_signal = queue->doorbell_signal;
    state->virtual_wptr.store(current_wdid, std::memory_order_relaxed);
    state->next_scan_pos   = current_wdid;
    state->next_submit_pos = current_wdid;

    // Phase 1 fields.
    state->mode               = mode;
    state->corr_ring_mask     = queue->size - 1;
    state->last_observed_wdid.store(current_wdid, std::memory_order_relaxed);
    if(mode == QueueState::Mode::tracing_only)
    {
        // CorrEntry has std::atomic<uint64_t> gen, which is neither copy- nor
        // move-constructible. vector::resize() instantiates the relocation
        // path and fails the static_assert on libstdc++. The vector(N) ctor
        // allocates and default-initializes in place — no relocation — which
        // works for non-movable element types.
        state->corr_slots = std::vector<CorrEntry>(queue->size);
    }

    get_queue_registry().wlock([&](auto& map) { map[queue] = state; });
    get_doorbell_map().wlock([&](auto& map) { map[queue->doorbell_signal.handle] = state; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    // §3.5 Invariants 3 (single mutex) and 4 (exactly-once retirement):
    // Drain any captured-but-not-consumed corr_slots BEFORE erasing the
    // QueueState from the registry. Doing this after erase would race
    // with lookup_queue_state returning null and leak correlation_id
    // refcounts. The slot_publish_mu serializes against in-flight
    // capture (launch path) and consume (drainer).
    if(auto state = lookup_queue_state(queue))
    {
        std::lock_guard<std::mutex> g(state->slot_publish_mu);
        for(auto& entry : state->corr_slots)
        {
            if(entry.gen.load(std::memory_order_relaxed) != 0 && entry.corr_id)
            {
                entry.corr_id->sub_kern_count();
                entry.corr_id->sub_ref_count();
                entry.corr_id = nullptr;
                entry.gen.store(0, std::memory_order_release);
            }
        }
    }

    hsa_signal_t      doorbell = {0};
    queue_state_ptr_t doomed   = {};
    get_queue_registry().wlock([&](auto& map) {
        auto it = map.find(queue);
        if(it == map.end())
        {
            return;
        }
        doomed = it->second;
        if(doomed) doorbell = doomed->doorbell_signal;
        map.erase(it);
    });

    if(!doomed) return;
    if(doorbell.handle != 0) unregister_doorbell(doorbell);
}

namespace
{
std::atomic<bool> s_intercept_installed = false;

// Saved next-in-chain function pointers (tracing functors or raw HSA, depending on
// when install_intercept is called). Our wrappers chain through these for untracked
// queues and for the final doorbell ring on tracked queues.
CoreApiTable s_next_table = {};

bool
should_bypass_inline_intercept()
{
    return (!s_intercept_installed.load(std::memory_order_acquire) ||
            registration::get_fini_status() > 0);
}

// --- add_write_index wrappers (4) ---

uint64_t
wrap_add_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);
}

uint64_t
wrap_add_write_index_scacq_screl(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);
}

uint64_t
wrap_add_write_index_scacquire(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);
}

uint64_t
wrap_add_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);

    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);
}

// --- store_write_index wrappers (2) ---

void
wrap_store_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
        return;
    }

    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
}

void
wrap_store_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
        return;
    }

    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
}

// --- cas_write_index wrappers (4) ---

uint64_t
wrap_cas_write_index_relaxed(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacq_screl(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacquire(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_screlease(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);

    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);
}

// --- load_write_index wrappers (2) ---

uint64_t
wrap_load_write_index_relaxed(const hsa_queue_t* q)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);

    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get());
    return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);
}

uint64_t
wrap_load_write_index_scacquire(const hsa_queue_t* q)
{
    if(should_bypass_inline_intercept())
        return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);

    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get());
    return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);
}

// --- signal_store wrappers (2) ---

void
wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_store_relaxed_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(!s)
    {
        // Not one of our queues.
        s_next_table.hsa_signal_store_relaxed_fn(sig, val);
        return;
    }

    if(s->mode == QueueState::Mode::tracing_only)
    {
        process_doorbell_tracing_only(s, val);
        // ALWAYS chain through — the application's queue is the real HSA
        // queue; the GPU needs to see the doorbell. Unlike
        // process_doorbell_impl, the tracing_only path does not chain
        // internally.
        s_next_table.hsa_signal_store_relaxed_fn(sig, val);
        return;
    }

    // Mode::full_intercept — existing PR 5219 path. process_doorbell_impl
    // chains through via the lambda (directly when no new packets, or via
    // publish_submitted_packets/tls_ring_doorbell otherwise).
    process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
        s_next_table.hsa_signal_store_relaxed_fn(db, v);
    });
}

void
wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    if(should_bypass_inline_intercept())
    {
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    auto s = lookup_queue_state_by_doorbell(sig);
    if(!s)
    {
        // Not one of our queues.
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    if(s->mode == QueueState::Mode::tracing_only)
    {
        process_doorbell_tracing_only(s, val);
        // ALWAYS chain through — the application's queue is the real HSA
        // queue; the GPU needs to see the doorbell. Unlike
        // process_doorbell_impl, the tracing_only path does not chain
        // internally.
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    // Mode::full_intercept — existing PR 5219 path. process_doorbell_impl
    // chains through via the lambda (directly when no new packets, or via
    // publish_submitted_packets/tls_ring_doorbell otherwise).
    process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
        s_next_table.hsa_signal_store_screlease_fn(db, v);
    });
}

}  // namespace

bool
is_intercepting_inline()
{
    return s_intercept_installed.load(std::memory_order_acquire);
}

void
shutdown_intercept()
{
    s_intercept_installed.store(false, std::memory_order_release);

    // §3.5 Invariants 3 (single mutex) and 4 (exactly-once retirement):
    // Walk every QueueState still in the registry and drain any
    // captured-but-not-consumed corr_slots before clearing the registry.
    // Without this, process shutdown with in-flight slots leaks the
    // correlation_id refcounts. Take each state's slot_publish_mu so the
    // drain serializes against any (now-bypassed) capture/consume.
    get_queue_registry().rlock([](const auto& map) {
        for(const auto& kv : map)
        {
            const auto& state = kv.second;
            if(!state) continue;
            std::lock_guard<std::mutex> g(state->slot_publish_mu);
            for(auto& entry : state->corr_slots)
            {
                if(entry.gen.load(std::memory_order_relaxed) != 0 && entry.corr_id)
                {
                    entry.corr_id->sub_kern_count();
                    entry.corr_id->sub_ref_count();
                    entry.corr_id = nullptr;
                    entry.gen.store(0, std::memory_order_release);
                }
            }
        }
    });

    get_queue_registry().wlock([](auto& map) { map.clear(); });
    get_doorbell_map().wlock([](auto& map) { map.clear(); });
}

void
install_intercept(CoreApiTable& core_table)
{
    ROCP_INFO << "[queue-intercept] inline intercept path ENGAGED (tracing-only, no expansion)";

    // Save current table entries as our next-in-chain (tracing functors when called
    // after update_table, or raw HSA functions otherwise)
    s_next_table = core_table;

    core_table.hsa_queue_add_write_index_relaxed_fn     = wrap_add_write_index_relaxed;
    core_table.hsa_queue_add_write_index_scacq_screl_fn = wrap_add_write_index_scacq_screl;
    core_table.hsa_queue_add_write_index_scacquire_fn   = wrap_add_write_index_scacquire;
    core_table.hsa_queue_add_write_index_screlease_fn   = wrap_add_write_index_screlease;

    core_table.hsa_queue_store_write_index_relaxed_fn   = wrap_store_write_index_relaxed;
    core_table.hsa_queue_store_write_index_screlease_fn = wrap_store_write_index_screlease;

    core_table.hsa_queue_cas_write_index_relaxed_fn     = wrap_cas_write_index_relaxed;
    core_table.hsa_queue_cas_write_index_scacq_screl_fn = wrap_cas_write_index_scacq_screl;
    core_table.hsa_queue_cas_write_index_scacquire_fn   = wrap_cas_write_index_scacquire;
    core_table.hsa_queue_cas_write_index_screlease_fn   = wrap_cas_write_index_screlease;

    core_table.hsa_queue_load_write_index_relaxed_fn   = wrap_load_write_index_relaxed;
    core_table.hsa_queue_load_write_index_scacquire_fn = wrap_load_write_index_scacquire;

    core_table.hsa_signal_store_relaxed_fn   = wrap_signal_store_relaxed;
    core_table.hsa_signal_store_screlease_fn = wrap_signal_store_screlease;

    s_intercept_installed.store(true, std::memory_order_release);
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
