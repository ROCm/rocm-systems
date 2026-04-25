// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/dispatch_ring_buffer_support.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
{
// 16-byte MEC firmware dispatch record. Two records per dispatch:
//   record_type == 1 -> dispatch start
//   record_type == 2 -> dispatch end (EOP)
// dispatch_idx is read_dispatch_id[31:0] from the AQL queue, used to
// look up the kernel object from queue->base_address[dispatch_idx %
// queue->size].kernel_object.
#pragma pack(push, 1)
struct mec_dispatch_record_16
{
    uint32_t ts_lo;
    uint32_t ts_hi;
    uint32_t record_type;
    uint32_t dispatch_idx;
};
#pragma pack(pop)

struct queue_ring_state_t
{
    hsa_queue_t*       queue{};
    void*              buf{};
    uint32_t           ring_bytes{};
    volatile uint32_t* wptr{};
    uint32_t           record_size{};
    hsa_agent_t        agent{};
    uint64_t           dispatch_count{0};

    // Per-ring read cursor: index into the firmware ring (modulo
    // num_slots) of the next slot to read. Advances monotonically as
    // we consume. Wraparound is handled by the
    // last_consumed_dispatch_idx check below: an old slot whose
    // record we already consumed has the same dispatch_idx as last
    // time, while a freshly-overwritten slot has a different one.
    uint32_t read_cursor{0};

    // Per-slot last-consumed dispatch_idx, used to detect when the
    // firmware has overwritten a slot we already consumed (vs. a new
    // record at a still-empty slot). Sized to num_slots on first
    // drain.
    std::vector<uint32_t> last_consumed_dispatch_idx;

    // Per-queue pending START records, keyed by dispatch_idx for
    // exact O(1) pairing with END records. Maps dispatch_idx ->
    // start timestamp.
    std::unordered_map<uint32_t, uint64_t> pending_starts;

    // Phase 1: stored at registration so the drainer can call
    // queue_intercept::lookup_queue_state(queue_ptr) to reach the
    // correlation side-table populated by the launching-thread doorbell
    // hook. See PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md §3.4.
    hsa_queue_t* queue_ptr = nullptr;
};

std::mutex                                       g_ring_mu;
std::unordered_map<uint64_t, queue_ring_state_t> g_queue_rings;
std::atomic<bool>                                g_drainer_stop{true};
std::thread                                      g_drainer_thread;
std::atomic<rocprofiler_dispatch_id_t>           g_next_dispatch_id{1};

uint32_t
infer_record_size(uint32_t ring_bytes)
{
    if(ring_bytes >= sizeof(mec_dispatch_record_16) &&
       ring_bytes % sizeof(mec_dispatch_record_16) == 0)
    {
        return sizeof(mec_dispatch_record_16);
    }
    return 0;
}

uint64_t
lookup_kernel_object(queue_ring_state_t& qs, uint32_t dispatch_idx)
{
    if(!qs.queue || !qs.queue->base_address || qs.queue->size == 0) return 0;
    const uint32_t q_size = qs.queue->size;
    const uint32_t slot   = dispatch_idx % q_size;
    const auto*    pkts =
        static_cast<const hsa_kernel_dispatch_packet_t*>(qs.queue->base_address);
    uint64_t ko = 0;
    std::memcpy(&ko, &pkts[slot].kernel_object, sizeof(ko));
    return ko;
}

void
emit_kernel_dispatch_tracing(hsa_agent_t                            hag,
                             hsa_queue_t*                           queue,
                             uint64_t                               raw_start_ts,
                             uint64_t                               raw_end_ts,
                             uint64_t                               kernel_object,
                             rocprofiler_dispatch_id_t              dispatch_id,
                             context::correlation_id*               cid,
                             rocprofiler_thread_id_t                thr_id,
                             tracing::external_correlation_id_map_t ext_ids,
                             uint64_t /*enqueue_ts*/)
{
    tracing::tracing_data td{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                               td);
    if(td.callback_contexts.empty() && td.buffered_contexts.empty()) return;

    uint64_t start_ns = raw_start_ts;
    uint64_t end_ns   = raw_end_ts;
    auto*    ext      = hsa::get_amd_ext_table();
    if(ext && ext->hsa_amd_profiling_convert_tick_to_system_domain_fn)
    {
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, raw_start_ts, &start_ns);
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, raw_end_ts, &end_ns);
    }

    const auto* rocp_agent = rocprofiler::agent::get_rocprofiler_agent(hag);
    if(!rocp_agent) return;

    auto kid = rocprofiler::code_object::get_kernel_id(kernel_object);
    // TODO(ai/KNOWN_ISSUES.md item 7): on a late attach, the code object
    // load callback may not have run for this kernel yet. We currently
    // surface the raw kernel_object handle in place of a kernel_id so
    // tools can still attribute timing. A real fix is to either (a) pre-
    // populate the code object cache from existing executables at
    // attach time, or (b) defer record emission until the symbol is
    // known.
    if(kid == 0 && kernel_object != 0) kid = kernel_object;

    constexpr auto kernel_dispatch_info_rt_size =
        common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();
    auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    dispatch_info.size      = kernel_dispatch_info_rt_size;
    dispatch_info.agent_id  = rocp_agent->id;
    dispatch_info.queue_id  = rocprofiler_queue_id_t{queue->id};
    dispatch_info.kernel_id = rocprofiler_kernel_id_t{kid};
    dispatch_info.dispatch_id = dispatch_id;
    // TODO(ai/KNOWN_ISSUES.md item 8): workgroup_size, grid_size, and
    // segment sizes are zeroed because we do not yet read them off the
    // AQL packet at lookup time. lookup_kernel_object() reads
    // queue->base_address[dispatch_idx % size]; reading the same
    // hsa_kernel_dispatch_packet_t for these fields is a small follow-up.
    dispatch_info.workgroup_size       = {0, 0, 0};
    dispatch_info.grid_size            = {0, 0, 0};
    dispatch_info.private_segment_size = 0;
    dispatch_info.group_segment_size   = 0;

    auto tracer_data = rocprofiler_callback_tracing_kernel_dispatch_data_t{};
    tracer_data.size            = sizeof(tracer_data);
    tracer_data.start_timestamp = start_ns;
    tracer_data.end_timestamp   = end_ns;
    tracer_data.dispatch_info   = dispatch_info;

    // tid and external correlation IDs are captured at launching-thread
    // doorbell-store time and passed in via parameters (Phase 1
    // tracing-only). For the fallback path they are populated by the
    // caller (process_dispatch_record) using common::get_tid() and an
    // empty external_corr_ids map.
    auto internal_corr_id = cid->internal;
    auto ancestor_corr_id = cid->ancestor;

    if(!td.callback_contexts.empty())
    {
        tracing::execute_phase_none_callbacks(td.callback_contexts,
                                              thr_id,
                                              internal_corr_id,
                                              ext_ids,
                                              ancestor_corr_id,
                                              ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                              ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                              tracer_data);
    }

    if(!td.buffered_contexts.empty())
    {
        auto record = rocprofiler_buffer_tracing_kernel_dispatch_record_t{
            sizeof(rocprofiler_buffer_tracing_kernel_dispatch_record_t),
            ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
            rocprofiler_async_correlation_id_t{},
            thr_id,
            start_ns,
            end_ns,
            dispatch_info};
        tracing::execute_buffer_record_emplace(td.buffered_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               ext_ids,
                                               ancestor_corr_id,
                                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                               std::move(record));
    }
}

void
process_dispatch_record(queue_ring_state_t* st,
                        uint32_t            dispatch_idx,
                        uint64_t            raw_start_ts,
                        uint64_t            raw_end_ts,
                        uint64_t            kernel_object)
{
    // Look up correlation captured by the launching-thread doorbell hook.
    // See PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md §3.4-3.5.
    auto qstate = hsa::queue_intercept::lookup_queue_state(st->queue_ptr);

    context::correlation_id*               cid    = nullptr;
    rocprofiler_thread_id_t                thr_id = 0;
    uint64_t                               enq_ts = 0;
    tracing::external_correlation_id_map_t ext_ids;

    if(qstate && !qstate->corr_slots.empty())
    {
        // §3.5 Invariant 3: take the slot mutex.
        std::lock_guard<std::mutex> g(qstate->slot_publish_mu);
        auto& slot = qstate->corr_slots[dispatch_idx & qstate->corr_ring_mask];

        // §3.5 Invariant 1: alias guard. The firmware record carries the
        // low 32 bits of the wdid; reject the slot if it's been
        // overwritten by a wraparound capture since this dispatch was
        // enqueued.
        if(slot.gen.load(std::memory_order_acquire) != 0 && slot.corr_id != nullptr &&
           static_cast<uint32_t>(slot.captured_wdid & 0xFFFFFFFFu) == dispatch_idx)
        {
            cid     = slot.corr_id;
            thr_id  = slot.tid;
            enq_ts  = slot.enqueue_ts;
            ext_ids = std::move(slot.external_corr_ids);
            // Consume: clear corr_id then gen.
            slot.corr_id = nullptr;
            slot.gen.store(0, std::memory_order_release);
        }
    }

    // Late-attach for an in-flight kernel, alias-rejected, or slot
    // already cleaned up by destroy_queue_state. Fallback emission with
    // zero correlation IDs and the drainer thread's tid.
    //
    // TODO(ai/KNOWN_ISSUES.md item 1): the static thread_local
    // fallback_cid carries zeroed internal/ancestor IDs and is NOT
    // refcount-managed. Tools cannot link these records back to a
    // launching API call.
    static thread_local context::correlation_id fallback_cid{};
    if(!cid)
    {
        cid    = &fallback_cid;
        thr_id = common::get_tid();
        // ext_ids stays empty
    }

    auto dispatch_id = g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    emit_kernel_dispatch_tracing(st->agent,
                                 st->queue,
                                 raw_start_ts,
                                 raw_end_ts,
                                 kernel_object,
                                 dispatch_id,
                                 cid,
                                 thr_id,
                                 std::move(ext_ids),
                                 enq_ts);

    // §3.5 Invariant 4: drainer-consume retirement path. Skip for the
    // fallback path (the fallback_cid is a static thread_local sentinel
    // that doesn't participate in refcount lifecycle).
    if(cid != &fallback_cid)
    {
        cid->sub_kern_count();
        cid->sub_ref_count();
    }
}

hsa_status_t
register_or_refresh_queue(hsa_queue_t* queue, void* /*data*/)
{
    if(rocprofiler::registration::get_fini_status() > 0) return HSA_STATUS_SUCCESS;

    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        auto                        it = g_queue_rings.find(queue->id);
        if(it != g_queue_rings.end())
        {
            if(it->second.queue_ptr == queue)
            {
                // Same queue (and same address) — already registered.
                return HSA_STATUS_SUCCESS;
            }
            // Same id, different pointer = HSA reused the id after
            // destroy. Evict the stale ring/metadata entry. The
            // associated QueueState (if any) is owned by
            // queue_intercept and will be cleaned up when its
            // hsa_queue_destroy fires (which it should have, ahead of
            // the new create). Then fall through to fresh registration.
            g_queue_rings.erase(it);
        }
    }

    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_queue_get_info_fn ||
       !ext->hsa_amd_profiling_set_profiler_enabled_fn)
        return HSA_STATUS_SUCCESS;

    hsa_agent_t agent{};
    if(ext->hsa_amd_queue_get_info_fn(queue, HSA_AMD_QUEUE_INFO_AGENT, &agent) !=
       HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    hsa_device_type_t dt = HSA_DEVICE_TYPE_CPU;
    if(hsa::get_core_table()->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &dt) !=
           HSA_STATUS_SUCCESS ||
       dt != HSA_DEVICE_TYPE_GPU)
        return HSA_STATUS_SUCCESS;

    // TODO(ai/KNOWN_ISSUES.md item 6): we call
    // hsa_amd_profiling_set_profiler_enabled(true) on every discovered
    // GPU queue. This is an externally observable side effect and we
    // never disable it. Move to first-discovery-only and tear down on
    // stop.
    if(ext->hsa_amd_profiling_set_profiler_enabled_fn(queue, true) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    void*              buf  = nullptr;
    uint32_t           sz   = 0;
    volatile uint32_t* wptr = nullptr;
    auto get_fn = hsa::dispatch_ring_buffer_get_dispatch_records_fn_v();
    if(!get_fn || get_fn(queue, &buf, &sz, &wptr) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    const uint32_t rec_sz = infer_record_size(sz);
    if(rec_sz == 0 || !buf || !wptr) return HSA_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lk(g_ring_mu);
    auto&                       ent = g_queue_rings[queue->id];
    ent.queue       = queue;
    ent.buf         = buf;
    ent.ring_bytes  = sz;
    ent.wptr        = wptr;
    ent.agent       = agent;
    ent.record_size = rec_sz;
    ent.queue_ptr   = queue;

    // Late-attach: ensure a QueueState exists for this queue so the
    // doorbell wrapper has a place to capture correlation. Idempotent —
    // returns immediately if a state already exists (e.g., the queue was
    // created after the SDK loaded and went through the normal
    // QueueController::add_queue → create_queue_state path).
    //
    // We always pass Mode::tracing_only here because the drainer only
    // runs when the firmware-ring tracing-only path is engaged (Task 8
    // gate).
    auto* amd_q = reinterpret_cast<amd_queue_t*>(queue);
    hsa::queue_intercept::create_queue_state(
        queue,
        &amd_q->write_dispatch_id,
        &amd_q->read_dispatch_id,
        hsa::queue_intercept::QueueState::Mode::tracing_only);

    return HSA_STATUS_SUCCESS;
}

void
drain_all()
{
    // Find any queue with a populated AQL ring; we use it as the source
    // of kernel_object lookups when the queue we found the record on
    // does not itself have a base_address. This is a multi-XCC
    // workaround.
    queue_ring_state_t* aql_qs = nullptr;
    for(auto& [_, qs] : g_queue_rings)
    {
        if(qs.queue && qs.queue->base_address && qs.queue->size > 0)
        {
            aql_qs = &qs;
            break;
        }
    }

    for(auto& [qid, qs] : g_queue_rings)
    {
        if(!qs.buf || qs.record_size == 0 || qs.ring_bytes < qs.record_size) continue;
        if(qs.record_size != sizeof(mec_dispatch_record_16)) continue;

        const auto*    base      = static_cast<const uint8_t*>(qs.buf);
        const uint32_t num_slots = qs.ring_bytes / qs.record_size;

        // Lazy-init last_consumed_dispatch_idx on first drain (or on
        // the first drain after the ring geometry changes). UINT32_MAX
        // is a sentinel that doesn't equal any real dispatch_idx the
        // firmware will produce on the first pass, so the first
        // consumed record at every slot is always treated as fresh.
        if(qs.last_consumed_dispatch_idx.size() != num_slots)
            qs.last_consumed_dispatch_idx.assign(num_slots, UINT32_MAX);

        // Correlation lookup MUST use the queue that owns the
        // firmware-ring records — the launching thread captured into
        // THIS queue's corr_slots, not aql_qs's. The aql_qs fallback is
        // ONLY for kernel_object lookup in the multi-XCC case where qs
        // itself lacks a base_address.
        queue_ring_state_t* corr_lookup_qs = &qs;
        queue_ring_state_t* kobj_lookup_qs =
            (qs.queue && qs.queue->base_address) ? &qs : aql_qs;
        if(!kobj_lookup_qs) continue;  // no source for kernel_object — skip

        // Walk forward from the per-ring read cursor. Bounded at
        // num_slots iterations per drain pass to keep per-tick work
        // proportional to ring size and prevent starving other queues
        // when one queue is producing fast.
        for(uint32_t consumed = 0; consumed < num_slots; ++consumed)
        {
            const uint32_t slot = qs.read_cursor;

            mec_dispatch_record_16 r16{};
            std::memcpy(&r16, base + slot * sizeof(r16), sizeof(r16));

            const uint64_t ts = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;

            // Empty slot — firmware hasn't written here yet. Stop the
            // scan; subsequent slots can't be newer.
            if(ts == 0) break;

            // Already-consumed slot: same dispatch_idx as last time we
            // looked. The firmware hasn't overwritten it since, so
            // there's nothing new past this point either.
            //
            // Caveat: if 32-bit dispatch_idx wraps AND the new record
            // happens to land on the same low-32-bits as the
            // last-consumed one, we'd miss it. That's the same
            // 4-billion-dispatch wrap concern called out elsewhere.
            if(qs.last_consumed_dispatch_idx[slot] == r16.dispatch_idx) break;

            // Defensive skip for unknown record types. Still advance the
            // cursor so we don't get stuck.
            if(r16.record_type != 1 && r16.record_type != 2)
            {
                qs.last_consumed_dispatch_idx[slot] = r16.dispatch_idx;
                qs.read_cursor                      = (qs.read_cursor + 1) % num_slots;
                continue;
            }

            if(r16.record_type == 1)
            {
                // START: park the timestamp keyed by dispatch_idx for
                // the matching END to find. If a duplicate START
                // arrives (shouldn't happen) the newer one wins.
                qs.pending_starts[r16.dispatch_idx] = ts;
            }
            else  // r16.record_type == 2 (END)
            {
                auto it = qs.pending_starts.find(r16.dispatch_idx);
                if(it != qs.pending_starts.end())
                {
                    const uint64_t start_ts = it->second;
                    qs.pending_starts.erase(it);

                    const uint64_t kernel_obj =
                        lookup_kernel_object(*kobj_lookup_qs, r16.dispatch_idx);

                    process_dispatch_record(
                        corr_lookup_qs, r16.dispatch_idx, start_ts, ts, kernel_obj);
                    corr_lookup_qs->dispatch_count++;
                }
                // No matching START: late-attached drainer or lost
                // START record. Drop silently and keep going.
            }

            qs.last_consumed_dispatch_idx[slot] = r16.dispatch_idx;
            qs.read_cursor                      = (qs.read_cursor + 1) % num_slots;
        }
    }
}

void
discover_queues()
{
    auto it_fn = hsa::dispatch_ring_buffer_queue_iterate_fn_v();
    if(it_fn) it_fn(register_or_refresh_queue, nullptr);
}

void
drainer_loop()
{
    // TODO(ai/KNOWN_ISSUES.md item 4): hard-coded 1ms cadence; no adaptive
    // backoff. On idle processes this is 1000 wakeups/sec.
    while(!g_drainer_stop.load(std::memory_order_acquire))
    {
        if(hsa::firmware_dispatch_ring_available())
        {
            // Discover queues OUTSIDE g_ring_mu — register_or_refresh_queue
            // (the iterate callback) takes g_ring_mu itself. Holding the
            // mutex across discover_queues would self-deadlock because
            // std::mutex is non-recursive.
            discover_queues();

            // Drain holds the lock for the duration of the iteration.
            std::lock_guard<std::mutex> lk(g_ring_mu);
            drain_all();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Final drain after stop is requested. With the per-ring cursor
    // model there's no scan/dedup state to reset — drain_all() picks
    // up wherever the cursor was left and consumes any newly-arrived
    // records past it.
    if(hsa::firmware_dispatch_ring_available())
    {
        // Same lock-ordering rule as above: discover OUTSIDE the mutex.
        discover_queues();

        std::lock_guard<std::mutex> lk(g_ring_mu);
        drain_all();
    }
}
}  // namespace

void
start_firmware_dispatch_ring_drainer()
{
    if(!hsa::firmware_dispatch_ring_available()) return;

    g_drainer_stop.store(false, std::memory_order_release);
    if(g_drainer_thread.joinable()) g_drainer_thread.join();
    g_drainer_thread = std::thread{drainer_loop};
}

void
unregister_queue(hsa_queue_t* queue)
{
    if(!queue) return;
    std::lock_guard<std::mutex> lk(g_ring_mu);
    g_queue_rings.erase(queue->id);
}

void
stop_firmware_dispatch_ring_drainer()
{
    if(!g_drainer_thread.joinable()) return;

    // TODO(ai/KNOWN_ISSUES.md item 3): the 10ms grace sleep gives the
    // drainer a few extra cycles to pick up records emitted by recent
    // dispatches. This is a substitute for a proper flush handshake
    // with the firmware. Replace with a real fence in a follow-up.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    g_drainer_stop.store(true, std::memory_order_release);
    g_drainer_thread.join();

    std::lock_guard<std::mutex> lk(g_ring_mu);
    g_queue_rings.clear();
}

}  // namespace kernel_dispatch
}  // namespace rocprofiler
