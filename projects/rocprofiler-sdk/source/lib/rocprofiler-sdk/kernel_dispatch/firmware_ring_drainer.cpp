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
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

struct timed_record_t
{
    uint64_t ts;
    uint32_t record_type;
    uint32_t slot;
    uint32_t dispatch_idx;
};

struct queue_ring_state_t
{
    hsa_queue_t*       queue{};
    void*              buf{};
    uint32_t           ring_bytes{};
    volatile uint32_t* wptr{};
    uint32_t           record_size{};
    hsa_agent_t        agent{};
    uint64_t           dispatch_count{0};
    // TODO(ai/KNOWN_ISSUES.md item 5): last_processed_record_count is a
    // monotonic per-buffer counter that grows without bound and is reset
    // only at shutdown. Acceptable for short-lived processes; not for
    // long-running ones.
    uint32_t last_processed_record_count{0};
};

std::mutex                                       g_ring_mu;
std::unordered_map<uint64_t, queue_ring_state_t> g_queue_rings;
std::atomic<bool>                                g_drainer_stop{true};
std::thread                                      g_drainer_thread;
std::atomic<rocprofiler_dispatch_id_t>           g_next_dispatch_id{1};

// TODO(ai/KNOWN_ISSUES.md item 5): g_emitted_dispatch_idx is an unbounded
// unordered_set that is never pruned during normal operation. After
// 2^32 dispatches the dispatch_idx wraps and dedup will become
// incorrect. Bound this in a follow-up by per-queue ring of recently
// seen indices.
std::unordered_set<uint32_t> g_emitted_dispatch_idx;

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
emit_kernel_dispatch_tracing(hsa_agent_t                hag,
                             hsa_queue_t*               queue,
                             uint64_t                   raw_start_ts,
                             uint64_t                   raw_end_ts,
                             uint64_t                   kernel_object,
                             rocprofiler_dispatch_id_t  dispatch_id,
                             context::correlation_id*   cid)
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

    // TODO(ai/KNOWN_ISSUES.md item 9): thread_id is the drainer thread's
    // tid, not the launching thread's. The interception path captures
    // the launching tid at enqueue time. Without queue interception we
    // do not know the launching tid here.
    auto  thr_id           = common::get_tid();
    auto  internal_corr_id = cid->internal;
    auto  ancestor_corr_id = cid->ancestor;
    auto& extern_corr      = td.external_correlation_ids;

    if(!td.callback_contexts.empty())
    {
        tracing::execute_phase_none_callbacks(td.callback_contexts,
                                              thr_id,
                                              internal_corr_id,
                                              extern_corr,
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
                                               extern_corr,
                                               ancestor_corr_id,
                                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                               std::move(record));
    }
}

void
process_dispatch_record(queue_ring_state_t* st,
                        uint64_t            raw_start_ts,
                        uint64_t            raw_end_ts,
                        uint64_t            kernel_object)
{
    constexpr uint32_t init_ref = 2;
    auto* cid = context::correlation_tracing_service::construct(init_ref);
    // TODO(ai/KNOWN_ISSUES.md item 1): during finalization the correlation
    // service may already be shut down and return null. We hand back a
    // zero-initialized fallback so the dispatch record still flows out.
    // This means the record's correlation IDs will be all zero and
    // cannot be linked to anything. Real fix is a flush-handshake on
    // shutdown so the drainer drains before the correlation service
    // tears down.
    static thread_local context::correlation_id fallback_cid{};
    if(!cid) cid = &fallback_cid;

    auto dispatch_id = g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    emit_kernel_dispatch_tracing(
        st->agent, st->queue, raw_start_ts, raw_end_ts, kernel_object, dispatch_id, cid);
}

hsa_status_t
register_or_refresh_queue(hsa_queue_t* queue, void* /*data*/)
{
    if(rocprofiler::registration::get_fini_status() > 0) return HSA_STATUS_SUCCESS;

    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        if(g_queue_rings.count(queue->id)) return HSA_STATUS_SUCCESS;
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

        std::vector<timed_record_t> records;
        records.reserve(num_slots);
        for(uint32_t i = 0; i < num_slots; i++)
        {
            mec_dispatch_record_16 r16{};
            std::memcpy(&r16, base + i * sizeof(r16), sizeof(r16));
            const uint64_t ts = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;
            if(ts != 0 && (r16.record_type == 1 || r16.record_type == 2))
                records.push_back({ts, r16.record_type, i, r16.dispatch_idx});
        }

        if(records.size() <= qs.last_processed_record_count) continue;
        qs.last_processed_record_count = static_cast<uint32_t>(records.size());

        // Filter records whose dispatch_idx has already been emitted by
        // a prior drain pass.
        std::vector<timed_record_t> fresh;
        fresh.reserve(records.size());
        for(const auto& r : records)
        {
            if(!g_emitted_dispatch_idx.count(r.dispatch_idx)) fresh.push_back(r);
        }

        std::sort(fresh.begin(), fresh.end(),
                  [](const timed_record_t& a, const timed_record_t& b) { return a.ts < b.ts; });

        queue_ring_state_t* lookup_qs = (aql_qs ? aql_qs : &qs);

        // TODO(ai/KNOWN_ISSUES.md item 2): pair START with END using a
        // smallest-positive-gap heuristic. This produces correct
        // pairings only when concurrent dispatches are well-separated
        // in time. For overlapping kernels on different XCCs the
        // heuristic will mis-pair. Real fix needs an XCC/pipe id in
        // the firmware record.
        struct pending_t
        {
            uint64_t ts;
            uint32_t dispatch_idx;
        };
        std::vector<pending_t> pending_starts;

        for(const auto& r : fresh)
        {
            if(r.record_type == 1)
            {
                pending_starts.push_back({r.ts, r.dispatch_idx});
            }
            else if(r.record_type == 2 && !pending_starts.empty())
            {
                int      best_i   = -1;
                uint64_t best_gap = UINT64_MAX;
                for(int i = static_cast<int>(pending_starts.size()) - 1; i >= 0; i--)
                {
                    if(pending_starts[i].ts < r.ts)
                    {
                        const uint64_t gap = r.ts - pending_starts[i].ts;
                        if(gap < best_gap)
                        {
                            best_gap = gap;
                            best_i   = i;
                        }
                    }
                }
                if(best_i < 0) continue;

                const auto start = pending_starts[best_i];
                pending_starts.erase(pending_starts.begin() + best_i);

                if(g_emitted_dispatch_idx.count(start.dispatch_idx)) continue;

                const uint64_t kernel_obj =
                    lookup_kernel_object(*lookup_qs, start.dispatch_idx);

                process_dispatch_record(lookup_qs, start.ts, r.ts, kernel_obj);
                g_emitted_dispatch_idx.insert(start.dispatch_idx);
                lookup_qs->dispatch_count++;
            }
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
            {
                std::lock_guard<std::mutex> lk(g_ring_mu);
                discover_queues();
                drain_all();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Final drain after stop is requested. Reset the per-queue
    // last_processed_record_count so we re-examine every slot in case
    // late records arrived after the most recent drain pass.
    if(hsa::firmware_dispatch_ring_available())
    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        for(auto& [_, qs] : g_queue_rings)
            qs.last_processed_record_count = 0;
        discover_queues();
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
    g_emitted_dispatch_idx.clear();
}

}  // namespace kernel_dispatch
}  // namespace rocprofiler
