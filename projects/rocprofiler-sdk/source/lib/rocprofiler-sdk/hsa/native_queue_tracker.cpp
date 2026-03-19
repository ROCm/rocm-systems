// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/native_queue_tracker.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace
{
static std::atomic<rocprofiler_dispatch_id_t> native_dispatch_counter{0};

// AQL packet type is in bits [0..7] of the header field.
constexpr uint8_t
aql_packet_type(uint16_t header)
{
    return static_cast<uint8_t>(header & 0xFF);
}
}  // namespace

NativeQueueTracker::~NativeQueueTracker()
{
    std::lock_guard<std::mutex> lock(_mutex);

    // last attempt to harvest pending dispatches whose GPU signals have completed. Don't throw away valid profiling data just because shutdown happened before next on_hip_api_exit() would've drained them naturally. 
    drain_completed();

    // Anything still pending is a false positive or a dispatch whose signal
    // was reused.  Clean up correlation IDs so finalization doesn't hang.
    for(auto& pd : _pending)
    {
        if(pd && pd->corr_id)
        {
            pd->corr_id->sub_kern_count();
            pd->corr_id->sub_ref_count();
        }
    }
    // this is needed because it was causing hangs in proton.finalize()! unresolved pending dispatches were leading to a deadlock because of rec-count imbalance. 
    _pending.clear();
}

// queue discovery

using hsa_amd_queue_iterate_fn_t = hsa_status_t (*)(
    hsa_status_t (*)(hsa_queue_t*, hsa_agent_t, void*), void*);

// callback invoked once per queue by hsa_amd_queue_iterate()
hsa_status_t
NativeQueueTracker::queue_iterate_cb(hsa_queue_t* queue, hsa_agent_t agent, void* data)
{
    auto* tracker = static_cast<NativeQueueTracker*>(data);

    // skip if already intercepted by WriteInterceptor path
    // maybe not needed 
    auto* qc = get_queue_controller();
    if(qc && qc->get_queue(*queue))
    {
        ROCP_TRACE << "[NQT] queue " << queue->id << " already intercepted, skipping";
        return HSA_STATUS_SUCCESS;
    }

    // don't care about non-GPU agents
    hsa_device_type_t type = {};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if(type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;

    // create a entry for this queue, and use its current write index as that is where the next AQL packet will be written to. 
    // when on_hip_api_exit() is called, it will scan this write index to the current write index and detect which ring buffer slots are new since we started tracking. Anything before this index was submitted before profiling was enabled so we skip it(which we wouldn't have valid timestamps for anyway). 
    auto entry       = std::make_unique<NativeQueueEntry>();
    entry->queue     = queue;
    entry->agent     = agent;
    entry->last_index.store(
        tracker->_core_api.hsa_queue_load_write_index_relaxed_fn(queue),
        std::memory_order_relaxed);

    // enable profiling on this queue so firmware will write timestamps into the completion signal.
    auto status = tracker->_ext_api.hsa_amd_profiling_set_profiler_enabled_fn(queue, 1);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to enable profiling on native queue " << queue->id
                     << " (status " << status << ")";
        return HSA_STATUS_SUCCESS;
    }

    const auto* rocp_agent = agent::get_rocprofiler_agent(agent);
    if(rocp_agent)
    {
        const auto* cached_agent = agent::get_agent_cache(rocp_agent);
        if(cached_agent)
        {
            
            entry->rocp_queue = std::make_unique<Queue>(*cached_agent, tracker->_core_api);
        }
    }

    ROCP_TRACE << "[NQT] tracking native queue " << queue->id
               << " (write_index=" << entry->last_index.load() << ", profiling enabled)";

    tracker->_entries.push_back(std::move(entry));
    return HSA_STATUS_SUCCESS;
}

void
NativeQueueTracker::discover_queues(const CoreApiTable& core_api, const AmdExtTable& ext_api)
{
    ROCP_TRACE << "[NQT] discover_queues called";
    _core_api = core_api;
    _ext_api  = ext_api;
    _initialized.store(true, std::memory_order_release);

    auto* iterate_fn = reinterpret_cast<hsa_amd_queue_iterate_fn_t>(
        dlsym(RTLD_DEFAULT, "hsa_amd_queue_iterate"));

    if(!iterate_fn)
    {
        ROCP_WARNING << "[NQT] dlsym(hsa_amd_queue_iterate) failed — "
                        "custom ROCR runtime not loaded";
        return;
    }

    iterate_fn(queue_iterate_cb, this);
    ROCP_TRACE << "[NQT] discover_queues done: entries=" << _entries.size();
}

void
NativeQueueTracker::try_rediscover()
{
    if(_rediscovered.exchange(true, std::memory_order_acq_rel)) return;

    auto* iterate_fn = reinterpret_cast<hsa_amd_queue_iterate_fn_t>(
        dlsym(RTLD_DEFAULT, "hsa_amd_queue_iterate"));
    if(!iterate_fn) return;

    ROCP_TRACE << "[NQT] try_rediscover";
    iterate_fn(queue_iterate_cb, this);
    ROCP_TRACE << "[NQT] rediscovery done: entries=" << _entries.size();
}

// ─── HIP API enter/exit: snapshot + detect dispatches ───────────────────────

// Per-thread snapshot of write indices, captured in on_hip_api_enter().
// on_hip_api_exit() reads these to scan only the freshly-written slots,
// avoiding stale data left by the GPU after it clears processed packets.
static thread_local std::vector<uint64_t> tls_enter_write_indices;

void
NativeQueueTracker::on_hip_api_enter()
{
    if(!_initialized.load(std::memory_order_relaxed)) return;

    if(_entries.empty())
    {
        try_rediscover();
        if(_entries.empty()) return;
    }

    auto n = _entries.size();
    tls_enter_write_indices.resize(n);
    for(size_t i = 0; i < n; i++)
    {
        tls_enter_write_indices[i] =
            _core_api.hsa_queue_load_write_index_relaxed_fn(_entries[i]->queue);
    }
}

void
NativeQueueTracker::on_hip_api_exit()
{
    if(_entries.empty()) return;

    std::lock_guard<std::mutex> lock(_mutex);

    drain_completed();

    constexpr uint64_t stale_threshold_ns = 500'000'000ULL;
    discard_stale(stale_threshold_ns);

    bool have_snapshot = (tls_enter_write_indices.size() == _entries.size());

    for(size_t qi = 0; qi < _entries.size(); qi++)
    {
        auto&    entry     = _entries[qi];
        auto*    queue     = entry->queue;
        uint64_t cur_write = _core_api.hsa_queue_load_write_index_relaxed_fn(queue);

        uint64_t scan_from = have_snapshot
                                 ? tls_enter_write_indices[qi]
                                 : entry->last_index.load(std::memory_order_relaxed);

        entry->last_index.store(cur_write, std::memory_order_relaxed);

        if(cur_write <= scan_from)
            continue;

        auto*    base = reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(queue->base_address);
        uint32_t mask = queue->size - 1;

        for(uint64_t idx = scan_from; idx < cur_write; idx++)
        {
            const auto& pkt = base[idx & mask];
            uint8_t ptype = aql_packet_type(pkt.header);

            // Accept kernel dispatches (type=2) OR packets the GPU already
            // consumed (type=0 or type=1) that still have a non-null signal
            // and kernel_object — the GPU clears the type field but not the
            // rest of the packet, so fast-completing kernels can be recovered.
            bool is_kernel = (ptype == HSA_PACKET_TYPE_KERNEL_DISPATCH) ||
                             (ptype <= 1 && pkt.kernel_object != 0);
            if(!is_kernel)
                continue;

            if(pkt.completion_signal.handle == 0) continue;
            if(registration::get_fini_status() > 0) continue;

            auto* rqueue = entry->rocp_queue.get();
            if(!rqueue) continue;

            auto tracing_data_v = tracing::tracing_data{};
            tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                       ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                       tracing_data_v);
            if(tracing_data_v.callback_contexts.empty() &&
               tracing_data_v.buffered_contexts.empty())
                continue;

            // Use the HIP callback's correlation ID when available — it's
            // registered in Proton's corrIdToExternId map so buffer records
            // will be associated with the correct scope.
            auto* corr_id     = context::get_latest_correlation_id();
            context::correlation_id* corr_id_pop = nullptr;
            if(!corr_id)
            {
                corr_id     = context::correlation_tracing_service::construct(1);
                corr_id_pop = corr_id;
            }
            if(!corr_id) continue;

            corr_id->add_ref_count();
            corr_id->add_kern_count();

            auto thr_id           = corr_id->thread_idx;
            auto internal_corr_id = corr_id->internal;
            auto ancestor_corr_id = corr_id->ancestor;

            auto _dtor = common::scope_destructor{[corr_id_pop]() {
                if(corr_id_pop)
                {
                    context::pop_latest_correlation_id(corr_id_pop);
                    corr_id_pop->sub_ref_count();
                }
            }};

            tracing::populate_external_correlation_ids(
                tracing_data_v.external_correlation_ids, thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, internal_corr_id);

            uint64_t kernel_id = code_object::get_kernel_id(pkt.kernel_object);

            constexpr auto info_rt_size =
                common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

            auto dispatch_id     = ++native_dispatch_counter;
            auto callback_record = rocprofiler_callback_tracing_kernel_dispatch_data_t{
                sizeof(rocprofiler_callback_tracing_kernel_dispatch_data_t),
                rocprofiler_timestamp_t{0},
                rocprofiler_timestamp_t{0},
                rocprofiler_kernel_dispatch_info_t{
                    .size                 = info_rt_size,
                    .agent_id             = rqueue->get_agent().get_rocp_agent()->id,
                    .queue_id             = {.handle = queue->id},
                    .kernel_id            = kernel_id,
                    .dispatch_id          = dispatch_id,
                    .private_segment_size = pkt.private_segment_size,
                    .group_segment_size   = pkt.group_segment_size,
                    .workgroup_size       = rocprofiler_dim3_t{pkt.workgroup_size_x,
                                                         pkt.workgroup_size_y,
                                                         pkt.workgroup_size_z},
                    .grid_size            = rocprofiler_dim3_t{pkt.grid_size_x,
                                                    pkt.grid_size_y,
                                                    pkt.grid_size_z},
                    .reserved_padding     = {0}}};

            {
                auto tracer_data = callback_record;
                tracing::execute_phase_enter_callbacks(
                    tracing_data_v.callback_contexts, thr_id, internal_corr_id,
                    tracing_data_v.external_correlation_ids, ancestor_corr_id,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, tracer_data);
            }

            tracing::update_external_correlation_ids(
                tracing_data_v.external_correlation_ids, thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

            // No async_started() — we manage our own lifecycle.

            auto pending       = std::make_unique<PendingDispatch>();
            pending->queue     = rqueue;
            pending->signal    = pkt.completion_signal;
            pending->hsa_agent = entry->agent;
            pending->kernel_id = kernel_id;
            pending->dispatch_id   = dispatch_id;
            pending->tid           = thr_id;
            pending->enqueue_ts    = common::timestamp_ns();
            pending->callback_record = callback_record;
            pending->tracing_data    = tracing_data_v;
            pending->corr_id         = corr_id;

            _pending.push_back(std::move(pending));

            {
                auto tracer_data = callback_record;
                tracing::execute_phase_exit_callbacks(
                    tracing_data_v.callback_contexts,
                    tracing_data_v.external_correlation_ids,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, tracer_data);
            }
        }
    }
}

void
NativeQueueTracker::drain_completed()
{
    if(_pending.empty()) return;

    auto it = _pending.begin();
    while(it != _pending.end())
    {
        auto* pd  = it->get();
        auto* sig = reinterpret_cast<const amd_signal_t*>(pd->signal.handle);
        auto  val = __atomic_load_n(&sig->value, __ATOMIC_ACQUIRE);

        if(val > 0)
        {
            ++it;
            continue;
        }

        uint64_t start = sig->start_ts;
        uint64_t end   = sig->end_ts;

        // Sanity check: if timestamps are zero, this signal was never written
        // by the MEC (stale or reused).  Discard silently.
        if(start == 0 || end == 0 || end < start)
        {
            if(pd->corr_id)
            {
                pd->corr_id->sub_kern_count();
                pd->corr_id->sub_ref_count();
            }
            it = _pending.erase(it);
            continue;
        }

        rocprofiler_packet kernel_pkt = {};
        kernel_pkt.kernel_dispatch.completion_signal = pd->signal;

        auto dispatch_time = tracing::profiling_time{
            HSA_STATUS_SUCCESS, start, end};

        queue_info_session info_session{
            .queue            = *pd->queue,
            .inst_pkt         = {},
            .interrupt_signal = {.handle = 0},
            .tid              = pd->tid,
            .enqueue_ts       = pd->enqueue_ts,
            .user_data        = {.value = 0},
            .correlation_id   = pd->corr_id,
            .kernel_pkt       = kernel_pkt,
            .callback_record  = pd->callback_record,
            .tracing_data     = pd->tracing_data,
            .is_serialized    = false};

        kernel_dispatch::dispatch_complete(info_session, dispatch_time);

        if(pd->corr_id)
        {
            pd->corr_id->sub_kern_count();
            pd->corr_id->sub_ref_count();
        }

        // No async_complete() — we don't use Queue's active kernel tracking.

        it = _pending.erase(it);
    }
}

void
NativeQueueTracker::discard_stale(uint64_t max_age_ns)
{
    if(_pending.empty()) return;

    auto now = common::timestamp_ns();
    auto it  = _pending.begin();
    while(it != _pending.end())
    {
        auto* pd = it->get();
        if(now > pd->enqueue_ts &&
           (now - pd->enqueue_ts) > max_age_ns)
        {
            ROCP_TRACE << "[NQT] discarding stale dispatch " << pd->dispatch_id
                       << " (age " << (now - pd->enqueue_ts) / 1000000 << " ms)";
            if(pd->corr_id)
            {
                pd->corr_id->sub_kern_count();
                pd->corr_id->sub_ref_count();
            }
            it = _pending.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void
NativeQueueTracker::flush(uint64_t timeout_ms)
{
    std::lock_guard<std::mutex> lock(_mutex);

    // Poll for completed dispatches over the timeout window.
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    while(!_pending.empty() && std::chrono::steady_clock::now() < deadline)
    {
        drain_completed();
        if(!_pending.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Anything still pending after the timeout is a false positive or
    // a dispatch whose signal was reused before we could read it.
    for(auto& pd : _pending)
    {
        if(pd && pd->corr_id)
        {
            pd->corr_id->sub_kern_count();
            pd->corr_id->sub_ref_count();
        }
    }
    _pending.clear();
}

NativeQueueTracker*
get_native_queue_tracker()
{
    static auto*& tracker = common::static_object<NativeQueueTracker>::construct();
    return tracker;
}

}  // namespace hsa
}  // namespace rocprofiler
