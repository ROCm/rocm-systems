// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hip/event.hpp"
#include "lib/common/abi.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

#include <hsa/amd_hsa_signal.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace hip
{
namespace event
{
using profiling_time = tracing::profiling_time;

// Per-thread record of the HIP event API call currently executing on this thread. The
// barrier interception below consults it to attribute an in-flight barrier packet to the
// hipEventRecord/hipStreamWaitEvent call that produced it.
struct active_event_context_t
{
    rocprofiler_hip_event_operation_t operation        = ROCPROFILER_HIP_EVENT_NONE;
    uint64_t                          hip_event_handle = 0;
    bool                              barrier_captured = false;
    // Stream this hipEventRecord targets, as supplied by the caller. Carried so it can be
    // stored alongside the record info and compared against a later wait's stream.
    hipStream_t record_stream = nullptr;
};

struct event_record_info_t
{
    rocprofiler_queue_id_t queue_id        = {.handle = 0};
    rocprofiler_agent_id_t agent_id        = {.handle = 0};
    uint64_t               original_signal = 0;
    // Stream the event was recorded on, exactly as the caller supplied it. CLR
    // short-circuits a wait issued on this same stream, so a wait matching it produces no
    // GPU dependency to trace. Stored unnormalized: see register_deferred_wait.
    hipStream_t record_stream = nullptr;
    // Set when the record barrier for this generation completes on the GPU. Reset by
    // record_event_info when the event is re-recorded.
    bool completed = false;
};

// One hipEventRecord that CLR folded onto a barrier shared with other records.
struct coalesce_pending_t
{
    tracing::tracing_data                         tracing_data     = {};
    rocprofiler_callback_tracing_hip_event_data_t callback_record  = {};
    rocprofiler_thread_id_t                       tid              = 0;
    uint64_t                                      internal_corr_id = 0;
    uint64_t                                      ancestor_corr_id = 0;
    context::correlation_id*                      corr_id_ref      = nullptr;
};

struct coalesce_group_t
{
    profiling_time                                         barrier_time = {};
    bool                                                   completed    = false;
    common::container::small_vector<coalesce_pending_t, 4> pending      = {};
};

using coalesce_group_ptr_t = std::shared_ptr<common::Synchronized<coalesce_group_t>>;

// Declared ahead of their definitions so that ordering within this file does not matter.
// None of these are part of the interface in event.hpp: they are only reachable here.
void
barrier_complete(tracing::tracing_data&                        tracing_data_v,
                 rocprofiler_thread_id_t                       tid,
                 uint64_t                                      internal_corr_id,
                 uint64_t                                      ancestor_corr_id,
                 profiling_time                                barrier_time,
                 rocprofiler_hip_event_operation_t             operation,
                 rocprofiler_callback_tracing_hip_event_data_t callback_record);

active_event_context_t*
get_active_event_context();

void
record_event_info(uint64_t hip_event_handle, event_record_info_t info);

void
mark_event_completed(uint64_t hip_event_handle);

event_record_info_t
lookup_event_info(uint64_t hip_event_handle);

void
store_coalesce_group(uint64_t hip_event_handle, coalesce_group_ptr_t group);

coalesce_group_ptr_t
lookup_coalesce_group(uint64_t hip_event_handle);

void
erase_event_info(uint64_t hip_event_handle);

bool
has_pending_waits();

// A hipStreamWaitEvent whose GPU dependency CLR folded into a later packet's barrier.
// Registered against the recording event's completion signal, then claimed by the packet
// submission that carries that signal as a dependency.
struct pending_wait_t
{
    tracing::tracing_data    tracing_data     = {};
    rocprofiler_thread_id_t  tid              = 0;
    uint64_t                 internal_corr_id = 0;
    uint64_t                 ancestor_corr_id = 0;
    context::correlation_id* corr_id_ref      = nullptr;
    uint64_t                 hip_event_handle = 0;
    event_record_info_t      source_info      = {};
};

using pending_wait_array_t = common::container::small_vector<pending_wait_t, 4>;

namespace
{
#define ROCPROFILER_HIP_EVENT_INFO(CODE)                                                           \
    template <>                                                                                    \
    struct hip_event_info<ROCPROFILER_##CODE>                                                      \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_##CODE;                                  \
        static constexpr auto name          = #CODE;                                               \
    };

template <size_t Idx>
struct hip_event_info;

ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_NONE)
ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_RECORD)
ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_WAIT)

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hip_event_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HIP_EVENT_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, hip_event_info<Idx>::operation_idx), ...);
}

}  // namespace

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_HIP_EVENT_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
    return _data;
}

void
barrier_complete(tracing::tracing_data&                        tracing_data_v,
                 rocprofiler_thread_id_t                       tid,
                 uint64_t                                      internal_corr_id,
                 uint64_t                                      ancestor_corr_id,
                 profiling_time                                barrier_time,
                 rocprofiler_hip_event_operation_t             operation,
                 rocprofiler_callback_tracing_hip_event_data_t callback_record)
{
    using hip_event_record_t = rocprofiler_buffer_tracing_hip_event_record_t;

    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    // get_barrier_time already logged the failure. Records are only emitted with GPU times.
    if(barrier_time.status != HSA_STATUS_SUCCESS) return;

    const auto& _extern_corr_ids = tracing_data_v.external_correlation_ids;

    callback_record.start_timestamp = barrier_time.start;
    callback_record.end_timestamp   = barrier_time.end;

    if(!tracing_data_v.callback_contexts.empty())
    {
        auto tracer_data = callback_record;
        tracing::execute_phase_none_callbacks(tracing_data_v.callback_contexts,
                                              tid,
                                              internal_corr_id,
                                              _extern_corr_ids,
                                              ancestor_corr_id,
                                              ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                              operation,
                                              tracer_data);
    }

    if(!tracing_data_v.buffered_contexts.empty())
    {
        auto record = common::init_public_api_struct(hip_event_record_t{},
                                                     ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                                                     operation,
                                                     rocprofiler_async_correlation_id_t{},
                                                     tid,
                                                     callback_record.start_timestamp,
                                                     callback_record.end_timestamp,
                                                     callback_record.agent_id,
                                                     callback_record.queue_id,
                                                     callback_record.hip_event_handle,
                                                     callback_record.source_queue_id);

        tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                               tid,
                                               internal_corr_id,
                                               _extern_corr_ids,
                                               ancestor_corr_id,
                                               ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                                               operation,
                                               record);
    }
}
namespace
{
thread_local active_event_context_t g_active_event_ctx = {};

using event_info_map_t     = std::unordered_map<uint64_t, event_record_info_t>;
using coalesce_group_map_t = std::unordered_map<uint64_t, coalesce_group_ptr_t>;
using pending_wait_map_t   = std::unordered_multimap<uint64_t, pending_wait_t>;

using stream_is_capturing_fn_t = hipError_t (*)(hipStream_t, hipStreamCaptureStatus*);

stream_is_capturing_fn_t&
get_stream_is_capturing_fn()
{
    static auto _v = stream_is_capturing_fn_t{nullptr};
    return _v;
}

// The map accessors below can return nullptr. Each caches a reference to the
// static_object's pointer, which destroy_static_objects() nulls at teardown, and nothing
// unwraps our HIP dispatch table entries. After rocprofiler_register_detach the process
// keeps running with the wrappers still installed, so an application hipEventRecord can
// reach these accessors once the maps are gone. Every dereference is guarded.
auto*
get_event_info_map()
{
    static auto*& _v = common::static_object<common::Synchronized<event_info_map_t>>::construct();
    return _v;
}

auto*
get_coalesce_group_map()
{
    static auto*& _v =
        common::static_object<common::Synchronized<coalesce_group_map_t>>::construct();
    return _v;
}

auto*
get_pending_wait_map()
{
    static auto*& _v = common::static_object<common::Synchronized<pending_wait_map_t>>::construct();
    return _v;
}

std::atomic<uint32_t> g_pending_wait_count{0};

// Process-global fast-path gate: false until a tool configures a HIP_EVENT service. Lets
// is_active() (called from WriteInterceptor on every packet batch) skip the active-context
// walk entirely when HIP event tracing is never used.
std::atomic<bool>&
hip_event_service_configured_flag()
{
    static auto*& _v = common::static_object<std::atomic<bool>>::construct(false);
    return *_v;
}

bool
context_has_hip_event(const tracing::context_t* ctx)
{
    return (CHECK_NOTNULL(ctx) &&
            ((ctx->callback_tracer &&
              ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT)) ||
             (ctx->buffered_tracer &&
              ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_HIP_EVENT))));
}

void
register_pending_wait(uint64_t signal_handle, pending_wait_t pw)
{
    auto* map_v = get_pending_wait_map();
    if(!map_v) return;

    map_v->wlock([&](auto& map) {
        map.emplace(signal_handle, std::move(pw));
        g_pending_wait_count.fetch_add(1, std::memory_order_release);
    });
}

// Consumes one wait registered against signal_handle, skipping any registered by the
// hipStreamWaitEvent on event exclude_hip_event_handle (no exclusion when it is 0).
pending_wait_t
consume_pending_wait(uint64_t signal_handle, uint64_t exclude_hip_event_handle)
{
    auto  result = pending_wait_t{};
    auto* map_v  = get_pending_wait_map();
    if(!map_v) return result;

    map_v->wlock([&](auto& map) {
        auto [begin, end] = map.equal_range(signal_handle);
        for(auto it = begin; it != end; ++it)
        {
            if(exclude_hip_event_handle != 0 &&
               it->second.hip_event_handle == exclude_hip_event_handle)
                continue;

            result = std::move(it->second);
            map.erase(it);
            g_pending_wait_count.fetch_sub(1, std::memory_order_release);
            break;
        }
    });
    return result;
}

pending_wait_t
consume_pending_wait_for_handle(uint64_t signal_handle, uint64_t hip_event_handle)
{
    auto  result = pending_wait_t{};
    auto* map_v  = get_pending_wait_map();
    if(!map_v) return result;

    map_v->wlock([&](auto& map) {
        auto [begin, end] = map.equal_range(signal_handle);
        for(auto it = begin; it != end; ++it)
        {
            if(it->second.hip_event_handle == hip_event_handle)
            {
                result = std::move(it->second);
                map.erase(it);
                g_pending_wait_count.fetch_sub(1, std::memory_order_release);
                break;
            }
        }
    });
    return result;
}

bool
is_stream_capturing(hipStream_t stream)
{
    if(!get_stream_is_capturing_fn()) return false;

    // Pass the caller's stream through unmodified. The saved function is the non-_spt
    // hipStreamIsCapturing, whose common path already reports CaptureStatusNone for
    // nullptr and hipStreamLegacy. Remapping them to hipStreamPerThread here would be
    // doing the _spt variant's PER_THREAD_DEFAULT_STREAM substitution on behalf of an
    // entry point that does not perform it, and would answer for a different stream.
    auto status = hipStreamCaptureStatusNone;
    auto err    = get_stream_is_capturing_fn()(stream, &status);
    return (err == hipSuccess && status == hipStreamCaptureStatusActive);
}

void
check_coalesced_record(uint64_t hip_event_handle, hipStream_t stream)
{
    if(g_active_event_ctx.barrier_captured) return;

    auto group = lookup_coalesce_group(hip_event_handle);
    if(!group) return;

    auto hip_event_tracing_data = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                               ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                               hip_event_tracing_data);
    if(hip_event_tracing_data.empty()) return;

    // Deferred until every cheaper rejection above has passed: this calls back into the HIP
    // runtime, and it is only needed once a record is actually going to be emitted. Placed
    // before the correlation id is retained below so a captured stream takes no references.
    if(is_stream_capturing(stream)) return;

    auto*                    corr_id      = context::get_latest_correlation_id();
    context::correlation_id* corr_id_self = nullptr;
    if(!corr_id)
    {
        corr_id      = context::correlation_tracing_service::construct(1);
        corr_id_self = corr_id;
    }
    if(!corr_id) return;

    auto _corr_cleanup = common::scope_destructor{[corr_id_self]() {
        if(corr_id_self)
        {
            context::pop_latest_correlation_id(corr_id_self);
            corr_id_self->sub_ref_count();
        }
    }};

    auto thr_id = corr_id->thread_idx;
    tracing::populate_external_correlation_ids(hip_event_tracing_data.external_correlation_ids,
                                               thr_id,
                                               ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT,
                                               ROCPROFILER_HIP_EVENT_RECORD,
                                               corr_id->internal);

    auto event_info = lookup_event_info(hip_event_handle);

    auto pending         = coalesce_pending_t{};
    pending.tracing_data = std::move(hip_event_tracing_data);
    pending.callback_record =
        common::init_public_api_struct(rocprofiler_callback_tracing_hip_event_data_t{},
                                       rocprofiler_timestamp_t{0},
                                       rocprofiler_timestamp_t{0},
                                       event_info.agent_id,
                                       event_info.queue_id,
                                       hip_event_handle,
                                       event_info.queue_id);
    pending.tid              = thr_id;
    pending.internal_corr_id = corr_id->internal;
    pending.ancestor_corr_id = corr_id->ancestor;
    pending.corr_id_ref      = corr_id;
    corr_id->add_ref_count();
    corr_id->add_kern_count();

    auto already_completed = false;
    auto completed_time    = profiling_time{};

    group->wlock([&](auto& grp) {
        if(grp.completed)
        {
            already_completed = true;
            completed_time    = grp.barrier_time;
        }
        else
        {
            grp.pending.emplace_back(std::move(pending));
        }
    });

    if(already_completed)
    {
        barrier_complete(pending.tracing_data,
                         pending.tid,
                         pending.internal_corr_id,
                         pending.ancestor_corr_id,
                         completed_time,
                         ROCPROFILER_HIP_EVENT_RECORD,
                         pending.callback_record);
        pending.corr_id_ref->sub_kern_count();
        pending.corr_id_ref->sub_ref_count();
    }
}

using event_record_fn_t            = hipError_t (*)(hipEvent_t, hipStream_t);
using event_record_with_flags_fn_t = hipError_t (*)(hipEvent_t, hipStream_t, unsigned int);
using stream_wait_event_fn_t       = hipError_t (*)(hipStream_t, hipEvent_t, unsigned int);
using event_destroy_fn_t           = hipError_t (*)(hipEvent_t);

struct saved_table_t
{
    event_record_fn_t            hipEventRecord_fn          = nullptr;
    event_record_fn_t            hipEventRecord_spt_fn      = nullptr;
    event_record_with_flags_fn_t hipEventRecordWithFlags_fn = nullptr;
    stream_wait_event_fn_t       hipStreamWaitEvent_fn      = nullptr;
    stream_wait_event_fn_t       hipStreamWaitEvent_spt_fn  = nullptr;
    event_destroy_fn_t           hipEventDestroy_fn         = nullptr;
};

saved_table_t&
get_saved_table()
{
    static auto _v = saved_table_t{};
    return _v;
}

template <event_record_fn_t saved_table_t::*SavedField>
hipError_t
event_record_impl(hipEvent_t event, hipStream_t stream)
{
    g_active_event_ctx = {
        ROCPROFILER_HIP_EVENT_RECORD, reinterpret_cast<uint64_t>(event), false, stream};
    auto _cleanup = common::scope_destructor{[]() { g_active_event_ctx = {}; }};
    auto ret      = (get_saved_table().*SavedField)(event, stream);
    if(ret == hipSuccess) check_coalesced_record(reinterpret_cast<uint64_t>(event), stream);
    return ret;
}

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
hipError_t
event_record_with_flags_impl(hipEvent_t event, hipStream_t stream, unsigned int flags)
{
    g_active_event_ctx = {
        ROCPROFILER_HIP_EVENT_RECORD, reinterpret_cast<uint64_t>(event), false, stream};
    auto _cleanup = common::scope_destructor{[]() { g_active_event_ctx = {}; }};
    auto ret      = get_saved_table().hipEventRecordWithFlags_fn(event, stream, flags);
    if(ret == hipSuccess) check_coalesced_record(reinterpret_cast<uint64_t>(event), stream);
    return ret;
}
#endif

void
discard_pending_wait(uint64_t signal_handle, uint64_t hip_event_handle)
{
    // Consume and release the pre-registered pending wait without emitting a record.
    // Called when we determine no GPU-side dependency was actually created (error,
    // graph capture, or a direct standalone barrier was intercepted instead).
    auto pw = consume_pending_wait_for_handle(signal_handle, hip_event_handle);
    if(pw.corr_id_ref)
    {
        pw.corr_id_ref->sub_kern_count();
        pw.corr_id_ref->sub_ref_count();
    }
}

// Returns the signal handle of the registered entry, or 0 if nothing was registered.
uint64_t
register_deferred_wait(uint64_t hip_event_handle, hipStream_t wait_stream)
{
    auto event_info = lookup_event_info(hip_event_handle);
    if(event_info.original_signal == 0) return 0;

    // CLR returns early from Event::streamWait when the wait targets the stream the event
    // was recorded on, so no GPU dependency is created and nothing will ever claim an
    // entry registered here. Registering one anyway strands it until the event is
    // re-recorded or destroyed, retaining a correlation id that finalization reports as
    // dangling.
    //
    // Compares the caller-supplied stream values as-is, deliberately without collapsing
    // nullptr onto a default-stream sentinel. nullptr denotes the per-thread default
    // stream through the _spt entry points and the legacy default stream otherwise, and
    // those are different streams; treating them as interchangeable could suppress a wait
    // that really does cross streams, losing a record. Failing open costs at most a
    // stranded entry, which is bounded and recovered at re-record or destroy, so the
    // comparison only suppresses when the two are unambiguously the same stream.
    //
    // Also narrower than CLR, which compares event_->command().queue(): two distinct
    // streams backed by one queue are not caught here.
    if(event_info.record_stream == wait_stream) return 0;

    // CLR skips completed signals when it assembles a barrier's dependency list, so a
    // wait issued after the record barrier finished produces no GPU work to trace.
    // Completion is monotonic within a record generation, so refusing here can only
    // suppress waits CLR would have dropped as well. An event that completes after this
    // check leaves an entry that record_event_info or erase_event_info releases.
    if(event_info.completed) return 0;

    auto tracing_data = tracing::tracing_data{};
    tracing::populate_contexts(
        ROCPROFILER_CALLBACK_TRACING_HIP_EVENT, ROCPROFILER_BUFFER_TRACING_HIP_EVENT, tracing_data);
    if(tracing_data.empty()) return 0;

    auto*                    corr_id      = context::get_latest_correlation_id();
    context::correlation_id* corr_id_self = nullptr;
    if(!corr_id)
    {
        corr_id      = context::correlation_tracing_service::construct(1);
        corr_id_self = corr_id;
    }
    if(!corr_id) return 0;

    auto _corr_cleanup = common::scope_destructor{[corr_id_self]() {
        if(corr_id_self)
        {
            context::pop_latest_correlation_id(corr_id_self);
            corr_id_self->sub_ref_count();
        }
    }};

    auto thr_id = corr_id->thread_idx;
    tracing::populate_external_correlation_ids(tracing_data.external_correlation_ids,
                                               thr_id,
                                               ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT,
                                               ROCPROFILER_HIP_EVENT_WAIT,
                                               corr_id->internal);

    auto pw             = pending_wait_t{};
    pw.tracing_data     = std::move(tracing_data);
    pw.tid              = thr_id;
    pw.internal_corr_id = corr_id->internal;
    pw.ancestor_corr_id = corr_id->ancestor;
    pw.corr_id_ref      = corr_id;
    pw.hip_event_handle = hip_event_handle;
    pw.source_info      = event_info;
    corr_id->add_ref_count();
    corr_id->add_kern_count();

    register_pending_wait(event_info.original_signal, std::move(pw));
    return event_info.original_signal;
}

template <stream_wait_event_fn_t saved_table_t::*SavedField>
hipError_t
stream_wait_event_impl(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    const auto hip_event_handle = reinterpret_cast<uint64_t>(event);

    g_active_event_ctx = {ROCPROFILER_HIP_EVENT_WAIT, hip_event_handle, false};
    auto _cleanup      = common::scope_destructor{[]() { g_active_event_ctx = {}; }};

    // Pre-register the pending wait BEFORE calling CLR. This ensures the entry
    // is in the map before CLR can enqueue any barrier carrying the dep_signal,
    // even if another thread concurrently submits work to the waiting stream.
    const auto registered_signal = register_deferred_wait(hip_event_handle, stream);

    auto ret = (get_saved_table().*SavedField)(stream, event, flags);

    // Discard the pre-registered entry in cases where no GPU-side dependency was created:
    // - Error return: CLR did nothing.
    // - Graph capture: stream is being captured, no real barrier submitted.
    // - barrier_captured: WriteInterceptor intercepted a standalone WAIT barrier and
    //   already emitted (or will emit) a record via BarrierAsyncSignalHandler.
    if(registered_signal != 0)
    {
        if(ret != hipSuccess || is_stream_capturing(stream) || g_active_event_ctx.barrier_captured)
            discard_pending_wait(registered_signal, hip_event_handle);
    }

    return ret;
}

hipError_t
event_destroy_impl(hipEvent_t event)
{
    auto ret = get_saved_table().hipEventDestroy_fn(event);
    if(ret == hipSuccess) erase_event_info(reinterpret_cast<uint64_t>(event));
    return ret;
}
}  // namespace

active_event_context_t*
get_active_event_context()
{
    if(g_active_event_ctx.operation == ROCPROFILER_HIP_EVENT_NONE) return nullptr;
    return &g_active_event_ctx;
}

void
record_event_info(uint64_t hip_event_handle, event_record_info_t info)
{
    auto* info_map = get_event_info_map();
    if(!info_map) return;

    auto old_signal = uint64_t{0};
    info_map->wlock([&](auto& map) {
        // Single lookup: operator[] default-constructs a fresh entry when the event has no
        // prior generation, and original_signal is zero there, which is the same value the
        // previous find()-then-insert produced for a miss.
        auto& slot = map[hip_event_handle];
        old_signal = slot.original_signal;
        slot       = info;
    });

    // A new record supersedes the previous generation. Completion signals are pooled and
    // the same handle can be handed back for the new record, so a matching handle is not
    // evidence that outstanding waits belong to the current generation. Those waits can
    // no longer be attributed to a generation and are released without emitting a record.
    // The drain is scoped by event handle so that another event holding a recycled signal
    // keeps its own waits.
    if(old_signal != 0)
    {
        for(;;)
        {
            auto pw = consume_pending_wait_for_handle(old_signal, hip_event_handle);
            if(!pw.corr_id_ref) break;
            pw.corr_id_ref->sub_kern_count();
            pw.corr_id_ref->sub_ref_count();
        }
    }
}

void
mark_event_completed(uint64_t hip_event_handle)
{
    auto* map_v = get_event_info_map();
    if(!map_v) return;

    map_v->wlock([&](auto& map) {
        auto it = map.find(hip_event_handle);
        if(it != map.end()) it->second.completed = true;
    });
}

event_record_info_t
lookup_event_info(uint64_t hip_event_handle)
{
    auto* map_v = get_event_info_map();
    if(!map_v) return event_record_info_t{};

    return map_v->rlock([&](const auto& map) -> event_record_info_t {
        auto it = map.find(hip_event_handle);
        if(it != map.end()) return it->second;
        return event_record_info_t{};
    });
}

void
store_coalesce_group(uint64_t hip_event_handle, coalesce_group_ptr_t group)
{
    auto* map_v = get_coalesce_group_map();
    if(!map_v) return;

    map_v->wlock([&](auto& map) { map[hip_event_handle] = std::move(group); });
}

void
erase_event_info(uint64_t hip_event_handle)
{
    // Release every pending wait for this event, including any left under the signal of
    // an earlier record generation. Matching on the event handle rather than the signal
    // keeps waits owned by another event that holds a recycled signal handle intact.
    //
    // Correlation ids are collected under the lock and released after it is dropped.
    // sub_ref_count can retire the id, which emplaces a retirement record; a lossless
    // buffer or one at its watermark then flushes and invokes the tool's callback. A tool
    // making a wrapped HIP call from that callback re-enters this map, so holding the
    // write lock across the decrement would deadlock against ourselves. Every other site
    // that releases a pending wait already drops the lock first.
    auto released = common::container::small_vector<context::correlation_id*, 4>{};

    // The scan below is linear in the whole map, so destroying M events would otherwise be
    // O(N*M). Nothing can be registered for this event when the global count is zero, and
    // the count is only ever raised while a wait is registered, so skipping is exact.
    auto* wait_map = has_pending_waits() ? get_pending_wait_map() : nullptr;
    if(wait_map)
        wait_map->wlock([&](auto& map) {
            for(auto it = map.begin(); it != map.end();)
            {
                if(it->second.hip_event_handle == hip_event_handle)
                {
                    if(it->second.corr_id_ref) released.emplace_back(it->second.corr_id_ref);
                    it = map.erase(it);
                    g_pending_wait_count.fetch_sub(1, std::memory_order_release);
                }
                else
                {
                    ++it;
                }
            }
        });

    for(auto* corr_id : released)
    {
        corr_id->sub_kern_count();
        corr_id->sub_ref_count();
    }

    if(auto* info_map = get_event_info_map())
        info_map->wlock([&](auto& map) { map.erase(hip_event_handle); });
    if(auto* group_map = get_coalesce_group_map())
        group_map->wlock([&](auto& map) { map.erase(hip_event_handle); });
}

coalesce_group_ptr_t
lookup_coalesce_group(uint64_t hip_event_handle)
{
    auto* map_v = get_coalesce_group_map();
    if(!map_v) return nullptr;

    return map_v->rlock([&](const auto& map) -> coalesce_group_ptr_t {
        auto it = map.find(hip_event_handle);
        if(it != map.end()) return it->second;
        return nullptr;
    });
}

bool
has_pending_waits()
{
    return g_pending_wait_count.load(std::memory_order_acquire) > 0;
}

void
set_service_configured(bool enabled)
{
    // Skip during finalization: the flag is a static_object that may already be destroyed.
    if(registration::get_fini_status() > 0) return;
    hip_event_service_configured_flag().store(enabled, std::memory_order_relaxed);
}

bool
is_active()
{
    // Skip during finalization: the flag and the context registry are static_objects that may be
    // destroyed by then, and WriteInterceptor can still call this from HIP/HSA teardown.
    if(registration::get_fini_status() > 0) return false;
    // Cheap common-case rejection: if no HIP_EVENT service was ever configured, skip the walk.
    if(!hip_event_service_configured_flag().load(std::memory_order_relaxed)) return false;
    // Stopped contexts take the fast path too: with none started there is nothing to record for,
    // so the interceptor should bail rather than build a snapshot it will find empty later.
    if(context::get_active_contexts(context_has_hip_event).empty()) return false;

    return get_active_event_context() != nullptr || has_pending_waits();
}

namespace
{
// One event reported by an intercepted barrier: either the record or wait whose API call
// produced the barrier, or a deferred wait whose dependency CLR folded into it. Each carries
// the correlation of its own API call.
struct barrier_data_t
{
    tracing::tracing_data                         tracing_data     = {};
    rocprofiler_callback_tracing_hip_event_data_t callback_record  = {};
    rocprofiler_hip_event_operation_t             operation        = ROCPROFILER_HIP_EVENT_NONE;
    coalesce_group_ptr_t                          coalesce_group   = {};
    rocprofiler_thread_id_t                       tid              = 0;
    uint64_t                                      internal_corr_id = 0;
    uint64_t                                      ancestor_corr_id = 0;
    context::correlation_id*                      corr_id_ref      = nullptr;
};

using barrier_data_array_t = common::container::small_vector<barrier_data_t, 1>;

// A barrier packet whose completion signal was replaced by a pooled one, carried from the
// packet-write moment to the GPU completion handler below. Every event in barrier_data is
// timed by this one barrier.
struct barrier_info_session_t
{
    hsa::Queue&                                    queue;
    uint32_t                                       packet_type       = HSA_PACKET_TYPE_BARRIER_AND;
    rocprofiler_timestamp_t                        enqueue_ts        = 0;
    hsa_signal_t                                   completion_signal = {.handle = 0};
    common::container::pool_object<hsa::signal_t>* pooled_signal     = nullptr;
    // One inline slot: a barrier usually reports a single event, and each slot is large.
    barrier_data_array_t barrier_data = {};
};

// plan_barrier writes the pooled signal through the barrier-AND view of the packet for both
// barrier kinds, which relies on the two layouts placing completion_signal identically.
static_assert(offsetof(hsa_barrier_and_packet_t, completion_signal) ==
                  offsetof(hsa_amd_barrier_value_packet_t, completion_signal),
              "barrier-AND and barrier-value completion_signal offsets differ");

// GPU start and end of the barrier, in the clock domain of kernel dispatch records. When the CP
// did not write the timestamps, the returned status is not HSA_STATUS_SUCCESS and the events
// timed by this barrier are not reported, as kernel dispatch and memory copy records are not.
profiling_time
get_barrier_time(const barrier_info_session_t& session, rocprofiler_timestamp_t handler_ts)
{
    static const auto strict_ts = common::get_env(
        "ROCPROFILER_CI_STRICT_TIMESTAMPS", (ROCPROFILER_CI_STRICT_TIMESTAMPS > 0) ? true : false);

    const auto& agent    = session.queue.get_agent();
    auto        gpu_time = hsa_amd_profiling_dispatch_time_t{};
    auto        status   = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        agent.get_hsa_agent(), session.completion_signal, &gpu_time);

    // Rejected before adjust_profiling_time, which would clamp an unwritten 0/0 into the host
    // interval and pass it off as valid. ROCr's tick translation reports success with 0 for a
    // timestamp the CP never wrote, and start_barrier_session clears both before the packet is
    // submitted. A start after the end is left to adjust_profiling_time, which swaps it (or fails
    // under ROCPROFILER_CI_STRICT_TIMESTAMPS) exactly as it does for kernel dispatch times.
    if(status == HSA_STATUS_SUCCESS && gpu_time.start != 0 && gpu_time.end != 0)
    {
        return tracing::adjust_profiling_time(
            "hip_event",
            "hsa_amd_profiling_get_dispatch_time",
            profiling_time{status, gpu_time.start, gpu_time.end},
            profiling_time{HSA_STATUS_SUCCESS, session.enqueue_ts, handler_ts});
    }

    const auto* rocp_agent = CHECK_NOTNULL(agent.get_rocp_agent());
    auto        msg        = fmt::format(
        "hsa_amd_profiling_get_dispatch_time returned no valid GPU timestamps for a HIP event "
        "barrier (packet type {}) on agent-{}: status={}, start={}, end={}",
        session.packet_type,
        rocp_agent->node_id,
        static_cast<int>(status),
        gpu_time.start,
        gpu_time.end);

    ROCP_FATAL_IF(strict_ts) << msg;
    ROCP_CI_LOG(ERROR) << msg << ". The HIP events timed by this barrier are not reported";

    return profiling_time{(status == HSA_STATUS_SUCCESS) ? HSA_STATUS_ERROR : status, 0, 0};
}

// Event barriers get their own session type and completion handler rather than riding on
// hsa::queue_info_session_t and AsyncSignalHandler. Three things prevent folding them in:
//
//  - Registration point. The queue session registers one handler per batch, on the last
//    dispatch's completion signal, and only when packet_data is non-empty. A
//    hipEventRecord routinely produces a batch with no dispatch packets at all, so no
//    handler would ever be registered for it. Even alongside kernels, an event must be
//    timestamped by its own barrier, not by some later kernel.
//
//  - Timing source. kernel_dispatch::get_dispatch_time takes its signal and agent from the
//    session's packet_data_t (a kernel packet) and names a kernel id in its failure log. An
//    event is timed by its barrier's completion signal and has no kernel to name.
//
//  - Record shape. packet_data_t::callback_record is typed
//    rocprofiler_callback_tracing_kernel_dispatch_data_t. Event records carry different
//    fields (hip_event_handle, source_queue_id), so sharing the struct would mean
//    populating a kernel dispatch record that describes no kernel.
bool
BarrierAsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    using session_info_t = std::shared_ptr<barrier_info_session_t>;

    const auto handler_ts = common::timestamp_ns();

    ROCP_FATAL_IF(!data) << "BarrierAsyncSignalHandler called with null data pointer";

    auto* _session_ptr = static_cast<session_info_t*>(data);

    if(registration::get_fini_status() > 0)
    {
        _session_ptr->reset();
        delete _session_ptr;
        return false;
    }

    auto _cleanup = common::scope_destructor{[&_session_ptr]() {
        _session_ptr->reset();
        delete _session_ptr;
        _session_ptr = nullptr;
    }};

    auto _session = *_session_ptr;
    ROCP_FATAL_IF(!_session.get()) << "nullptr to barrier session information";

    auto& barrier_session = *_session;

    // Read before the signal goes back to the pool below, which lets another barrier reuse it.
    const auto barrier_time = get_barrier_time(barrier_session, handler_ts);

    for(auto& bdata : barrier_session.barrier_data)
    {
        // Mark ahead of the callbacks below so that tool callback duration does not widen
        // the window in which a wait can still be registered against a completed event.
        if(bdata.operation == ROCPROFILER_HIP_EVENT_RECORD)
            mark_event_completed(bdata.callback_record.hip_event_handle);

        barrier_complete(bdata.tracing_data,
                         bdata.tid,
                         bdata.internal_corr_id,
                         bdata.ancestor_corr_id,
                         barrier_time,
                         bdata.operation,
                         bdata.callback_record);

        // Records CLR coalesced onto this barrier mark the same stream position, so they
        // report its times, even when their own API call came after the barrier retired.
        if(bdata.operation == ROCPROFILER_HIP_EVENT_RECORD && bdata.coalesce_group)
        {
            auto pending_entries = common::container::small_vector<coalesce_pending_t, 4>{};

            bdata.coalesce_group->wlock([&](auto& grp) {
                grp.barrier_time = barrier_time;
                grp.completed    = true;
                pending_entries  = std::move(grp.pending);
                grp.pending.clear();
            });

            for(auto& p : pending_entries)
            {
                barrier_complete(p.tracing_data,
                                 p.tid,
                                 p.internal_corr_id,
                                 p.ancestor_corr_id,
                                 barrier_time,
                                 ROCPROFILER_HIP_EVENT_RECORD,
                                 p.callback_record);
                if(p.corr_id_ref)
                {
                    p.corr_id_ref->sub_kern_count();
                    p.corr_id_ref->sub_ref_count();
                }
            }
        }

        if(bdata.corr_id_ref)
        {
            bdata.corr_id_ref->sub_kern_count();
            bdata.corr_id_ref->sub_ref_count();
        }
    }

    if(barrier_session.pooled_signal) hsa::Queue::release_signal(barrier_session.pooled_signal);

    barrier_session.queue.async_complete();

    return false;
}

// Claims a deferred wait registered against each dependency signal of a barrier packet.
// own_wait_handle, when non-zero, is a hipStreamWaitEvent this thread is executing whose
// barrier was captured directly: it reports through that capture and stream_wait_event_impl
// discards its registration, so it must not be claimed here as well.
pending_wait_array_t
claim_deferred_waits(const hsa::rocprofiler_packet& pkt,
                     uint32_t                       packet_type,
                     uint64_t                       own_wait_handle)
{
    auto claimed = pending_wait_array_t{};

    // Cheap rejection before taking the pending-wait lock. This runs on every barrier packet
    // whenever any queue-intercepting service is active, so the common case of "no deferred
    // waits outstanding" must not cost a lock acquisition. A wait is always registered before
    // the CLR call that creates the GPU dependency, and CLR serializes stream submission, so a
    // dependency visible here implies the increment is already visible.
    if(!has_pending_waits()) return claimed;

    auto claim = [&](hsa_signal_t dep) {
        if(dep.handle == 0) return;
        auto pw = consume_pending_wait(dep.handle, own_wait_handle);
        if(pw.corr_id_ref) claimed.emplace_back(std::move(pw));
    };

    if(packet_type == HSA_PACKET_TYPE_BARRIER_AND)
    {
        for(const auto& dep : pkt.barrier_and.dep_signal)
            claim(dep);
    }
    else
    {
        claim(reinterpret_cast<const hsa_amd_barrier_value_packet_t&>(pkt).signal);
    }

    return claimed;
}

barrier_data_t
make_deferred_wait_data(const hsa::Queue& queue, pending_wait_t&& pw)
{
    auto bdata         = barrier_data_t{};
    bdata.tracing_data = std::move(pw.tracing_data);
    bdata.callback_record =
        common::init_public_api_struct(rocprofiler_callback_tracing_hip_event_data_t{},
                                       rocprofiler_timestamp_t{0},
                                       rocprofiler_timestamp_t{0},
                                       queue.get_agent().get_rocp_agent()->id,
                                       queue.get_id(),
                                       pw.hip_event_handle,
                                       pw.source_info.queue_id);
    bdata.operation        = ROCPROFILER_HIP_EVENT_WAIT;
    bdata.tid              = pw.tid;
    bdata.internal_corr_id = pw.internal_corr_id;
    bdata.ancestor_corr_id = pw.ancestor_corr_id;
    bdata.corr_id_ref      = pw.corr_id_ref;
    return bdata;
}

// Builds the record for the hipEventRecord or hipStreamWaitEvent executing on this thread,
// whose barrier pkt is, and fires its ENTER and EXIT callbacks.
barrier_data_t
capture_event_barrier(hsa::Queue&                    queue,
                      const hsa::rocprofiler_packet& pkt,
                      active_event_context_t&        event_ctx,
                      const tracing::tracing_data&   tracing_data_v,
                      context::correlation_id*       corr_id)
{
    const auto thr_id           = corr_id->thread_idx;
    const auto internal_corr_id = corr_id->internal;
    const auto ancestor_corr_id = corr_id->ancestor;

    corr_id->add_ref_count();
    corr_id->add_kern_count();

    auto _barrier_data             = barrier_data_t{};
    _barrier_data.tracing_data     = tracing_data_v;
    _barrier_data.operation        = event_ctx.operation;
    _barrier_data.tid              = thr_id;
    _barrier_data.internal_corr_id = internal_corr_id;
    _barrier_data.ancestor_corr_id = ancestor_corr_id;
    _barrier_data.corr_id_ref      = corr_id;

    tracing::populate_external_correlation_ids(_barrier_data.tracing_data.external_correlation_ids,
                                               thr_id,
                                               ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT,
                                               event_ctx.operation,
                                               internal_corr_id);

    auto source_queue = queue.get_id();
    if(event_ctx.operation == ROCPROFILER_HIP_EVENT_RECORD)
    {
        record_event_info(event_ctx.hip_event_handle,
                          event_record_info_t{queue.get_id(),
                                              queue.get_agent().get_rocp_agent()->id,
                                              pkt.barrier_and.completion_signal.handle,
                                              event_ctx.record_stream});
    }
    else if(event_ctx.operation == ROCPROFILER_HIP_EVENT_WAIT)
    {
        source_queue = lookup_event_info(event_ctx.hip_event_handle).queue_id;
    }

    _barrier_data.callback_record =
        common::init_public_api_struct(rocprofiler_callback_tracing_hip_event_data_t{},
                                       rocprofiler_timestamp_t{0},
                                       rocprofiler_timestamp_t{0},
                                       queue.get_agent().get_rocp_agent()->id,
                                       queue.get_id(),
                                       event_ctx.hip_event_handle,
                                       source_queue);

    // ENTER/EXIT bracket the barrier enqueue on the CPU and carry no timestamps. The
    // PHASE_NONE callback with the GPU times fires later from BarrierAsyncSignalHandler.
    {
        auto tracer_data = _barrier_data.callback_record;
        tracing::execute_phase_enter_callbacks(_barrier_data.tracing_data.callback_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               _barrier_data.tracing_data.external_correlation_ids,
                                               ancestor_corr_id,
                                               ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                               event_ctx.operation,
                                               tracer_data);
    }

    tracing::update_external_correlation_ids(_barrier_data.tracing_data.external_correlation_ids,
                                             thr_id,
                                             ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT);

    {
        auto tracer_data = _barrier_data.callback_record;
        tracing::execute_phase_exit_callbacks(_barrier_data.tracing_data.callback_contexts,
                                              _barrier_data.tracing_data.external_correlation_ids,
                                              ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                              event_ctx.operation,
                                              tracer_data);
    }

    event_ctx.barrier_captured = true;

    if(event_ctx.operation == ROCPROFILER_HIP_EVENT_RECORD)
    {
        auto group = std::make_shared<common::Synchronized<coalesce_group_t>>();
        store_coalesce_group(event_ctx.hip_event_handle, group);
        _barrier_data.coalesce_group = group;
    }

    return _barrier_data;
}

// Gives the barrier a pooled completion signal so the CP stamps its execution, and registers
// the handler that reports every event in barrier_data once it retires. Returns the packet
// to submit in place of pkt.
hsa::rocprofiler_packet
start_barrier_session(hsa::Queue&                    queue,
                      const hsa::rocprofiler_packet& pkt,
                      uint32_t                       packet_type,
                      barrier_data_array_t&&         barrier_data)
{
    auto barrier = pkt;
    auto session = std::make_shared<barrier_info_session_t>(
        barrier_info_session_t{.queue = queue, .packet_type = packet_type});

    session->pooled_signal = queue.create_signal(0, &barrier.barrier_and.completion_signal, true);
    // Read the signal from the packet rather than through pooled_signal: Queue::create_signal
    // returns a null pool object whenever the pool is unavailable, while still writing a
    // usable raw signal to the out parameter.
    session->completion_signal = barrier.barrier_and.completion_signal;
    // The barrier decrements this to -1, the value the handler is registered for.
    hsa::get_core_table()->hsa_signal_store_screlease_fn(session->completion_signal, 0);
    // A reused pool signal keeps the previous submission's timestamps. Clear them so that a
    // barrier the CP does not stamp reads back 0, which get_barrier_time rejects.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* _amd_signal     = reinterpret_cast<amd_signal_t*>(session->completion_signal.handle);
    _amd_signal->start_ts = 0;
    _amd_signal->end_ts   = 0;

    session->barrier_data = std::move(barrier_data);
    // The packet is written to the queue after this returns, so the CP cannot stamp it earlier.
    session->enqueue_ts = common::timestamp_ns();

    const auto raw_signal = session->completion_signal;

    queue.async_started();

    auto status = hsa::get_amd_ext_table()->hsa_amd_signal_async_handler_fn(
        raw_signal,
        HSA_SIGNAL_CONDITION_EQ,
        -1,
        BarrierAsyncSignalHandler,
        new std::shared_ptr<barrier_info_session_t>(std::move(session)));

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) << fmt::format(
        "Error: hsa_amd_signal_async_handler for barrier (signal={{.handle={}}}) failed with "
        "error code {}",
        raw_signal.handle,
        static_cast<int>(status));

    return barrier;
}
}  // namespace

barrier_plan_t
plan_barrier(hsa::Queue&                    queue,
             const hsa::rocprofiler_packet& pkt,
             uint32_t                       packet_type,
             bool                           is_barrier,
             const tracing::tracing_data&   tracing_data_v,
             context::correlation_id*       corr_id)
{
    auto plan = barrier_plan_t{};

    if(!is_barrier) return plan;

    auto* event_ctx = get_active_event_context();

    const auto original_completion_signal = pkt.barrier_and.completion_signal;
    const bool existing_completion_signal = (original_completion_signal.handle != 0);

    const bool capture = (event_ctx && !event_ctx->barrier_captured && existing_completion_signal &&
                          !tracing_data_v.empty());

    const auto own_wait_handle = (event_ctx && event_ctx->operation == ROCPROFILER_HIP_EVENT_WAIT &&
                                  (capture || event_ctx->barrier_captured))
                                     ? event_ctx->hip_event_handle
                                     : uint64_t{0};

    // Claimed before the capture below: a hipEventRecord drops the waits still registered
    // against its previous generation, and a wait folded into this very record barrier is one
    // of them even though it is satisfied here.
    auto claimed = claim_deferred_waits(pkt, packet_type, own_wait_handle);

    if(!capture && claimed.empty()) return plan;

    auto barrier_data = barrier_data_array_t{};
    if(capture)
        barrier_data.emplace_back(
            capture_event_barrier(queue, pkt, *event_ctx, tracing_data_v, corr_id));
    for(auto& pw : claimed)
        barrier_data.emplace_back(make_deferred_wait_data(queue, std::move(pw)));

    plan.intercepted = true;
    plan.barrier     = start_barrier_session(queue, pkt, packet_type, std::move(barrier_data));

    // The original signal still has to fire, and only after this barrier retires.
    if(existing_completion_signal)
    {
        auto forwarding   = hsa_barrier_and_packet_t{};
        forwarding.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        forwarding.header |= (1 << HSA_PACKET_HEADER_BARRIER);
        forwarding.completion_signal = original_completion_signal;

        plan.has_forwarding = true;
        plan.forwarding     = hsa::rocprofiler_packet{forwarding};
    }

    return plan;
}

std::optional<hsa::rocprofiler_packet>
plan_dependency_barrier(hsa::Queue& queue, hsa_signal_t dep_signal)
{
    if(dep_signal.handle == 0) return std::nullopt;

    // The form CLR emits ahead of a kernel when the dispatch packet cannot carry the
    // dependency itself (kNopPacketHeader in VirtualGPU::dispatchBlockingWait).
    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= (1 << HSA_PACKET_HEADER_BARRIER);
    barrier.dep_signal[0] = dep_signal;

    const auto pkt     = hsa::rocprofiler_packet{barrier};
    auto       claimed = claim_deferred_waits(pkt, HSA_PACKET_TYPE_BARRIER_AND, 0);
    if(claimed.empty()) return std::nullopt;

    auto barrier_data = barrier_data_array_t{};
    for(auto& pw : claimed)
        barrier_data.emplace_back(make_deferred_wait_data(queue, std::move(pw)));

    return start_barrier_session(queue, pkt, HSA_PACKET_TYPE_BARRIER_AND, std::move(barrier_data));
}

namespace
{
template <auto AbiOffset>
bool
table_has_entry(const ::HipDispatchTable* table)
{
    return common::abi::compute_table_offset(static_cast<size_t>(AbiOffset)) < table->size;
}
}  // namespace

template <>
void
update_table<::HipDispatchTable>(::HipDispatchTable* table)
{
    if(table == nullptr) return;

    auto& saved = get_saved_table();

    // Access table fields only after confirming they exist within the runtime table's
    // allocated size. The member pointer is dereferenced inside the lambda body, after
    // the size guard, to avoid UB when loading an older HIP runtime with a shorter table.
    auto wrap = [&](auto table_fn_ptr, auto saved_fn_ptr, auto wrapper, auto abi_offset) {
        if(common::abi::compute_table_offset(abi_offset) >= table->size) return;
        auto& table_fn = table->*table_fn_ptr;
        auto& saved_fn = saved.*saved_fn_ptr;
        if(!table_fn || table_fn == wrapper) return;
        ROCP_TRACE << "hip::event wrapping table entry at ABI offset " << abi_offset;
        saved_fn = table_fn;
        // Publish the wrapper only after the saved pointer is visible. late.cpp
        // re-propagates API tables into a running process during attach, so an application
        // thread can be calling through this entry while it is being replaced. The two
        // stores are independent, so without this fence the compiler or a weakly ordered
        // CPU could publish the wrapper first and the thread would call a null saved
        // pointer.
        std::atomic_thread_fence(std::memory_order_release);
        table_fn = wrapper;
    };

    wrap(&::HipDispatchTable::hipEventRecord_fn,
         &saved_table_t::hipEventRecord_fn,
         &event_record_impl<&saved_table_t::hipEventRecord_fn>,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord);

    wrap(&::HipDispatchTable::hipEventRecord_spt_fn,
         &saved_table_t::hipEventRecord_spt_fn,
         &event_record_impl<&saved_table_t::hipEventRecord_spt_fn>,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord_spt);

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
    wrap(&::HipDispatchTable::hipEventRecordWithFlags_fn,
         &saved_table_t::hipEventRecordWithFlags_fn,
         &event_record_with_flags_impl,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecordWithFlags);
#endif

    wrap(&::HipDispatchTable::hipStreamWaitEvent_fn,
         &saved_table_t::hipStreamWaitEvent_fn,
         &stream_wait_event_impl<&saved_table_t::hipStreamWaitEvent_fn>,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent);

    wrap(&::HipDispatchTable::hipStreamWaitEvent_spt_fn,
         &saved_table_t::hipStreamWaitEvent_spt_fn,
         &stream_wait_event_impl<&saved_table_t::hipStreamWaitEvent_spt_fn>,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt);

    wrap(&::HipDispatchTable::hipEventDestroy_fn,
         &saved_table_t::hipEventDestroy_fn,
         &event_destroy_impl,
         ROCPROFILER_HIP_RUNTIME_API_ID_hipEventDestroy);

    if(table_has_entry<ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamIsCapturing>(table) &&
       table->hipStreamIsCapturing_fn)
        get_stream_is_capturing_fn() = table->hipStreamIsCapturing_fn;
}

}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
