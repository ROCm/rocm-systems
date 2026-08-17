// MIT License
//
/* Copyright (c) 2022-2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/common/container/pool.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hip/event.hpp"
#include "lib/rocprofiler-sdk/hip/graph.hpp"
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>

// static assert for rocprofiler_packet ABI compatibility
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_kernel_dispatch_packet_t),
              "unexpected ABI incompatibility");
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_barrier_and_packet_t),
              "unexpected ABI incompatibility");
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_barrier_or_packet_t),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_kernel_dispatch_packet_t, completion_signal),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_barrier_and_packet_t, completion_signal),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_barrier_or_packet_t, completion_signal),
              "unexpected ABI incompatibility");
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_amd_ext_kernel_dispatch_packet_t, completion_signal),
              "unexpected ABI incompatibility");
#endif

namespace rocprofiler
{
namespace hsa
{
namespace
{
constexpr auto null_hsa_signal = hsa_signal_t{.handle = 0};

// A single blocking wait -- Queue::sync(), or the replay drain barrier -- gives up after one slice
// and reports whether it drained. The replay drain helpers retry for at most max_slices slices and
// report elapsed time in multiples of one slice, so every wait in that loop has to use the same
// slice length; hardcoding it separately would make those elapsed figures fiction as soon as one
// of them changed.
constexpr int  seconds_per_slice = 5;
constexpr int  max_slices        = 12;
constexpr auto drain_slice       = std::chrono::seconds{seconds_per_slice};
constexpr int  drain_budget_secs = seconds_per_slice * max_slices;

// Per-agent replay serialization (reader/writer). A replay's snapshot->restore window must exclude
// any concurrent GPU work on the agent that could mutate tracked device memory, but ordinary
// dispatches do not conflict with one another. So this is a shared_mutex:
//   * a replay window takes the UNIQUE (writer) lock for the whole drain->snap->passes->restore
//     sequence, excluding all other replays and every non-replay dispatch on the agent;
//   * a non-replay dispatch takes the SHARED (reader) lock across its submit, so many normal
//     dispatches still run concurrently while a pending replay writer waits for in-flight submits
//     to finish and blocks new ones from entering the window.
// Different agents use different mutexes and run concurrently; combined with agent-scoped snapshots
// this keeps multi-GPU replay isolated. The reader lock only bounds *submission*; the async GPU
// tail is drained by the replay window before it snapshots.
std::shared_mutex&
agent_replay_mutex(rocprofiler_agent_id_t agent_id)
{
    // No get_fini_status() guard is needed here. Every caller reaches this only when
    // has_active_replay_contexts() is true, and that returns false during finalization, so the
    // static lock map below is never touched after teardown.
    using lock_map_t    = std::unordered_map<rocprofiler_agent_id_t, std::shared_mutex>;
    static auto*& locks = common::static_object<common::Synchronized<lock_map_t>>::construct();

    return locks->wlock([](lock_map_t& _map, auto _id) -> std::shared_mutex& { return _map[_id]; },
                        agent_id);
}

template <typename TryDrainFn>
void
replay_wait_or_fatal(TryDrainFn&& try_drain_once, std::string_view what)
{
    // One try_drain_once() call blocks for at most one slice, so the two together bound the wait
    // at drain_budget_secs.
    for(int i = 0; i < max_slices; ++i)
    {
        if(try_drain_once()) return;
        ROCP_WARNING << fmt::format("kernel replay: still waiting for {} (~{}s elapsed)",
                                    what,
                                    (i + 1) * seconds_per_slice);
    }
    ROCP_FATAL << fmt::format(
        "kernel replay: {} did not drain after ~{}s", what, drain_budget_secs);
}

// Drain a queue's in-flight async completion handler(s) during replay. Unlike Queue::sync()'s
// teardown use (warn once and proceed), a replay pass must NOT proceed while a handler is still
// running: PASS-EXIT, the tool's continue-decision, restore(), and the next submit would race the
// handler that is still emitting records, releasing signals, and dropping correlation-id refs.
// Each sync() call blocks up to one slice and reports whether the queue drained.
void
replay_drain_or_fatal(const Queue& queue)
{
    replay_wait_or_fatal([&]() { return queue.sync(); },
                         "this queue's async completion handler(s)");
}

// Drain every queue on `agent` before snapshotting, WITHOUT holding the queue-map lock across the
// wait. iterate_queues holds the queue-map read lock for the duration of its callback, so a
// per-sibling blocking drain there would block stream creation/destruction for the whole drain
// budget. Instead poll each queue's in-flight async count under a brief read lock and sleep
// between polls, so the map lock is held only for the microsecond poll -- never across the wait.
// Safe against concurrent queue destruction: a Queue is only dereferenced while the read lock is
// held (destroy_queue erases under the write lock), and the live set is re-read every poll. The
// per-agent writer lock held by the replay window blocks new dispatches on the agent, so in-flight
// work only decreases and the poll converges; fatal on a genuinely stuck queue (beta feature),
// matching replay_drain_or_fatal.
void
replay_drain_agent_or_fatal(hsa_agent_t agent)
{
    auto* queue_controller = get_queue_controller();
    if(queue_controller == nullptr) return;

    constexpr auto poll_interval = std::chrono::milliseconds{2};
    constexpr auto max_wait      = std::chrono::seconds{drain_budget_secs};
    const auto     deadline      = std::chrono::steady_clock::now() + max_wait;

    for(;;)
    {
        int64_t in_flight = 0;
        queue_controller->iterate_queues([&](const Queue* sibling) {
            if(sibling != nullptr && sibling->get_agent().get_hsa_agent() == agent)
                in_flight += sibling->active_async_packets();
        });

        if(in_flight == 0) return;

        ROCP_FATAL_IF(std::chrono::steady_clock::now() >= deadline) << fmt::format(
            "kernel replay: agent-wide drain stuck ({} async handler(s) still active after ~{}s)",
            in_flight,
            drain_budget_secs);

        std::this_thread::sleep_for(poll_interval);
    }
}

template <typename DomainT, typename... Args>
inline bool
context_filter(const context::context* ctx, DomainT domain, Args... args)
{
    if constexpr(std::is_same<DomainT, rocprofiler_buffer_tracing_kind_t>::value)
    {
        return (ctx->buffered_tracer && ctx->buffered_tracer->domains(domain, args...));
    }
    else if constexpr(std::is_same<DomainT, rocprofiler_callback_tracing_kind_t>::value)
    {
        return (ctx->callback_tracer && ctx->callback_tracer->domains(domain, args...));
    }
    else
    {
        static_assert(common::mpl::assert_false<DomainT>::value, "unsupported domain type");
        return false;
    }
}

bool
full_packet_instrumentation_context_filter(const context::context* ctx)
{
    return (context_filter(ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) ||
            context_filter(ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
}

bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    using session_info_t = std::shared_ptr<queue_info_session_t>;

    if(!data)
    {
        ROCP_FATAL << "AsyncSignalHandler called with null data pointer";
        return true;
    }

    auto* _session_ptr = static_cast<session_info_t*>(data);

    // if we have fully finalized, delete the data and return
    if(registration::get_fini_status() > 0)
    {
        _session_ptr->reset();
        delete _session_ptr;
        return false;
    }

    // cleanup the pooled signal data and release the signal back to the pool for reuse
    auto _cleanup = common::scope_destructor{[&_session_ptr]() {
        _session_ptr->reset();
        delete _session_ptr;
        _session_ptr = nullptr;
    }};

    auto _session = *_session_ptr;  // make a copy of the shared pointer to extend lifetime for the
                                    // duration of this function
    if(!_session.get())
    {
        ROCP_FATAL << fmt::format("nullptr to session information");
        return true;
    }

    auto& queue_info_session = *_session;

    for(auto& packet : queue_info_session.packet_data)
    {
        auto dispatch_time = kernel_dispatch::get_dispatch_time(queue_info_session, packet);
        kernel_dispatch::dispatch_complete(queue_info_session, packet, dispatch_time);

        // Calls our internal callbacks to callers who need to be notified post
        // kernel execution.
        queue_info_session.queue.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_data] : map)
            {
                cb_data.signal_completion(queue_info_session.queue,
                                          packet.kernel_packet,
                                          _session,
                                          packet,
                                          packet.instrumentation_packets,
                                          dispatch_time);
            }
        });

        CHECK_NOTNULL(hsa::get_queue_controller())
            ->serializer(&queue_info_session.queue)
            .wlock([&](auto& serializer) {
                serializer.kernel_completion_signal(queue_info_session.queue, packet.is_serialized);
            });

        auto _should_destroy_signal = [&packet](auto _hsa_signal) {
            // if there is a pooled signal, make sure we return value if the .handle matches.
            // if there is not a pool signal, return true if the signal is not null
            return (packet.pooled_signal) ? (_hsa_signal != null_hsa_signal &&
                                             _hsa_signal != packet.pooled_signal->get().value)
                                          : (_hsa_signal != null_hsa_signal);
        };

        // Signal that we have completed via the interrupt signal.
        if(packet.interrupt_signal != null_hsa_signal)
        {
#if !defined(NDEBUG)
            CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
                signals.erase(packet.interrupt_signal.handle);
            });
#endif

            hsa::get_core_table()->hsa_signal_store_screlease_fn(packet.interrupt_signal, -1);
        }

        if(_should_destroy_signal(packet.interrupt_signal))
        {
            ROCP_TRACE << fmt::format("Destroying interrupt signal {{.handle={}}}",
                                      packet.interrupt_signal.handle);
            hsa::get_core_table()->hsa_signal_destroy_fn(packet.interrupt_signal);
        }

        if(_should_destroy_signal(packet.completion_signal) &&
           packet.completion_signal != packet.interrupt_signal)
        {
            ROCP_TRACE << fmt::format("Destroying completion signal {{.handle={}}}",
                                      packet.completion_signal.handle);
            hsa::get_core_table()->hsa_signal_destroy_fn(packet.completion_signal);
        }

        // if the completion signal was from the pool, we just release it back to the pool for
        // reuse.
        if(packet.pooled_signal)
        {
            Queue::release_signal(packet.pooled_signal);
        }

        // we need to decrement this reference count at the end of the functions
        auto* _corr_id = queue_info_session.correlation_id;
        if(_corr_id)
        {
            ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
                << "reference counter for correlation id " << _corr_id->internal << " from thread "
                << _corr_id->thread_idx << " has no reference count";
            _corr_id->sub_kern_count();
            _corr_id->sub_ref_count();
        }
    }

    queue_info_session.queue.async_complete();

    return false;
}

bool
BarrierAsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    using session_info_t = std::shared_ptr<barrier_info_session_t>;

    if(!data)
    {
        ROCP_FATAL << "BarrierAsyncSignalHandler called with null data pointer";
        return true;
    }

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
    if(!_session.get())
    {
        ROCP_FATAL << "nullptr to barrier session information";
        return true;
    }

    auto& barrier_session = *_session;

    for(auto& bdata : barrier_session.barrier_data)
    {
        auto barrier_time = hip::event::profiling_time{.status = HSA_STATUS_SUCCESS,
                                                       .start  = barrier_session.enqueue_ts,
                                                       .end    = common::timestamp_ns()};

        auto* _corr_id          = barrier_session.correlation_id;
        auto  _tid              = barrier_session.tid;
        auto  _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;
        auto  _ancestor_corr_id = (_corr_id) ? _corr_id->ancestor : 0;

        hip::event::barrier_complete(bdata.tracing_data,
                                     _tid,
                                     _internal_corr_id,
                                     _ancestor_corr_id,
                                     barrier_time,
                                     bdata.operation,
                                     bdata.callback_record);

        if(bdata.operation == ROCPROFILER_HIP_EVENT_RECORD)
        {
            auto group = hip::event::lookup_coalesce_group(bdata.callback_record.hip_event_handle);
            if(group)
            {
                group->wlock([&](auto& g) {
                    g.barrier_time = barrier_time;
                    g.completed    = true;
                    for(auto& p : g.pending)
                    {
                        hip::event::barrier_complete(p.tracing_data,
                                                     p.tid,
                                                     p.internal_corr_id,
                                                     p.ancestor_corr_id,
                                                     barrier_time,
                                                     ROCPROFILER_HIP_EVENT_RECORD,
                                                     p.callback_record);
                    }
                    g.pending.clear();
                });
            }
        }

        if(bdata.pooled_signal)
        {
            Queue::release_signal(bdata.pooled_signal);
        }

        if(_corr_id)
        {
            _corr_id->sub_kern_count();
            _corr_id->sub_ref_count();
        }
    }

    barrier_session.queue.async_complete();

    return false;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
Integral
bit_extract(Integral x, int first, int last)
{
    static_assert(std::is_integral<Integral>::value, "Integral type required");

    auto&& bit_mask = [](int _first, int _last) {
        ROCP_FATAL_IF(!(_last >= _first)) << fmt::format(
            "[queue::bit_extract::bit_mask] -> invalid argument. last (={}) is not >= first (={})",
            _last,
            _first);

        size_t num_bits = _last - _first + 1;
        return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                                   /* num_bits exceed the size of Integral */
                                                   : ((Integral{1} << num_bits) - 1))
               << _first;
    };

    return (x >> first) & bit_mask(0, last - first);
}

/**
 * @brief This function is a queue write interceptor. It intercepts the
 * packet write function. Creates an instance of packet class with the raw
 * pointer. invoke the populate function of the packet class which returns a
 * pointer to the packet. This packet is written into the queue by this
 * interceptor by invoking the writer function.
 */
void
WriteInterceptor(const void* packets,
                 uint64_t    pkt_count,
                 uint64_t,
                 void*                                 data,
                 hsa_amd_queue_intercept_packet_writer writer)
{
    if(registration::get_fini_status() > 0)
    {
        writer(packets, pkt_count);
        return;
    }

    ROCP_TRACE << fmt::format("WriteInterceptor called with pkt_count={}", pkt_count);

    using callback_record_t = packet_data_t::callback_record_t;
    using packet_vector_t   = common::container::small_vector<rocprofiler_packet, 512>;

    // unique sequence id for the dispatch
    static auto sequence_counter = std::atomic<rocprofiler_dispatch_id_t>{0};

    // cannot capture since static
    static auto CreateBarrierPacket = [](hsa_signal_t*    dependency_signal,
                                         hsa_signal_t*    completion_signal,
                                         packet_vector_t& _packets) -> void {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
        if(dependency_signal != nullptr) barrier.dep_signal[0] = *dependency_signal;
        if(completion_signal != nullptr) barrier.completion_signal = *completion_signal;
        _packets.emplace_back(barrier);
    };

    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

    auto& queue = *static_cast<Queue*>(data);

    auto*      gls                 = ::rocprofiler::hip::graph::current_launch_state();
    const bool graph_launch_active = (gls != nullptr);
    const bool event_api_active    = (hip::event::get_active_event_context() != nullptr);
    const bool no_real_consumers =
        (queue.get_notifiers() == 0 &&
         context::get_active_contexts(full_packet_instrumentation_context_filter).empty());

    const bool has_kernel_replay = kernel_replay::has_active_replay_contexts();

    if(pkt_count == 0 || (no_real_consumers && !graph_launch_active && !has_kernel_replay &&
                          !event_api_active))
    {
        writer(packets, pkt_count);
        return;
    }

    const auto* packets_arr          = static_cast<const rocprofiler_packet*>(packets);
    auto        num_dispatch_packets = size_t{0};
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            ++num_dispatch_packets;
        }
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
        else if(packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
        {
            const auto& ext_packet = packets_arr[i].ext_kernel_dispatch;
            if(ext_packet.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
            {
                ++num_dispatch_packets;
            }
        }
#endif
    }

    auto* active_event_ctx = hip::event::get_active_event_context();
    if(num_dispatch_packets == 0 && !active_event_ctx)
    {
        writer(packets, pkt_count);
        return;
    }

    if(has_kernel_replay)
    {
        if(graph_launch_active)
        {
            LOG_FIRST_N(WARNING, 1) << "kernel replay: HIP graph launches are not supported";
        }
        else if(pkt_count != 1)
        {
            LOG_FIRST_N(WARNING, 1) << fmt::format(
                "kernel replay: only single-packet dispatch submissions are replayed. A "
                "submission of {} packets ({} dispatches) ran once without replay",
                pkt_count,
                num_dispatch_packets);
        }
    }

    // Fast path: graph_launch is the only reason we're here. Increment the per-launch
    // dispatch count and write the original packets without allocating signals or rewriting
    // packets. Large graph launches in summary-only mode would otherwise pay the full
    // tracing overhead and exhaust the HSA signal pool.
    //
    // Excluded while a replay service is active. A graph's dispatches are never replayed, but they
    // still write device memory, and this path neither takes the per-agent reader lock nor calls
    // async_started(). A replay on another thread would therefore find the reader lock free and
    // every queue reporting zero in-flight work, then snapshot underneath the running graph and
    // restore over its writes -- corrupting application data, not just the counters. Falling
    // through puts the graph on the ordinary instrumented path, which takes the reader lock to keep
    // it out of the snapshot window and registers it with async_started() so the agent-wide drain
    // can see it. process_packet_batch increments gls->dispatch_count per dispatch packet, so the
    // graph summary is unaffected by which path runs.
    if(graph_launch_active && no_real_consumers && !has_kernel_replay)
    {
        gls->dispatch_count += num_dispatch_packets;
        writer(packets, pkt_count);
        return;
    }

    // these are for the services (dispatch counter collection, pc sampling, ATT) which use
    // the queue/queue_controller callback mechanism
    const auto queue_callback_context_filter = [](const context::context* ctx) {
        return (ctx->dispatch_counter_collection || ctx->pc_sampler || ctx->dispatch_thread_trace ||
                ctx->dispatch_spm);
    };

    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               tracing_data_v);

    auto hip_event_tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                               ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                               hip_event_tracing_data_v);

    for(const auto* itr : context::get_active_contexts(queue_callback_context_filter))
        tracing_data_v.external_correlation_ids.emplace(itr, tracing::empty_user_data);

    // all packets should have the same correlation id so we can just look at the first one to
    // get the correlation id for the entire batch of packets
    auto*                    corr_id      = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    // Allocate a correlation id if we have at least one dispatch packet and we don't have a
    // correlation id already. There will not be a correlation id if there is no API tracing but
    // it was requested by tools to always provide one.
    if(!corr_id)
    {
        constexpr auto ref_count = 1;
        corr_id                  = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = corr_id;
    }

    // During finalization, correlation tracing service will not construct a correlation id so
    // just write packet through without tracing
    if(!corr_id)
    {
        writer(packets, pkt_count);
        return;
    }

    // if we constructed a correlation id, this decrements the reference count after the
    // underlying function returns
    auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop]() {
        if(_corr_id_pop)
        {
            context::pop_latest_correlation_id(_corr_id_pop);
            _corr_id_pop->sub_ref_count();
        }
    }};

    using packet_writer_fn_t = std::function<void(packet_vector_t &&)>;

    auto process_packet_batch = [&queue,
                                 &corr_id,
                                 tracing_data_v,
                                 hip_event_tracing_data_v,
                                 has_kernel_replay](
                                    const rocprofiler_packet* _packets,
                                    uint64_t                  _num_packets,
                                    const packet_writer_fn_t& _writer,
                                    bool                      is_replay_pass       = false,
                                    rocprofiler_dispatch_id_t reserved_dispatch_id = 0) {
        auto transformed_packets = packet_vector_t{};

        auto thr_id           = corr_id->thread_idx;
        auto internal_corr_id = corr_id->internal;
        auto ancestor_corr_id = corr_id->ancestor;

        using packet_data_array_t = queue_info_session_t::packet_data_array_t;

        auto _info_session = queue_info_session_t{.queue          = queue,
                                                  .tid            = thr_id,
                                                  .enqueue_ts     = common::timestamp_ns(),
                                                  .correlation_id = corr_id,
                                                  .packet_data    = packet_data_array_t{}};

        // Searching across all the packets given during this write
        for(size_t i = 0; i < _num_packets; ++i)
        {
            const auto& original_packet = _packets[i].kernel_dispatch;
            auto        packet_type =
                bit_extract(original_packet.header,
                            HSA_PACKET_HEADER_TYPE,
                            HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
            bool is_kernel_dispatch     = (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH);
            bool is_ext_kernel_dispatch = false;

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
            if(packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
            {
                const auto& ext_packet = _packets[i].ext_kernel_dispatch;
                if(ext_packet.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
                {
                    is_ext_kernel_dispatch = true;
                }
            }
#endif

            if(!is_kernel_dispatch && !is_ext_kernel_dispatch)
            {
                // CLR uses BARRIER_AND or vendor-specific BARRIER_VALUE for event
                // barriers depending on device settings. BARRIER_OR is never used
                // by CLR's event path and is not intercepted here.
                bool is_barrier = (packet_type == HSA_PACKET_TYPE_BARRIER_AND);
                if(!is_barrier && packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
                {
                    const auto& vendor_hdr =
                        reinterpret_cast<const hsa_amd_vendor_packet_header_t&>(
                            _packets[i].ext_amd_aql_pm4.header);
                    is_barrier = (vendor_hdr.AmdFormat == HSA_AMD_PACKET_TYPE_BARRIER_VALUE);
                }

                auto* event_ctx = hip::event::get_active_event_context();

                if(is_barrier && event_ctx && !hip_event_tracing_data_v.empty())
                {
                    corr_id->add_ref_count();
                    corr_id->add_kern_count();

                    auto _barrier_data         = barrier_data_t{};
                    _barrier_data.tracing_data = hip_event_tracing_data_v;
                    _barrier_data.operation    = event_ctx->operation;

                    tracing::populate_external_correlation_ids(
                        _barrier_data.tracing_data.external_correlation_ids,
                        thr_id,
                        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT,
                        event_ctx->operation,
                        internal_corr_id);

                    auto source_queue = queue.get_id();
                    if(event_ctx->operation == ROCPROFILER_HIP_EVENT_RECORD)
                    {
                        hip::event::record_event_queue(event_ctx->hip_event_handle, queue.get_id());
                    }
                    else if(event_ctx->operation == ROCPROFILER_HIP_EVENT_WAIT)
                    {
                        source_queue = hip::event::lookup_event_queue(event_ctx->hip_event_handle);
                    }

                    _barrier_data.callback_record = rocprofiler_callback_tracing_hip_event_data_t{
                        sizeof(rocprofiler_callback_tracing_hip_event_data_t),
                        rocprofiler_timestamp_t{0},
                        rocprofiler_timestamp_t{0},
                        queue.get_agent().get_rocp_agent()->id,
                        queue.get_id(),
                        event_ctx->hip_event_handle,
                        source_queue};

                    {
                        auto tracer_data = _barrier_data.callback_record;
                        tracing::execute_phase_enter_callbacks(
                            _barrier_data.tracing_data.callback_contexts,
                            thr_id,
                            internal_corr_id,
                            _barrier_data.tracing_data.external_correlation_ids,
                            ancestor_corr_id,
                            ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                            event_ctx->operation,
                            tracer_data);
                    }

                    tracing::update_external_correlation_ids(
                        _barrier_data.tracing_data.external_correlation_ids,
                        thr_id,
                        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_EVENT);

                    const auto& barrier_pkt                = _packets[i].barrier_and;
                    const auto  original_completion_signal = barrier_pkt.completion_signal;
                    const bool  existing_completion_signal =
                        (original_completion_signal.handle != 0);

                    auto barrier_copy = _packets[i];
                    _barrier_data.pooled_signal =
                        queue.create_signal(0, &barrier_copy.barrier_and.completion_signal, true);

                    _barrier_data.completion_signal = barrier_copy.barrier_and.completion_signal;

                    get_core_table()->hsa_signal_store_screlease_fn(_barrier_data.completion_signal,
                                                                    0);

                    transformed_packets.emplace_back(barrier_copy);

                    if(existing_completion_signal)
                    {
                        auto forwarding   = hsa_barrier_and_packet_t{};
                        forwarding.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
                        forwarding.header |= (1 << HSA_PACKET_HEADER_BARRIER);
                        forwarding.completion_signal = original_completion_signal;
                        transformed_packets.emplace_back(forwarding);
                    }

                    {
                        auto tracer_data = _barrier_data.callback_record;
                        tracing::execute_phase_exit_callbacks(
                            _barrier_data.tracing_data.callback_contexts,
                            _barrier_data.tracing_data.external_correlation_ids,
                            ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                            event_ctx->operation,
                            tracer_data);
                    }

                    event_ctx->barrier_captured = true;

                    if(event_ctx->operation == ROCPROFILER_HIP_EVENT_RECORD)
                    {
                        auto group =
                            std::make_shared<common::Synchronized<hip::event::coalesce_group_t>>();
                        hip::event::store_coalesce_group(event_ctx->hip_event_handle, group);
                    }

                    auto barrier_session_data = barrier_info_session_t::barrier_data_array_t{};
                    barrier_session_data.emplace_back(std::move(_barrier_data));

                    auto barrier_session =
                        barrier_info_session_t{.queue          = queue,
                                               .tid            = thr_id,
                                               .enqueue_ts     = common::timestamp_ns(),
                                               .correlation_id = corr_id,
                                               .barrier_data   = std::move(barrier_session_data)};

                    auto shared =
                        std::make_shared<barrier_info_session_t>(std::move(barrier_session));

                    const auto raw_signal = shared->barrier_data.back().pooled_signal->get().value;

                    queue.async_started();

                    auto status = hsa::get_amd_ext_table()->hsa_amd_signal_async_handler_fn(
                        raw_signal,
                        HSA_SIGNAL_CONDITION_EQ,
                        -1,
                        BarrierAsyncSignalHandler,
                        new std::shared_ptr<barrier_info_session_t>(shared));

                    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
                        << fmt::format("Error: hsa_amd_signal_async_handler for barrier "
                                       "(signal={{.handle={}}}) failed with error code {}",
                                       raw_signal.handle,
                                       static_cast<int>(status));
                }
                else
                {
                    transformed_packets.emplace_back(_packets[i]);
                }
                continue;
            }

            // increase the reference count to denote that this correlation id is being used in a
            // kernel
            corr_id->add_ref_count();
            corr_id->add_kern_count();

            auto _packet_data = packet_data_t{};

            // make a copy of the tracing data
            _packet_data.tracing_data = tracing_data_v;

            // Kernel-replay localized context control: when a replay pass has toggled contexts,
            // drop the disabled ones so their timestamp records are skipped. Gated on a single
            // thread-local check so normal dispatches pay ~nothing. See
            // kernel_replay/local_context.hpp.
            if(kernel_replay::local_context_has_overrides())
            {
                auto disabled = [](const auto& e) {
                    auto ov = kernel_replay::local_context_override({.handle = e.ctx->context_idx});
                    return ov.has_value() && !*ov;
                };
                auto& cbc = _packet_data.tracing_data.callback_contexts;
                auto& bfc = _packet_data.tracing_data.buffered_contexts;
                cbc.erase(std::remove_if(cbc.begin(), cbc.end(), disabled), cbc.end());
                bfc.erase(std::remove_if(bfc.begin(), bfc.end(), disabled), bfc.end());
            }

            tracing::populate_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                internal_corr_id);

            // Lambda to extract packet info regardless of packet type
            auto extract_packet_info = [](const rocprofiler_packet& pkt, bool is_ext) {
                struct packet_info
                {
                    hsa_signal_t       completion_signal;
                    uint64_t           kernel_object;
                    uint32_t           private_segment_size;
                    uint32_t           group_segment_size;
                    rocprofiler_dim3_t workgroup_size;
                    rocprofiler_dim3_t grid_size;
                };

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
                if(is_ext)
                {
                    const auto& e = pkt.ext_kernel_dispatch;
                    return packet_info{e.completion_signal,
                                       e.kernel_object,
                                       e.private_segment_size,
                                       e.group_segment_size,
                                       {e.workgroup_size_x, e.workgroup_size_y, e.workgroup_size_z},
                                       {static_cast<uint32_t>(e.cluster_count_x) *
                                            static_cast<uint32_t>(e.cluster_size_x) *
                                            static_cast<uint32_t>(e.workgroup_size_x),
                                        static_cast<uint32_t>(e.cluster_count_y) *
                                            static_cast<uint32_t>(e.cluster_size_y) *
                                            static_cast<uint32_t>(e.workgroup_size_y),
                                        static_cast<uint32_t>(e.cluster_count_z) *
                                            static_cast<uint32_t>(e.cluster_size_z) *
                                            static_cast<uint32_t>(e.workgroup_size_z)}};
                }
#else
                (void) is_ext;
#endif
                {
                    const auto& s = pkt.kernel_dispatch;
                    return packet_info{s.completion_signal,
                                       s.kernel_object,
                                       s.private_segment_size,
                                       s.group_segment_size,
                                       {s.workgroup_size_x, s.workgroup_size_y, s.workgroup_size_z},
                                       {s.grid_size_x, s.grid_size_y, s.grid_size_z}};
                }
            };

            const auto     pkt_info = extract_packet_info(_packets[i], is_ext_kernel_dispatch);
            const auto     original_completion_signal = pkt_info.completion_signal;
            const bool     existing_completion_signal = (original_completion_signal.handle != 0);
            const uint64_t kernel_id = code_object::get_kernel_id(pkt_info.kernel_object);

            // Copy kernel pkt, copy is to allow for signal to be modified
            _packet_data.kernel_packet = _packets[i];
            // create a reference for short hand access
            auto& kernel_packet = _packet_data.kernel_packet;

            // create our own signal that we can get a callback on. if there is an original
            // completion signal we will create a barrier packet, assign the original completion
            // signal that that barrier packet, and add it right after the kernel packet
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
            if(is_ext_kernel_dispatch)
                _packet_data.pooled_signal = queue.create_signal(
                    0, &kernel_packet.ext_kernel_dispatch.completion_signal, true);
            else
#endif
                _packet_data.pooled_signal =
                    queue.create_signal(0, &kernel_packet.kernel_dispatch.completion_signal, true);

            // computes the "size" based on the offset of reserved_padding field
            constexpr auto kernel_dispatch_info_rt_size =
                common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

            static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                          "failed to compute size field based on offset of reserved_padding field");

            // A caller-reserved id (a replay dispatch: its passes, or the single run when replay is
            // declined) is used as-is so CONFIG, the passes, and every record share one id;
            // everything else mints a fresh id here. Exactly one mint per logical dispatch.
            ROCP_FATAL_IF(reserved_dispatch_id != 0 && (_num_packets != 1 || !has_kernel_replay))
                << fmt::format("kernel replay: a reserved dispatch id must only be used for a "
                               "single-packet replay dispatch");
            auto dispatch_id =
                (reserved_dispatch_id != 0) ? reserved_dispatch_id : ++sequence_counter;

            // Always feed HIP_GRAPH summary's kernel_dispatch_count (independent of
            // subscription).
            if(auto* graph_launch_state = ::rocprofiler::hip::graph::current_launch_state();
               graph_launch_state != nullptr)
                ++graph_launch_state->dispatch_count;

            _packet_data.callback_record =
                callback_record_t{sizeof(callback_record_t),
                                  rocprofiler_timestamp_t{0},
                                  rocprofiler_timestamp_t{0},
                                  rocprofiler_kernel_dispatch_info_t{
                                      .size        = kernel_dispatch_info_rt_size,
                                      .agent_id    = queue.get_agent().get_rocp_agent()->id,
                                      .queue_id    = queue.get_id(),
                                      .kernel_id   = kernel_id,
                                      .dispatch_id = dispatch_id,
                                      .private_segment_size = pkt_info.private_segment_size,
                                      .group_segment_size   = pkt_info.group_segment_size,
                                      .workgroup_size       = pkt_info.workgroup_size,
                                      .grid_size            = pkt_info.grid_size,
                                      .reserved_padding     = {0}}};

            {
                auto tracer_data = _packet_data.callback_record;
                tracing::execute_phase_enter_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    thr_id,
                    internal_corr_id,
                    _packet_data.tracing_data.external_correlation_ids,
                    ancestor_corr_id,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
            }

            // map all the external correlation ids (after enqueue enter phase) for all the contexts
            // captured by the info session
            tracing::update_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

            // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
            // along with an ID of the client we got the packet from (this will be returned via
            // completed_cb_t)

            // Signal callbacks that a kernel_packet is being enqueued
            queue.signal_callback([&](const auto& map) {
                for(const auto& [client_id, cb_data] : map)
                {
                    // NOTE: if map.size() > 1, multiple callbacks will be sharing the same user
                    // data. This needs to be fixed. (bewelton)
                    auto [packet, bSerial] = cb_data.write_interceptor(
                        queue,
                        kernel_packet,
                        kernel_id,
                        dispatch_id,
                        &_packet_data.user_data,
                        _packet_data.tracing_data.external_correlation_ids,
                        corr_id);
                    _packet_data.is_serialized |= bSerial;
                    if(packet)
                        _packet_data.instrumentation_packets.push_back(
                            std::make_pair(std::move(packet), client_id));
                }
            });

            bool inserted_before = false;
            if(_packet_data.is_serialized)
            {
                inserted_before = true;
                CHECK_NOTNULL(hsa::get_queue_controller())
                    ->serializer(&queue)
                    .rlock([&](const auto& serializer) {
                        for(auto& s_pkt : serializer.kernel_dispatch(queue))
                            transformed_packets.emplace_back(s_pkt.kernel_dispatch);
                    });
            }

            for(const auto& pkt_injection : _packet_data.instrumentation_packets)
            {
                if(!pkt_injection.first->before_krn_barrier_pkt.empty())
                {
                    for(const auto& pkt : pkt_injection.first->before_krn_barrier_pkt)
                    {
                        transformed_packets.emplace_back(pkt);
                    }
                }
            }
            for(const auto& pkt_injection : _packet_data.instrumentation_packets)
            {
                for(const auto& pkt : pkt_injection.first->before_krn_pkt)
                {
                    inserted_before = true;
                    transformed_packets.emplace_back(pkt);
                }
            }

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
            if(pc_sampling::is_pc_sample_service_configured(queue.get_agent().get_rocp_agent()->id))
            {
                transformed_packets.emplace_back(
                    pc_sampling::hsa::generate_marker_packet_for_kernel(
                        corr_id, _packet_data.tracing_data.external_correlation_ids, dispatch_id));
            }
#endif

            // emplace the kernel packet
            transformed_packets.emplace_back(kernel_packet);

            // If a profiling packet was inserted, wait for completion before executing the dispatch
            if(inserted_before)
                transformed_packets.back().kernel_dispatch.header |= 1 << HSA_PACKET_HEADER_BARRIER;

            // if the original completion signal exists, trigger it via a barrier packet.
            // Replay: suppress the app's completion signal on every pass; WriteInterceptor fires it
            // once after the whole loop so the application observes a single execution regardless
            // of pass count, early exit, or indefinite loops.
            if(existing_completion_signal && !is_replay_pass)
            {
                auto barrier   = hsa_barrier_and_packet_t{};
                barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
                barrier.header |= (1 << HSA_PACKET_HEADER_BARRIER);
                barrier.completion_signal = original_completion_signal;
                transformed_packets.emplace_back(barrier);
            }

            bool injected_end_pkt = false;
            for(const auto& pkt_injection : _packet_data.instrumentation_packets)
            {
                for(const auto& pkt : pkt_injection.first->after_krn_pkt)
                {
                    transformed_packets.emplace_back(pkt);
                    injected_end_pkt = true;
                }
            }

            auto& completion_signal = _packet_data.completion_signal;
            auto& interrupt_signal  = _packet_data.interrupt_signal;
            if(injected_end_pkt)
            {
                // Adding a barrier packet with the original packet's completion signal.
                queue.create_signal(0, &interrupt_signal, false);
                completion_signal                                            = interrupt_signal;
                transformed_packets.back().kernel_dispatch.completion_signal = interrupt_signal;
                CreateBarrierPacket(&interrupt_signal, &interrupt_signal, transformed_packets);
            }
            else
            {
                completion_signal = kernel_packet.kernel_dispatch.completion_signal;
                get_core_table()->hsa_signal_store_screlease_fn(completion_signal, 0);
            }

            ROCP_FATAL_IF(!(is_kernel_dispatch || is_ext_kernel_dispatch))
                << "get_kernel_id below might need to be updated";

            {
                auto tracer_data = _packet_data.callback_record;
                tracing::execute_phase_exit_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    _packet_data.tracing_data.external_correlation_ids,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
            }

            _info_session.packet_data.emplace_back(std::move(_packet_data));
        }

        using info_session_t = queue_info_session_t;

        if(!_info_session.packet_data.empty())
        {
            auto* last_pooled_signal     = _info_session.packet_data.back().pooled_signal;
            auto  last_completion_signal = _info_session.packet_data.back().completion_signal;

            auto shared = std::make_shared<info_session_t>(std::move(_info_session));

            // Enqueue the signal into the handler. Will call completed_cb when signal completes.
            queue.async_started();
            queue.signal_async_handler(last_pooled_signal,
                                       last_completion_signal,
                                       new std::shared_ptr<info_session_t>(shared));
        }

        // Command is only executed if GLOG_v=2 or higher, otherwise it is a no-op
        ROCP_TRACE << fmt::format("QueueID {}: {}",
                                  queue.get_id().handle,
                                  fmt::join(transformed_packets, fmt::format(" ")));

        _writer(std::move(transformed_packets));
    };

    // Forwards a finished packet batch to the real queue submit. Shared by the replay passes, the
    // snapshot-declined single run, and the batched non-replay path (the non-batched path below
    // collects into a local vector instead).
    const auto forward_to_writer = [&writer](packet_vector_t&& _packets) {
        writer(_packets.data(), _packets.size());
    };

    // Kernel replay: re-run a single dispatch packet N times with device-memory snap/restore
    // between passes. The application observes only one execution. Runs synchronously on this
    // (WriteInterceptor) thread; reuses process_packet_batch for each pass so counter collection,
    // the serializer, the async signal handler, and record_callback all behave exactly as in the
    // single-pass path. Opt-in and gated so non-replay runs are unchanged.
    // TODO: inline process_packet_batch / graphs
    //
    // One dispatch id is reserved for this logical dispatch (below) and reused by every submit --
    // the replay passes and, if replay is declined, the single fall-through run -- so CONFIG/PASS
    // callbacks and every record share it and the counter is bumped exactly once. 0 means "not a
    // replay dispatch"; the normal path then mints its own id.
    rocprofiler_dispatch_id_t replay_dispatch_id = 0;
    // A graph node's dispatch is never replayed. Snapshot/restore around a graph's runtime-managed
    // memory and ordering is undefined, and graph replay is future work. Excluding
    // graph_launch_active from the gate declines the graph gracefully instead of aborting: it falls
    // through to the ordinary path and runs once, and the one-shot warning above already told the
    // tool. Non-graph single dispatches replay as usual below.
    if(has_kernel_replay && pkt_count == 1 && num_dispatch_packets == 1 && !graph_launch_active)
    {
        const auto thr_id           = corr_id->thread_idx;
        const auto internal_corr_id = corr_id->internal;
        const auto ancestor_corr_id = corr_id->ancestor;
        const auto dispatch_pkt     = packets_arr[0];

        // Reserve the dispatch id before CONFIG so replay_pass_count and every CONFIG/PASS
        // callback see the same id the replay's dispatch records will carry (make_dispatch_info
        // leaves it 0).
        // Every submit below passes it as reserved_dispatch_id, so the counter is bumped once.
        replay_dispatch_id     = ++sequence_counter;
        const auto replay_plan = kernel_replay::execute_config_phase_enter(
            queue, dispatch_pkt, thr_id, internal_corr_id, ancestor_corr_id, replay_dispatch_id);

        if(replay_plan.replay_requested)
        {
            // Runs synchronously on the calling (WriteInterceptor) thread. Concurrent work is
            // isolated by (a) the per-agent WRITER lock taken below, which serializes this whole
            // drain->snap->passes->restore window against other replays *and* against non-replay
            // dispatches on this agent (they hold the reader lock across their submit), and
            // (b) agent-scoped snapshots so a replay only saves/restores its own agent's device
            // memory (other GPUs untouched). Different agents hold different locks and run
            // concurrently.
            const auto& core         = queue.core_api();
            hsa_agent_t replay_agent = queue.get_agent().get_hsa_agent();
            const auto  replay_guard = std::unique_lock<std::shared_mutex>{
                agent_replay_mutex(queue.get_agent().get_rocp_agent()->id)};

            // The app's original completion signal (completion_signal is at the same offset for
            // dispatch and ext-dispatch packets, per the static_asserts at the top of this file).
            // It is suppressed on every replay pass and fired once after the loop so the
            // application observes a single completion regardless of pass count / early exit /
            // indefinite loop.
            hsa_signal_t app_completion_signal = dispatch_pkt.kernel_dispatch.completion_signal;

            // Drain barrier: fence the CPU against all prior in-flight GPU work on this queue so
            // device memory is stable before snapshotting.
            hsa_signal_t drain_signal = null_hsa_signal;
            Queue::create_signal(0, &drain_signal, /*use_pool=*/false);
            {
                auto drain_pkts = packet_vector_t{};
                CreateBarrierPacket(nullptr, &drain_signal, drain_pkts);
                writer(drain_pkts.data(), drain_pkts.size());

                replay_wait_or_fatal(
                    [&]() {
                        return core.hsa_signal_wait_scacquire_fn(
                                   drain_signal,
                                   HSA_SIGNAL_CONDITION_EQ,
                                   0,
                                   std::chrono::nanoseconds{drain_slice}.count(),
                                   HSA_WAIT_STATE_BLOCKED) == 0;
                    },
                    "this queue's prior GPU work");
            }

            // Agent-wide drain. Sibling queues on this agent can have kernels in flight that mutate
            // device memory while we snapshot and restore. The per-agent replay lock blocks other
            // threads' replayed dispatches, since every kernel dispatch passes through that gate.
            // It does not block work already submitted to their queues. So wait for every queue on
            // this agent to finish its outstanding kernels before snapshotting. We wait outside the
            // queue-map lock so it does not block stream creation or destruction. See
            // replay_drain_agent_or_fatal. Async SDMA copies bypass both the AQL queues and the
            // replay gate and serializing those is a separate follow-up (TODO: mkuriche,
            // amd-vkale).
            replay_drain_agent_or_fatal(replay_agent);

            // Save this agent's tracked device allocations so every pass runs against identical
            // inputs. snap() returns ok=false if it could not capture the complete set (host memory
            // pressure, a failed copy, or module-scope variables it could not enumerate). It logs
            // which of those it hit.
            const auto snapshot = kernel_replay::memory_snapshot::snap(replay_agent);

            // Snapshot incomplete: restoring a partial snapshot between passes would corrupt
            // application data, so decline replay. Close the CONFIG sequence, free our drain
            // signal, and run this dispatch once still under the writer lock with its original
            // completion signal, then return.
            if(!snapshot.ok)
            {
                LOG_FIRST_N(WARNING, 1) << "kernel replay: snapshot capture incomplete; running "
                                           "this dispatch once without replay";
                kernel_replay::execute_config_phase_exit(
                    replay_plan, thr_id, internal_corr_id, ancestor_corr_id);
                if(drain_signal != null_hsa_signal)
                    get_core_table()->hsa_signal_destroy_fn(drain_signal);
                process_packet_batch(packets_arr,
                                     1,
                                     forward_to_writer,
                                     /*is_replay_pass=*/false,
                                     replay_dispatch_id);
                return;
            }

            // Localized context control for this replay loop. This guard installs the thread-local
            // routing that connects the tool's PASS toggle callbacks (writers, via
            // replay_local_enable/disable_context) to the services that read it at dispatch (via
            // kernel_replay::local_context_override). It lives for the whole loop and is torn down
            // when the guard exits; global context state is never touched. It captures the contexts
            // active now (loop start) as the toggle mask, so a tool may only enable/disable one of
            // those and a local start cannot promote a globally-stopped context
            // (local_context.hpp).
            auto local_ctx_tls_guard =
                kernel_replay::scoped_local_context_control{context::get_active_contexts()};

            // Per-pass loop: PASS enter -> submit -> drain the async handler -> PASS exit -> ask
            // the tool whether to continue -> restore device memory before the next pass.
            for(uint64_t pass = 0;; ++pass)
            {
                const bool is_final =
                    !replay_plan.indefinite && (pass == replay_plan.total_passes - 1);

                auto pass_state = kernel_replay::pass_context_state_t{};
                kernel_replay::execute_pass_phase_enter(
                    replay_plan, pass, thr_id, internal_corr_id, ancestor_corr_id, pass_state);

                process_packet_batch(packets_arr,
                                     1,
                                     forward_to_writer,
                                     /*is_replay_pass=*/true,
                                     replay_dispatch_id);

                // Drain this pass's async handler (separate HSA thread: reads counters, emits
                // records, releases signals/corr-id refs) before PASS EXIT / continue-decision /
                // restore() / next submit, else we race its record delivery and reuse buffers and
                // signals it still holds. This also implies GPU drain. Exactly one handler is in
                // flight per pass (we drain before each submit, under the agent writer lock).
                ROCP_FATAL_IF(queue.active_async_packets() > 1) << fmt::format(
                    "kernel replay: more than one async handler in flight during a replay pass");
                replay_drain_or_fatal(queue);

                kernel_replay::execute_pass_phase_exit(replay_plan, pass, pass_state);

                // Stop once the tool (or the fixed pass count) says we're done; the last executed
                // pass leaves device memory as the app expects, so no restore follows the break.
                if(!kernel_replay::should_continue_replay(replay_plan, pass_state, pass, is_final))
                    break;

                // Restore device memory between passes so the next pass sees identical inputs.
                // A failed host->device copy leaves the snapshot only partially applied; continuing
                // would submit the next pass over corrupted memory and (because the final pass
                // skips restore) would also leave that corruption visible to the application.
                ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(snapshot)) << fmt::format(
                    "kernel replay: restore failed between passes (partial host->device copy); "
                    "aborting rather than continuing with corrupted device memory");
            }

            kernel_replay::execute_config_phase_exit(
                replay_plan, thr_id, internal_corr_id, ancestor_corr_id);

            // Fire the app's original completion signal once, now that the final executed pass has
            // completed (we already drained the final pass's handler above). A trailing barrier
            // decrements it exactly as the single-pass path would, so the application unblocks and
            // the next kernel on this GPU can dispatch. Deferred out of the per-pass path so
            // early-exit and indefinite loops signal on the actual last pass rather than at pass
            // N-1.
            if(app_completion_signal != null_hsa_signal)
            {
                auto completion_pkts = packet_vector_t{};
                CreateBarrierPacket(nullptr, &app_completion_signal, completion_pkts);
                writer(completion_pkts.data(), completion_pkts.size());
            }

            // Clean up our private signals (never the app's completion signal).
            if(drain_signal != null_hsa_signal)
                get_core_table()->hsa_signal_destroy_fn(drain_signal);
            return;
        }
    }

    // Kernel-replay reader side: while a replay service is active, a non-replay dispatch on this
    // agent must not submit into a replay's snapshot->restore window -- restore() would revert this
    // dispatch's device writes. Hold the per-agent SHARED lock across the submit below so a replay
    // writer waits for in-flight submits to finish and cannot open its window until we return,
    // while ordinary dispatches still run concurrently with each other. Gated on has_kernel_replay
    // so non-replay runs take no lock at all. (The async GPU tail is handled by the replay window's
    // agent-wide drain, not by this lock.)
    std::optional<std::shared_lock<std::shared_mutex>> replay_reader_guard{};
    if(has_kernel_replay)
        replay_reader_guard.emplace(agent_replay_mutex(queue.get_agent().get_rocp_agent()->id));

    bool should_batch_packets = true;
    queue.signal_callback([&should_batch_packets](const auto& map) {
        for(const auto& [_, cb_data] : map)
        {
            if(!cb_data.batch_packets())
            {
                should_batch_packets = false;
                break;
            }
        }
    });

    if(should_batch_packets)
    {
        ROCP_TRACE_IF(pkt_count > 1) << fmt::format(
            "[{}] Batching packets. Number of packets = {}", __FUNCTION__, pkt_count);

        process_packet_batch(packets_arr,
                             pkt_count,
                             forward_to_writer,
                             /*is_replay_pass=*/false,
                             replay_dispatch_id);
    }
    else
    {
        ROCP_TRACE_IF(pkt_count > 1)
            << fmt::format("[{}] Not batching packets due service which does not support batching. "
                           "Number of packets = {}",
                           __FUNCTION__,
                           pkt_count);

        auto transformed_packets = packet_vector_t{};

        for(size_t i = 0; i < pkt_count; ++i)
        {
            process_packet_batch(
                &packets_arr[i],
                1,
                [&transformed_packets](packet_vector_t&& _packets) {
                    transformed_packets.insert(transformed_packets.end(),
                                               std::make_move_iterator(_packets.begin()),
                                               std::make_move_iterator(_packets.end()));
                },
                /*is_replay_pass=*/false,
                replay_dispatch_id);
        }
        writer(transformed_packets.data(), transformed_packets.size());
    }
}
}  // namespace

Queue::Queue(const AgentCache& agent, CoreApiTable table)
: _core_api(table)
, _agent(agent)
{
    _core_api.hsa_signal_create_fn(0, 0, nullptr, &_active_kernels);
}

Queue::Queue(const AgentCache&  agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             CoreApiTable  core_api,
             AmdExtTable   ext_api,
             hsa_queue_t** queue)
: _core_api(core_api)
, _ext_api(ext_api)
, _agent(agent)
{
    ROCP_HSA_TABLE_CALL(FATAL,
                        _ext_api.hsa_amd_queue_intercept_create_fn(_agent.get_hsa_agent(),
                                                                   size,
                                                                   type,
                                                                   callback,
                                                                   data,
                                                                   private_segment_size,
                                                                   group_segment_size,
                                                                   &_intercept_queue))
        << "Could not create intercept queue";

    ROCP_HSA_TABLE_CALL(FATAL,
                        _ext_api.hsa_amd_profiling_set_profiler_enabled_fn(_intercept_queue, true))
        << "Could not setup intercept profiler";

    if(!context::get_registered_contexts([](const context::context* ctx) {
            return (ctx->dispatch_counter_collection || ctx->device_counter_collection ||
                    ctx->dispatch_spm || ctx->dispatch_thread_trace || ctx->device_thread_trace);
        }).empty())
    {
        CHECK(_agent.cpu_pool().handle != 0);
        CHECK(_agent.get_hsa_agent().handle != 0);

        // Set state of the queue to allow profiling
        aql::set_profiler_active_on_queue(
            _agent.cpu_pool(), _agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                hsa_signal_t completion;
                create_signal(0, &completion, false);
                pkt.ext_amd_aql_pm4.completion_signal = completion;
                counters::submitPacket(_intercept_queue, &pkt);
                constexpr auto timeout_hint =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{1});
                hsa_signal_value_t val;
                for(int i = 0; i < 3; i++)
                {
                    val = core_api.hsa_signal_wait_scacquire_fn(completion,
                                                                HSA_SIGNAL_CONDITION_EQ,
                                                                0,
                                                                timeout_hint.count(),
                                                                HSA_WAIT_STATE_ACTIVE);
                    if(val == 0)
                    {
                        core_api.hsa_signal_destroy_fn(completion);
                        return;
                    }
                }
                ROCP_FATAL << "Could not set agent to be profiled - Signal Value: " << val;
            });
    }

    ROCP_HSA_TABLE_CALL(
        FATAL,
        _ext_api.hsa_amd_queue_intercept_register_fn(_intercept_queue, WriteInterceptor, this))
        << "Could not register interceptor";

    create_signal(0, &ready_signal, false);
    create_signal(0, &block_signal, false);
    create_signal(0, &_active_kernels, false);
    _core_api.hsa_signal_store_screlease_fn(ready_signal, 0);
    _core_api.hsa_signal_store_screlease_fn(_active_kernels, 0);
    *queue = _intercept_queue;

    signal_pool_init();  // ensure the signal pool is constructed
}

Queue::Queue(
    const AgentCache&       agent,
    CoreApiTable            core_api,
    AmdExtTable             ext_api,
    hsa_queue_t*            queue,
    set_write_interceptor_t set_write_interceptor)  // NOLINT(performance-unnecessary-value-param)
: _core_api(core_api)
, _ext_api(ext_api)
, _agent(agent)
, _intercept_queue(queue)
{
    if(!context::get_registered_contexts([](const context::context* ctx) {
            return (ctx->dispatch_counter_collection || ctx->device_counter_collection ||
                    ctx->dispatch_thread_trace || ctx->device_thread_trace);
        }).empty())
    {
        CHECK(_agent.cpu_pool().handle != 0);
        CHECK(_agent.get_hsa_agent().handle != 0);

        // Set state of the queue to allow profiling
        aql::set_profiler_active_on_queue(
            _agent.cpu_pool(), _agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                hsa_signal_t completion;
                create_signal(0, &completion, false);
                pkt.ext_amd_aql_pm4.completion_signal = completion;
                counters::submitPacket(_intercept_queue, &pkt);
                constexpr auto timeout_hint =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{1});
                if(core_api.hsa_signal_wait_relaxed_fn(completion,
                                                       HSA_SIGNAL_CONDITION_EQ,
                                                       0,
                                                       timeout_hint.count(),
                                                       HSA_WAIT_STATE_ACTIVE) != 0)
                {
                    ROCP_FATAL << "Could not set agent to be profiled";
                }
                core_api.hsa_signal_destroy_fn(completion);
            });
    }

    create_signal(0, &ready_signal, false);
    create_signal(0, &block_signal, false);
    create_signal(0, &_active_kernels, false);
    _core_api.hsa_signal_store_screlease_fn(ready_signal, 0);
    _core_api.hsa_signal_store_screlease_fn(_active_kernels, 0);

    signal_pool_init();  // ensure the signal pool is constructed
    // Since this is an active queue, the write interceptor may be called immediately, so this needs
    // to appear after signal construction.
    if(!queue_interposition::supports_queue_interposition())
    {
        set_write_interceptor(WriteInterceptor, this);
    }
}

void
Queue::invoke_write_interceptor(const void*                           packets,
                                uint64_t                              pkt_count,
                                hsa_amd_queue_intercept_packet_writer writer) const
{
    WriteInterceptor(packets, pkt_count, 0, const_cast<Queue*>(this), writer);
}

Queue::~Queue()
{
    sync();

    if(_active_kernels.handle != 0 && _core_api.hsa_signal_destroy_fn != nullptr)
    {
        _core_api.hsa_signal_destroy_fn(_active_kernels);
    }
}

void
Queue::signal_async_handler(pooled_signal_t* signal, hsa_signal_t raw_signal, void* data) const
{
#if !defined(NDEBUG)
    CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
        signals[raw_signal.handle] = raw_signal;
    });
#endif

    ROCP_CI_LOG_IF(WARNING, signal && !signal->in_use())
        << fmt::format("pooled signal has not been acquired: hsa_signal_t(.handle={})",
                       signal->get().value.handle);

    hsa_status_t status = _ext_api.hsa_amd_signal_async_handler_fn(
        raw_signal, HSA_SIGNAL_CONDITION_EQ, -1, AsyncSignalHandler, data);

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << fmt::format("Error: hsa_amd_signal_async_handler (signal={{.handle={}}}) failed with "
                       "error code {} :: {} ",
                       raw_signal.handle,
                       static_cast<int>(status),
                       hsa::get_hsa_status_string(status));
}

Queue::pooled_signal_t*
Queue::create_signal(uint32_t attribute, hsa_signal_t* signal, bool use_pool)
{
    if(auto* pool = get_signal_pool(); use_pool && pool && attribute == 0)
    {
        auto& _signal = pool->acquire(construct_hsa_signal, 0, 0, nullptr, attribute);
        ROCP_FATAL_IF(!_signal.in_use()) << "Acquired signal from pool that is not in use";
        *signal = _signal.get().value;
        get_core_table()->hsa_signal_store_screlease_fn(_signal.get().value, 1);
        return &_signal;
    }

    hsa_status_t status =
        get_amd_ext_table()->hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);

    return nullptr;
}

void
Queue::release_signal(pooled_signal_t* signal)
{
    if(signal && signal->in_use())
    {
        // signal->get().data = nullptr;
        ROCP_WARNING_IF(!signal->release())
            << fmt::format("Failed to release a pooled signal: hsa_signal_t{{.handle={}}}",
                           signal->get().value.handle);
        ROCP_TRACE << fmt::format("released signal {}: hsa_signal_t{{.handle={}}}",
                                  signal->index(),
                                  signal->get().value.handle);
    }
}

void
Queue::destroy_signal(pooled_signal_t* signal)
{
    release_signal(signal);

    if(signal && get_core_table() && get_core_table()->hsa_signal_destroy_fn)
    {
        get_core_table()->hsa_signal_destroy_fn(signal->get().value);
        signal->get().value = null_hsa_signal;
    }
}

bool
Queue::sync() const
{
    if(_active_kernels == null_hsa_signal) return true;

    const auto _value =
        _core_api.hsa_signal_wait_relaxed_fn(_active_kernels,
                                             HSA_SIGNAL_CONDITION_EQ,
                                             0,
                                             std::chrono::nanoseconds{drain_slice}.count(),
                                             HSA_WAIT_STATE_BLOCKED);

    ROCP_WARNING_IF(_value != 0) << fmt::format(
        "Timeout while waiting for queue sync: {} kernels still active", _value);
    return _value == 0;
}

void
Queue::register_callback(ClientID id, queue_callbacks_t callbacks)
{
    _callbacks.wlock([&](auto& map) {
        ROCP_FATAL_IF(rocprofiler::common::get_val(map, id)) << "ID already exists!";
        _notifiers++;
        map[id] = std::move(callbacks);
    });
}

void
Queue::remove_callback(ClientID id)
{
    _callbacks.wlock([&](auto& map) {
        if(map.erase(id) == 1) _notifiers--;
    });
}

queue_state
Queue::get_state() const
{
    return _state;
}

void
Queue::set_state(queue_state state)
{
    _state = state;
}

void
queue_init()
{
    // placeholder for future global init if required
}

void
queue_fini()
{
    signal_pool_fini();
}
}  // namespace hsa
}  // namespace rocprofiler
