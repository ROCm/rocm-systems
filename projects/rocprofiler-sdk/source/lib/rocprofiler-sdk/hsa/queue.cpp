// MIT License
//
/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc.

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
#include "lib/common/scope_destructor.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <memory>
#include <sstream>

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

namespace rocprofiler
{
namespace hsa
{
namespace
{
std::atomic<int64_t>&
get_balanced_signal_slots()
{
    constexpr int64_t NUM_SIGNALS = 16;
    static auto*&     atomic = common::static_object<std::atomic<int64_t>>::construct(NUM_SIGNALS);
    return *atomic;
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
context_filter(const context::context* ctx)
{
    return (context_filter(ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) ||
            context_filter(ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
}

bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    if(!data) return true;

    // if we have fully finalized, delete the data and return
    if(registration::get_fini_status() > 0)
    {
        auto* _session = static_cast<Queue::queue_info_session_t**>(data);
        delete _session;
        return false;
    }

    get_balanced_signal_slots().fetch_add(1);

    auto& shared_ptr_info    = *static_cast<std::shared_ptr<Queue::queue_info_session_t>*>(data);
    auto& queue_info_session = *shared_ptr_info;

    auto dispatch_time = kernel_dispatch::get_dispatch_time(queue_info_session);

    kernel_dispatch::dispatch_complete(queue_info_session, dispatch_time);

    // Calls our internal callbacks to callers who need to be notified post
    // kernel execution.
    queue_info_session.queue.signal_callback([&](const auto& map) {
        for(const auto& [client_id, cb_pair] : map)
        {
            cb_pair.second(queue_info_session.queue,
                           queue_info_session.kernel_pkt,
                           shared_ptr_info,
                           queue_info_session.inst_pkt,
                           dispatch_time);
        }
    });

    if(queue_info_session.is_serialized)
    {
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->serializer(&queue_info_session.queue)
            .wlock([&](auto& serializer) {
                serializer.kernel_completion_signal(queue_info_session.queue);
            });
    }

    // Delete signals and packets, signal we have completed.
    if(queue_info_session.interrupt_signal.handle != 0u)
    {
#if !defined(NDEBUG)
        CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
            signals.erase(queue_info_session.interrupt_signal.handle);
        });
#endif
        hsa::get_core_table()->hsa_signal_store_screlease_fn(queue_info_session.interrupt_signal,
                                                             -1);
        hsa::get_core_table()->hsa_signal_destroy_fn(queue_info_session.interrupt_signal);
    }
    if(queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal.handle != 0u)
    {
        hsa::get_core_table()->hsa_signal_destroy_fn(
            queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal);
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

    queue_info_session.queue.async_complete();
    delete &shared_ptr_info;

    return false;
}

template <typename Integral = uint64_t>
constexpr Integral
bit_mask(int first, int last)
{
    assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
    size_t num_bits = last - first + 1;
    return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                               /* num_bits exceed the size of Integral */
                                               : ((Integral{1} << num_bits) - 1))
           << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
constexpr Integral
bit_extract(Integral x, int first, int last)
{
    return (x >> first) & bit_mask<Integral>(0, last - first);
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

    using callback_record_t = Queue::queue_info_session_t::callback_record_t;

    // unique sequence id for the dispatch
    static auto sequence_counter = std::atomic<rocprofiler_dispatch_id_t>{0};

    auto&& CreateBarrierPacket = [](hsa_signal_t*                    dependency_signal,
                                    hsa_signal_t*                    completion_signal,
                                    std::vector<rocprofiler_packet>& _packets) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
        if(dependency_signal != nullptr) barrier.dep_signal[0] = *dependency_signal;
        if(completion_signal != nullptr) barrier.completion_signal = *completion_signal;
        _packets.emplace_back(barrier);
    };

    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

    auto& queue = *static_cast<Queue*>(data);

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 ||
       (queue.get_notifiers() == 0 && context::get_active_contexts(context_filter).empty()))
    {
        writer(packets, pkt_count);
        return;
    }

    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               tracing_data_v);
    // these are for the services (dispatch counter collection, pc sampling, ATT) which use
    // the queue/queue_controller callback mechanism
    const auto queue_callback_context_filter = [](const context::context* ctx) {
        return (ctx->counter_collection || ctx->pc_sampler || ctx->dispatch_thread_trace);
    };

    for(const auto* itr : context::get_active_contexts(queue_callback_context_filter))
        tracing_data_v.external_correlation_ids.emplace(itr, tracing::empty_user_data);

    const auto* packets_arr         = static_cast<const rocprofiler_packet*>(packets);
    auto        transformed_packets = std::vector<rocprofiler_packet>{};

    // Searching accross all the packets given during this write
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            transformed_packets.emplace_back(packets_arr[i]);
            continue;
        }

        auto*                    corr_id      = context::get_latest_correlation_id();
        context::correlation_id* _corr_id_pop = nullptr;

        if(!corr_id)
        {
            constexpr auto ref_count = 1;
            corr_id                  = context::correlation_tracing_service::construct(ref_count);
            _corr_id_pop             = corr_id;
        }

        // increase the reference count to denote that this correlation id is being used in a kernel
        corr_id->add_ref_count();
        corr_id->add_kern_count();

        auto thr_id           = (corr_id) ? corr_id->thread_idx : common::get_tid();
        auto user_data        = rocprofiler_user_data_t{.value = 0};
        auto internal_corr_id = (corr_id) ? corr_id->internal : 0;
        auto ancestor_corr_id = (corr_id) ? corr_id->ancestor : 0;

        // if we constructed a correlation id, this decrements the reference count after the
        // underlying function returns
        auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop]() {
            if(_corr_id_pop)
            {
                context::pop_latest_correlation_id(_corr_id_pop);
                _corr_id_pop->sub_ref_count();
            }
        }};

        tracing::populate_external_correlation_ids(
            tracing_data_v.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
            internal_corr_id);

        queue.async_started();

        const auto     original_completion_signal = original_packet.completion_signal;
        const bool     existing_completion_signal = (original_completion_signal.handle != 0);
        const uint64_t kernel_id = code_object::get_kernel_id(original_packet.kernel_object);

        // Copy kernel pkt, copy is to allow for signal to be modified
        rocprofiler_packet kernel_pkt = packets_arr[i];
        // create our own signal that we can get a callback on. if there is an original completion
        // signal we will create a barrier packet, assign the original completion signal that that
        // barrier packet, and add it right after the kernel packet
        queue.create_signal(0, &kernel_pkt.kernel_dispatch.completion_signal);

        // computes the "size" based on the offset of reserved_padding field
        constexpr auto kernel_dispatch_info_rt_size =
            common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

        static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                      "failed to compute size field based on offset of reserved_padding field");

        auto dispatch_id     = ++sequence_counter;
        auto callback_record = callback_record_t{
            sizeof(callback_record_t),
            rocprofiler_timestamp_t{0},
            rocprofiler_timestamp_t{0},
            rocprofiler_kernel_dispatch_info_t{
                .size                 = kernel_dispatch_info_rt_size,
                .agent_id             = queue.get_agent().get_rocp_agent()->id,
                .queue_id             = queue.get_id(),
                .kernel_id            = kernel_id,
                .dispatch_id          = dispatch_id,
                .private_segment_size = kernel_pkt.kernel_dispatch.private_segment_size,
                .group_segment_size   = kernel_pkt.kernel_dispatch.group_segment_size,
                .workgroup_size   = rocprofiler_dim3_t{kernel_pkt.kernel_dispatch.workgroup_size_x,
                                                     kernel_pkt.kernel_dispatch.workgroup_size_y,
                                                     kernel_pkt.kernel_dispatch.workgroup_size_z},
                .grid_size        = rocprofiler_dim3_t{kernel_pkt.kernel_dispatch.grid_size_x,
                                                kernel_pkt.kernel_dispatch.grid_size_y,
                                                kernel_pkt.kernel_dispatch.grid_size_z},
                .reserved_padding = {0}}};

        {
            auto tracer_data = callback_record;
            tracing::execute_phase_enter_callbacks(tracing_data_v.callback_contexts,
                                                   thr_id,
                                                   internal_corr_id,
                                                   tracing_data_v.external_correlation_ids,
                                                   ancestor_corr_id,
                                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                                                   tracer_data);
        }

        // map all the external correlation ids (after enqueue enter phase) for all the contexts
        // captured by the info session
        tracing::update_external_correlation_ids(
            tracing_data_v.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

        // If there is a lot of contention for HSA signals, then schedule out the thread
        if(get_balanced_signal_slots().fetch_sub(1) <= 0)
        {
            sched_yield();
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }

        // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
        // along with an ID of the client we got the packet from (this will be returned via
        // completed_cb_t)
        auto inst_pkt = inst_pkt_t{};

        // True if any service (ATT,SPM,CC) requests this dispatch to be serialized
        bool bRequest_Serialize = false;

        // Signal callbacks that a kernel_pkt is being enqueued
        queue.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_pair] : map)
            {
                auto [packet, bSerial] = cb_pair.first(queue,
                                                       kernel_pkt,
                                                       kernel_id,
                                                       dispatch_id,
                                                       &user_data,
                                                       tracing_data_v.external_correlation_ids,
                                                       corr_id);
                bRequest_Serialize |= bSerial;
                if(packet) inst_pkt.push_back(std::make_pair(std::move(packet), client_id));
            }
        });

        bool inserted_before = false;
        if(bRequest_Serialize)
        {
            inserted_before = true;
            CHECK_NOTNULL(hsa::get_queue_controller())
                ->serializer(&queue)
                .wlock([&](auto& serializer) {
                    for(auto& s_pkt : serializer.kernel_dispatch(queue))
                        transformed_packets.emplace_back(s_pkt.ext_amd_aql_pm4);
                });
        }
        for(const auto& pkt_injection : inst_pkt)
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
            transformed_packets.emplace_back(pc_sampling::hsa::generate_marker_packet_for_kernel(
                corr_id, tracing_data_v.external_correlation_ids, dispatch_id));
        }
#endif

        // emplace the kernel packet
        transformed_packets.emplace_back(kernel_pkt);
        // If a profiling packet was inserted, wait for completion before executing the dispatch
        if(inserted_before)
            transformed_packets.back().kernel_dispatch.header |= 1 << HSA_PACKET_HEADER_BARRIER;

        // if the original completion signal exists, trigger it via a barrier packet
        if(existing_completion_signal)
        {
            auto barrier   = hsa_barrier_and_packet_t{};
            barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.header |= (1 << HSA_PACKET_HEADER_BARRIER);
            barrier.completion_signal = original_completion_signal;
            transformed_packets.emplace_back(barrier);
        }

        bool injected_end_pkt = false;
        for(const auto& pkt_injection : inst_pkt)
        {
            for(const auto& pkt : pkt_injection.first->after_krn_pkt)
            {
                transformed_packets.emplace_back(pkt);
                injected_end_pkt = true;
            }
        }

        auto completion_signal = hsa_signal_t{.handle = 0};
        auto interrupt_signal  = hsa_signal_t{.handle = 0};
        if(injected_end_pkt)
        {
            // Adding a barrier packet with the original packet's completion signal.
            queue.create_signal(0, &interrupt_signal);
            completion_signal                                            = interrupt_signal;
            transformed_packets.back().ext_amd_aql_pm4.completion_signal = interrupt_signal;
            CreateBarrierPacket(&interrupt_signal, &interrupt_signal, transformed_packets);
        }
        else
        {
            completion_signal = kernel_pkt.kernel_dispatch.completion_signal;
            get_core_table()->hsa_signal_store_screlease_fn(completion_signal, 0);
        }

        ROCP_FATAL_IF(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            << "get_kernel_id below might need to be updated";

        // Enqueue the signal into the handler. Will call completed_cb when
        // signal completes.

        {
            Queue::queue_info_session_t info_session{.queue            = queue,
                                                     .inst_pkt         = std::move(inst_pkt),
                                                     .interrupt_signal = interrupt_signal,
                                                     .tid              = thr_id,
                                                     .enqueue_ts       = common::timestamp_ns(),
                                                     .user_data        = user_data,
                                                     .correlation_id   = corr_id,
                                                     .kernel_pkt       = kernel_pkt,
                                                     .callback_record  = callback_record,
                                                     .tracing_data     = tracing_data_v,
                                                     .is_serialized    = bRequest_Serialize};

            auto shared = std::make_shared<Queue::queue_info_session_t>(std::move(info_session));

            queue.signal_async_handler(completion_signal,
                                       new std::shared_ptr<Queue::queue_info_session_t>(shared));

            auto tracer_data = callback_record;
            tracing::execute_phase_exit_callbacks(tracing_data_v.callback_contexts,
                                                  tracing_data_v.external_correlation_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                  ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                                                  tracer_data);
        }
    }

    // Command is only executed if GLOG_v=2 or higher, otherwise it is a no-op
    ROCP_TRACE << fmt::format(
        "QueueID {}: {}", queue.get_id().handle, fmt::join(transformed_packets, fmt::format(" ")));

    writer(transformed_packets.data(), transformed_packets.size());
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
            return (ctx->counter_collection || ctx->device_counter_collection ||
                    ctx->dispatch_thread_trace || ctx->device_thread_trace);
        }).empty())
    {
        CHECK(_agent.cpu_pool().handle != 0);
        CHECK(_agent.get_hsa_agent().handle != 0);

        // Set state of the queue to allow profiling
        aql::set_profiler_active_on_queue(
            _agent.cpu_pool(), _agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                hsa_signal_t completion;
                create_signal(0, &completion);
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

    create_signal(0, &ready_signal);
    create_signal(0, &block_signal);
    create_signal(0, &_active_kernels);
    _core_api.hsa_signal_store_screlease_fn(ready_signal, 0);
    _core_api.hsa_signal_store_screlease_fn(_active_kernels, 0);
    *queue = _intercept_queue;
}

Queue::~Queue()
{
    sync();
    _core_api.hsa_signal_destroy_fn(_active_kernels);
}

void
Queue::signal_async_handler(const hsa_signal_t& signal, void* data) const
{
#if !defined(NDEBUG)
    CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
        signals[signal.handle] = signal;
    });
#endif
    hsa_status_t status = _ext_api.hsa_amd_signal_async_handler_fn(
        signal, HSA_SIGNAL_CONDITION_EQ, -1, AsyncSignalHandler, data);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_async_handler failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);
}

void
Queue::create_signal(uint32_t attribute, hsa_signal_t* signal) const
{
    hsa_status_t status = _ext_api.hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);
}

void
Queue::sync() const
{
    if(_active_kernels.handle != 0u)
    {
        _core_api.hsa_signal_wait_relaxed_fn(
            _active_kernels, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}

void
Queue::register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb)
{
    _callbacks.wlock([&](auto& map) {
        ROCP_FATAL_IF(rocprofiler::common::get_val(map, id)) << "ID already exists!";
        _notifiers++;
        map[id] = std::make_pair(enqueue_cb, complete_cb);
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

std::string
Queue::to_string() const
{
    std::ostringstream oss;
    oss << "Queue{";

    // _notifiers
    oss << "_notifiers: " << _notifiers.load() << ", ";

    // _active_async_packets
    oss << "_active_async_packets: " << _active_async_packets.load() << ", ";

    // _callbacks (show count)
    oss << "_callbacks: {";
    _callbacks.rlock([&](const auto& callbacks) { oss << "count: " << callbacks.size(); });
    oss << "}, ";

    // _intercept_queue (show ID if not null)
    oss << "_intercept_queue: ";
    if(_intercept_queue)
    {
        oss << "{id: " << _intercept_queue->id << "}";
    }
    else
    {
        oss << "null";
    }
    oss << ", ";

    // _state
    oss << "_state: ";
    switch(_state)
    {
        case queue_state::normal: oss << "normal"; break;
        case queue_state::to_destroy: oss << "to_destroy"; break;
        case queue_state::done_destroy: oss << "done_destroy"; break;
        default: oss << "unknown"; break;
    }
    oss << ", ";

    // _active_kernels signal
    oss << "_active_kernels: {handle: " << _active_kernels.handle << "}, ";

    // block_signal and ready_signal with their values
    oss << "block_signal: {handle: " << block_signal.handle
        << ", value: " << _core_api.hsa_signal_load_relaxed_fn(block_signal) << "}, ";
    oss << "ready_signal: {handle: " << ready_signal.handle
        << ", value: " << _core_api.hsa_signal_load_relaxed_fn(ready_signal) << "}";
    oss << "}";
    return oss.str();
}

}  // namespace hsa
}  // namespace rocprofiler
