/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include "lib/rocprofiler/hsa/queue.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/buffer.hpp"
#include "lib/rocprofiler/context/context.hpp"
#include "lib/rocprofiler/hsa/code_object.hpp"

#include <glog/logging.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <rocprofiler/fwd.h>

#include <atomic>
#include <chrono>
#include <thread>

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

#if defined(ROCPROFILER_CI)
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(FATAL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(FATAL)
#else
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(NON_CI_LEVEL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(NON_CI_LEVEL)
#endif

namespace rocprofiler
{
namespace hsa
{
namespace
{
common::active_capacity_gate&
signal_limiter()
{
    // Limit the maximun number of HSA signals created.
    // There is a hard limit to the maximum that can exist.
    static common::active_capacity_gate _gate{96};
    return _gate;
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->buffered_tracer &&
            (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) ||
             ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)));
}

bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    // LOG(ERROR) << "signal value is " << signal_v;

    if(!data) return true;
    auto& queue_info_session = *static_cast<Queue::queue_info_session_t*>(data);

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = queue_info_session.correlation_id;
    // get the contexts that were active when the signal was created
    const auto& ctxs = queue_info_session.contexts;
    if(!ctxs.empty())
    {
        // only do the following work if there are contexts that require this info
        const auto* _rocp_agent      = queue_info_session.queue.get_agent().get_rocp_agent();
        auto        _hsa_agent       = queue_info_session.queue.get_agent().get_hsa_agent();
        auto        _queue_id        = queue_info_session.queue.get_id();
        auto        _signal          = queue_info_session.interrupt_signal;
        auto        _kern_id         = queue_info_session.kernel_id;
        const auto& _extern_corr_ids = queue_info_session.extern_corr_ids;

        auto dispatch_time = hsa_amd_profiling_dispatch_time_t{};
        auto dispatch_time_status =
            queue_info_session.queue.ext_api().hsa_amd_profiling_get_dispatch_time_fn(
                _hsa_agent, _signal, &dispatch_time);

        // if we encounter this in CI, it will cause test to fail
        ROCP_CI_LOG_IF(
            ERROR,
            dispatch_time_status == HSA_STATUS_SUCCESS && dispatch_time.end < dispatch_time.start)
            << "hsa_amd_profiling_get_dispatch_time for kernel_id=" << _kern_id
            << " on rocprofiler_agent=" << _rocp_agent->id.handle
            << " returned dispatch times where the end time (" << dispatch_time.end
            << ") was less than the start time (" << dispatch_time.start << ")";

        // try to extract the async copy time. this will return HSA_STATUS_ERROR if there
        // is not an async copy agent associated with the signal so we just predicate
        // putting something into the buffer based on whether or not
        // hsa_amd_profiling_get_async_copy_time returns HSA_STATUS_SUCCESS.
        auto copy_time = hsa_amd_profiling_async_copy_time_t{};
        auto copy_time_status =
            queue_info_session.queue.ext_api().hsa_amd_profiling_get_async_copy_time_fn(_signal,
                                                                                        &copy_time);

        // if we encounter this in CI, it will cause test to fail
        ROCP_CI_LOG_IF(ERROR,
                       copy_time_status == HSA_STATUS_SUCCESS && copy_time.end < copy_time.start)
            << "hsa_amd_profiling_get_async_copy_time for kernel_id=" << _kern_id
            << " on rocprofiler_agent=" << _rocp_agent->id.handle
            << " returned async times where the end time (" << copy_time.end
            << ") was less than the start time (" << copy_time.start << ")";

        for(const auto* itr : ctxs)
        {
            auto* _buffer = buffer::get_buffer(
                itr->buffered_tracer->buffer_data.at(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH));

            // go ahead and create the correlation id value since we expect at least one of these
            // domains will require it
            auto _corr_id_v =
                rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
            if(_corr_id)
            {
                _corr_id_v.internal = _corr_id->internal;
                _corr_id_v.external = _extern_corr_ids.at(itr);
            }

            if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH))
            {
                if(dispatch_time_status == HSA_STATUS_SUCCESS)
                {
                    const auto& dispatch_packet = queue_info_session.kernel_pkt.kernel_dispatch;

                    auto record = rocprofiler_buffer_tracing_kernel_dispatch_record_t{
                        sizeof(rocprofiler_buffer_tracing_kernel_dispatch_record_t),
                        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                        _corr_id_v,
                        dispatch_time.start,
                        dispatch_time.end,
                        _rocp_agent->id,
                        _queue_id,
                        _kern_id,
                        dispatch_packet.private_segment_size,
                        dispatch_packet.group_segment_size,
                        rocprofiler_dim3_t{dispatch_packet.workgroup_size_x,
                                           dispatch_packet.workgroup_size_y,
                                           dispatch_packet.workgroup_size_z},
                        rocprofiler_dim3_t{dispatch_packet.grid_size_x,
                                           dispatch_packet.grid_size_y,
                                           dispatch_packet.grid_size_z}};

                    _buffer->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING,
                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                     record);
                }
            }

            if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY))
            {
                if(copy_time_status == HSA_STATUS_SUCCESS)
                {
                    auto record = rocprofiler_buffer_tracing_memory_copy_record_t{
                        sizeof(rocprofiler_buffer_tracing_memory_copy_record_t),
                        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                        _corr_id_v,
                        copy_time.start,
                        copy_time.end,
                        _rocp_agent->id,
                        _queue_id,
                        _kern_id};

                    _buffer->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING,
                                     ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                     record);
                }
            }
        }
    }

    // Calls our internal callbacks to callers who need to be notified post
    // kernel execution.
    queue_info_session.queue.signal_callback([&](const auto& map) {
        for(const auto& [client_id, cb_pair] : map)
        {
            // If this is the client that gave us the AQLPacket,
            // return it to that client otherwise notify.
            if(queue_info_session.inst_pkt_id == client_id)
            {
                cb_pair.second(queue_info_session.queue,
                               client_id,
                               queue_info_session.kernel_pkt,
                               std::move(queue_info_session.inst_pkt));
            }
            else
            {
                cb_pair.second(
                    queue_info_session.queue, client_id, queue_info_session.kernel_pkt, nullptr);
            }
        }
    });

    size_t signals_to_remove = 0;
    // Delete signals and packets, signal we have completed.
    if(queue_info_session.interrupt_signal.handle != 0u)
    {
        signals_to_remove++;
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.interrupt_signal);
    }
    if(queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal.handle != 0u)
    {
        signals_to_remove++;
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal);
    }
    if(signals_to_remove > 0)
    {
        signal_limiter().remove_active(signals_to_remove);
    }
    queue_info_session.queue.async_complete();

    if(_corr_id)
    {
        LOG_IF(FATAL, _corr_id->ref_count.load() == 0)
            << "reference counter for correlation id " << _corr_id->internal << " from thread "
            << _corr_id->thread_idx << " has no reference count";
        _corr_id->ref_count.fetch_sub(1);
    }

    delete static_cast<Queue::queue_info_session_t*>(data);
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
    using context_array_t = Queue::context_array_t;

    auto&& AddVendorSpecificPacket = [](hsa_ext_amd_aql_pm4_packet_t     _packet,
                                        hsa_signal_t                     _signal,
                                        std::vector<rocprofiler_packet>& _packets) {
        _packets.emplace_back(_packet).ext_amd_aql_pm4.completion_signal = _signal;
    };

    auto&& CreateBarrierPacket = [](hsa_signal_t                     _signal,
                                    std::vector<rocprofiler_packet>& _packets) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header        = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.dep_signal[0] = _signal;
        _packets.emplace_back(barrier);
    };

    LOG_IF(FATAL, data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

    static thread_local auto ctxs = context_array_t{};
    context::get_active_contexts(ctxs, context_filter);

    auto& queue = *static_cast<Queue*>(data);

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 || (queue.get_notifiers() == 0 && ctxs.empty()))
    {
        writer(packets, pkt_count);
        return;
    }

    auto  thr_id  = common::get_tid();
    auto* corr_id = context::get_latest_correlation_id();

    // use thread-local value to reuse allocation
    static thread_local auto extern_corr_ids_tl =
        Queue::queue_info_session_t::external_corr_id_map_t{};

    // increase the reference count to denote that this correlation id is being used in a kernel
    if(corr_id)
    {
        extern_corr_ids_tl.clear();  // clear it so that it only contains the current contexts
        extern_corr_ids_tl.reserve(ctxs.size());  // reserve for performance
        for(const auto* ctx : ctxs)
            extern_corr_ids_tl.emplace(ctx,
                                       ctx->correlation_tracer.external_correlator.get(thr_id));
        corr_id->ref_count.fetch_add(1);
    }

    // move to local variable
    auto extern_corr_ids = std::move(extern_corr_ids_tl);

    // hsa_ext_amd_aql_pm4_packet_t
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

        // Copy kernel pkt, copy is to allow for signal to be modified
        rocprofiler_packet kernel_pkt = packets_arr[i];
        queue.create_signal(HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                            &kernel_pkt.ext_amd_aql_pm4.completion_signal);

        // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
        // along with an ID of the client we got the packet from (this will be returned via
        // completed_cb_t)
        ClientID                   inst_pkt_id = -1;
        std::unique_ptr<AQLPacket> inst_pkt;

        // Signal callbacks that a kernel_pkt is being enqueued
        queue.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_pair] : map)
            {
                if(auto maybe_pkt = cb_pair.first(queue, client_id, kernel_pkt))
                {
                    LOG_IF(FATAL, inst_pkt)
                        << "We do not support two injections into the HSA queue";
                    inst_pkt    = std::move(maybe_pkt);
                    inst_pkt_id = client_id;
                }
            }
        });

        constexpr auto dummy_signal = hsa_signal_t{.handle = 0};

        // Write instrumentation start packet (if one exists)
        if(inst_pkt)
        {
            inst_pkt->start.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->start, dummy_signal, transformed_packets);
            CreateBarrierPacket(inst_pkt->start.completion_signal, transformed_packets);
        }

        transformed_packets.emplace_back(kernel_pkt);

        // Make a copy of the original packet, adding its signal to a barrier
        // packet and create a new signal for it to get timestamps
        if(original_packet.completion_signal.handle != 0u)
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header            = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.completion_signal = original_packet.completion_signal;
            transformed_packets.emplace_back(barrier);
        }

        hsa_signal_t interrupt_signal{};
        // Adding a barrier packet with the original packet's completion signal.
        queue.create_signal(0, &interrupt_signal);

        if(inst_pkt)
        {
            inst_pkt->stop.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->stop, dummy_signal, transformed_packets);
            inst_pkt->read.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->read, interrupt_signal, transformed_packets);

            // Added Interrupt Signal with barrier and provided handler for it
            CreateBarrierPacket(interrupt_signal, transformed_packets);
        }
        else
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header            = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.completion_signal = interrupt_signal;
            transformed_packets.emplace_back(barrier);
        }

        LOG_IF(FATAL, packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            << "get_kernel_id below might need to be updated";
        uint64_t kernel_id = get_kernel_id(kernel_pkt.kernel_dispatch.kernel_object);

        // Enqueue the signal into the handler. Will call completed_cb when
        // signal completes.
        queue.async_started();
        queue.signal_async_handler(
            interrupt_signal,
            new Queue::queue_info_session_t{.queue            = queue,
                                            .inst_pkt         = std::move(inst_pkt),
                                            .inst_pkt_id      = inst_pkt_id,
                                            .interrupt_signal = interrupt_signal,
                                            .tid              = thr_id,
                                            .kernel_id        = kernel_id,
                                            .correlation_id   = corr_id,
                                            .kernel_pkt       = kernel_pkt,
                                            .contexts         = ctxs,
                                            .extern_corr_ids  = extern_corr_ids});
    }

    writer(transformed_packets.data(), transformed_packets.size());
}
}  // namespace

Queue::~Queue()
{
    // Potentially replace with condition variable at some point
    // but performance may not matter here.
    while(_active_async_packets.load(std::memory_order_relaxed) > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

void
Queue::signal_async_handler(const hsa_signal_t& signal, Queue::queue_info_session_t* data) const
{
    hsa_status_t status = _ext_api.hsa_amd_signal_async_handler_fn(
        signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandler, static_cast<void*>(data));
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_async_handler failed";
}

void
Queue::create_signal(uint32_t attribute, hsa_signal_t* signal) const
{
    signal_limiter().add_active(1);
    hsa_status_t status = _ext_api.hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed";
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
    LOG_IF(FATAL,
           _ext_api.hsa_amd_queue_intercept_create_fn(_agent.get_hsa_agent(),
                                                      size,
                                                      type,
                                                      callback,
                                                      data,
                                                      private_segment_size,
                                                      group_segment_size,
                                                      &_intercept_queue) != HSA_STATUS_SUCCESS)
        << "Could not create intercept queue";

    LOG_IF(FATAL,
           _ext_api.hsa_amd_profiling_set_profiler_enabled_fn(_intercept_queue, true) !=
               HSA_STATUS_SUCCESS)
        << "Could not setup intercept profiler";

    LOG_IF(FATAL,
           _ext_api.hsa_amd_queue_intercept_register_fn(_intercept_queue, WriteInterceptor, this))
        << "Could not register interceptor";

    bool enable_async_copy = false;
    for(const auto& itr : context::get_registered_contexts())
    {
        if(itr->buffered_tracer &&
           itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY))
            enable_async_copy = true;
    }

    if(enable_async_copy)
    {
        LOG_IF(FATAL, _ext_api.hsa_amd_profiling_async_copy_enable_fn(true) != HSA_STATUS_SUCCESS)
            << "Could not enable async copy timing";
    }

    *queue = _intercept_queue;
}

void
Queue::register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb)
{
    _callbacks.wlock([&](auto& map) {
        LOG_IF(FATAL, rocprofiler::common::get_val(map, id)) << "ID already exists!";
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
}  // namespace hsa
}  // namespace rocprofiler
