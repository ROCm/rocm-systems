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

#include <glog/logging.h>

namespace rocprofiler
{
namespace hsa
{
namespace
{
bool
AsyncSignalHandler(hsa_signal_value_t, void* data)
{
    if(!data) return true;
    auto& queue_info_session = *static_cast<Queue::queue_info_session_t*>(data);

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

    // Delete signals and packets, signal we have completed.
    if(queue_info_session.interrupt_signal.handle != 0u)
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.interrupt_signal);
    if(queue_info_session.kernel_pkt.completion_signal.handle != 0u)
    {
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.kernel_pkt.completion_signal);
    }
    queue_info_session.queue.async_complete();

    delete static_cast<Queue::queue_info_session_t*>(data);
    return false;
}

void
CreateBarrierPacket(const hsa_signal_t&                        packet_completion_signal,
                    std::vector<hsa_ext_amd_aql_pm4_packet_t>& transformed_packets)
{
    hsa_barrier_and_packet_t barrier{};
    barrier.header        = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.dep_signal[0] = packet_completion_signal;
    void* barrier_ptr     = &barrier;
    transformed_packets.emplace_back(*reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(barrier_ptr));
}

void
AddVendorSpecificPacket(const hsa_ext_amd_aql_pm4_packet_t&        packet,
                        std::vector<hsa_ext_amd_aql_pm4_packet_t>& transformed_packets,
                        const hsa_signal_t&                        packet_completion_signal)
{
    transformed_packets.emplace_back(packet).completion_signal = packet_completion_signal;
}
}  // namespace

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
    hsa_status_t status = _ext_api.hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed";
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
    Queue& queue_info = *static_cast<Queue*>(data);

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 || queue_info.get_notifiers() == 0)
    {
        writer(packets, pkt_count);
        return;
    }

    // hsa_ext_amd_aql_pm4_packet_t
    const hsa_ext_amd_aql_pm4_packet_t* packets_arr =
        static_cast<const hsa_ext_amd_aql_pm4_packet_t*>(packets);
    std::vector<hsa_ext_amd_aql_pm4_packet_t> transformed_packets;

    // Searching accross all the packets given during this write
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
        if(bit_extract(original_packet.header,
                       HSA_PACKET_HEADER_TYPE,
                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) !=
           HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            transformed_packets.emplace_back(packets_arr[i]);
            continue;
        }

        // Copy kernel pkt, copy is to allow for signal to be modified
        hsa_ext_amd_aql_pm4_packet_t kernel_pkt = packets_arr[i];
        queue_info.create_signal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &kernel_pkt.completion_signal);

        // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
        // along with an ID of the client we got the packet from (this will be returned via
        // CompletedCB)
        ClientID                   inst_pkt_id = -1;
        std::unique_ptr<AQLPacket> inst_pkt;

        // Signal callbacks that a kernel_pkt is being enqueued
        queue_info.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_pair] : map)
            {
                if(auto maybe_pkt = cb_pair.first(queue_info, client_id, kernel_pkt))
                {
                    LOG_IF(FATAL, inst_pkt)
                        << "We do not support two injections into the HSA queue";
                    inst_pkt    = std::move(maybe_pkt);
                    inst_pkt_id = client_id;
                }
            }
        });

        // Write instrumentation start packet (if one exists)
        if(inst_pkt)
        {
            hsa_signal_t dummy_signal{};
            dummy_signal.handle    = 0;
            inst_pkt->start.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->start, transformed_packets, dummy_signal);

            CreateBarrierPacket(inst_pkt->start.completion_signal, transformed_packets);
        }

        transformed_packets.emplace_back(kernel_pkt);

        // Make a copy of the original packet, adding its signal to a barrier
        // packet and create a new signal for it to get timestamps
        if(original_packet.completion_signal.handle != 0u)
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            hsa_ext_amd_aql_pm4_packet_t* __attribute__((__may_alias__)) pkt =
                (reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(&barrier));
            transformed_packets.emplace_back(*pkt).completion_signal =
                original_packet.completion_signal;
        }

        hsa_signal_t interrupt_signal{};
        // Adding a barrier packet with the original packet's completion signal.
        queue_info.create_signal(0, &interrupt_signal);

        if(inst_pkt)
        {
            hsa_signal_t dummy_signal{};
            dummy_signal.handle   = 0;
            inst_pkt->stop.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->stop, transformed_packets, dummy_signal);
            inst_pkt->read.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->read, transformed_packets, interrupt_signal);

            // Added Interrupt Signal with barrier and provided handler for it
            CreateBarrierPacket(interrupt_signal, transformed_packets);
        }
        else
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header            = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.completion_signal = interrupt_signal;
            hsa_ext_amd_aql_pm4_packet_t* __attribute__((__may_alias__)) pkt =
                (reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(&barrier));
            transformed_packets.emplace_back(*pkt);
        }

        // Enqueue the signal into the handler. Will call completed_cb when
        // signal completes.
        queue_info.async_started();
        queue_info.signal_async_handler(
            interrupt_signal,
            new Queue::queue_info_session_t{.queue            = queue_info,
                                            .inst_pkt         = std::move(inst_pkt),
                                            .inst_pkt_id      = inst_pkt_id,
                                            .kernel_pkt       = kernel_pkt,
                                            .interrupt_signal = interrupt_signal});
    }

    writer(transformed_packets.data(), transformed_packets.size());
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
           _ext_api.hsa_amd_queue_intercept_create_fn(_agent.get_agent(),
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
    *queue = _intercept_queue;
}

void
Queue::register_callback(ClientID id, QueueCB enqueue_cb, CompletedCB complete_cb)
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

void
QueueController::add_queue(hsa_queue_t* id, std::unique_ptr<Queue> queue)
{
    CHECK(queue);
    _callback_cache.wlock([&](auto& callbacks) {
        _queues.wlock([&](auto& map) {
            const auto agent_id = queue->get_agent().agent_t().id.handle;
            map[id]             = std::move(queue);
            for(const auto& [cbid, cb_tuple] : callbacks)
            {
                auto& [agent, qcb, ccb] = cb_tuple;
                if(agent.id.handle == agent_id)
                {
                    map[id]->register_callback(cbid, qcb, ccb);
                }
            }
        });
    });
}

void
QueueController::destory_queue(hsa_queue_t* id)
{
    _queues.wlock([&](auto& map) { map.erase(id); });
}

ClientID
QueueController::add_callback(const rocprofiler_agent_t& agent,
                              Queue::QueueCB             qcb,
                              Queue::CompletedCB         ccb)
{
    static std::atomic<ClientID> client_id = 1;
    ClientID                     return_id;
    _callback_cache.wlock([&](auto& cb_cache) {
        return_id           = client_id;
        cb_cache[client_id] = std::tuple(agent, qcb, ccb);
        client_id++;
        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                if(queue->get_agent().agent_t().id.handle == agent.id.handle)
                {
                    queue->register_callback(return_id, qcb, ccb);
                }
            }
        });
    });
    return return_id;
}

void
QueueController::remove_callback(ClientID id)
{
    _callback_cache.wlock([&](auto& cb_cache) {
        cb_cache.erase(id);
        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                queue->remove_callback(id);
            }
        });
    });
}

// HSA Intercept Functions (create_queue/destroy_queue)
hsa_status_t
create_queue(hsa_agent_t        agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             hsa_queue_t** queue)
{
    for(const auto& [_, agent_info] : get_queue_controller().get_supported_agents())
    {
        if(agent_info.get_agent().handle == agent.handle)
        {
            auto new_queue = std::make_unique<Queue>(agent_info,
                                                     size,
                                                     type,
                                                     callback,
                                                     data,
                                                     private_segment_size,
                                                     group_segment_size,
                                                     get_queue_controller().get_core_table(),
                                                     get_queue_controller().get_ext_table(),
                                                     queue);
            get_queue_controller().add_queue(*queue, std::move(new_queue));
            return HSA_STATUS_SUCCESS;
        }
    }
    LOG(FATAL) << "Could not find agent - " << agent.handle;
    return HSA_STATUS_ERROR_FATAL;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    get_queue_controller().destory_queue(hsa_queue);
    return HSA_STATUS_SUCCESS;
}

void
QueueController::Init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    _core_table = core_table;
    _ext_table  = ext_table;

    core_table.hsa_queue_create_fn  = create_queue;
    core_table.hsa_queue_destroy_fn = destroy_queue;

    // Generate supported agents
    rocprofiler_query_available_agents(
        [](const rocprofiler_agent_t** agents, size_t num_agents, void* user_data) {
            CHECK(user_data);
            QueueController& queue = *reinterpret_cast<QueueController*>(user_data);
            for(size_t i = 0; i < num_agents; i++)
            {
                const auto& agent = *agents[i];
                if(agent.type != ROCPROFILER_AGENT_TYPE_GPU) continue;
                try
                {
                    queue.get_supported_agents().emplace(
                        i, AgentCache{agent, i, queue.get_core_table(), queue.get_ext_table()});
                } catch(std::runtime_error& error)
                {
                    LOG(ERROR) << fmt::format("GPU Agent Construction Failed (HSA queue will not "
                                              "be intercepted): {} ({})",
                                              agent.id.handle,
                                              error.what());
                }
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_t),
        this);
}

QueueController&
get_queue_controller()
{
    static QueueController controller;
    return controller;
}

void
queue_controller_init(HsaApiTable* table)
{
    get_queue_controller().Init(*table->core_, *table->amd_ext_);
}

}  // namespace hsa
}  // namespace rocprofiler
