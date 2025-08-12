// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "details/queue_registration.hpp"
#include "queue_registration.hpp"

#include "lib/common/static_object.hpp"

namespace rocprofiler
{
namespace prestore
{

// This is the prestore library's WriteInterceptor that is provided to HSA.
// Since the interceptor function cannot be changed later, this shim is provided immediately.
// This shim will look up the associated queue and execute the user's write intercept function
// if it has been previously provided. Otherwise, it will execute the packet write normally.
void
write_interceptor(const void*                             packets,
                  uint64_t                                pkt_count,
                  uint64_t                                unused,
                  void*                                   data,
                  hsa_amd_queue_intercept_packet_writer_t writer)
{
    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";
    auto queue = static_cast<hsa_queue_t*>(data);

    auto& queue_map               = CHECK_NOTNULL(get_queue_registration())->get_all_queues();
    auto  queue_registration_pair = queue_map.find(queue);
    ROCP_FATAL_IF(queue_registration_pair == queue_map.end())
        << "WriteInterceptor was not passed a valid queue";
    auto& queue_registration = queue_registration_pair->second;

    if(queue_registration.user_write_interceptor_func)
    {
        queue_registration.user_write_interceptor_func(
            packets, pkt_count, unused, queue_registration.user_write_interceptor_data, writer);
    }
    else
    {
        writer(packets, pkt_count);
    }
}

queue_prestore_t
create_queue_prestore(hsa_agent_t        agent,
                      uint32_t           size,
                      hsa_queue_type32_t type,
                      callback_t         callback,
                      void*              data,
                      uint32_t           private_segment_size,
                      uint32_t           group_segment_size,
                      CoreApiTable       core_api,
                      AmdExtTable        ext_api)
{
    (void) core_api;  // unused

    hsa_queue_t* queue = nullptr;

    ROCP_HSA_TABLE_CALL(
        FATAL,
        ext_api.hsa_amd_queue_intercept_create_fn(
            agent, size, type, callback, data, private_segment_size, group_segment_size, &queue))
        << "Could not create intercept queue";

    ROCP_HSA_TABLE_CALL(FATAL, ext_api.hsa_amd_profiling_set_profiler_enabled_fn(queue, true))
        << "Could not setup intercept profiler";

    // Pass hsa_queue_t* as user data, used to identify
    ROCP_HSA_TABLE_CALL(
        FATAL, ext_api.hsa_amd_queue_intercept_register_fn(queue, write_interceptor, queue))
        << "Could not register interceptor";

    queue_prestore_t queue_prestore{};
    queue_prestore.agent                       = agent;
    queue_prestore.queue                       = queue;
    queue_prestore.user_write_interceptor_func = nullptr;
    queue_prestore.user_write_interceptor_data = nullptr;

    return queue_prestore;
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
    auto* registration = CHECK_NOTNULL(get_queue_registration());

    auto new_queue = create_queue_prestore(agent,
                                           size,
                                           type,
                                           callback,
                                           data,
                                           private_segment_size,
                                           group_segment_size,
                                           registration->get_core_table(),
                                           registration->get_ext_table());

    *queue = new_queue.queue;
    registration->add_queue(new_queue);
    ROCP_INFO << "created queue prestore for HSA agent handle " << agent.handle;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    if(get_queue_registration())
    {
        CHECK_NOTNULL(get_queue_registration())->remove_queue(hsa_queue);
    }
    return HSA_STATUS_SUCCESS;
}

void
QueueRegistration::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    m_core_table = core_table;
    m_ext_table  = ext_table;

    core_table.hsa_queue_create_fn  = prestore::create_queue;
    core_table.hsa_queue_destroy_fn = prestore::destroy_queue;
}

void
QueueRegistration::add_queue(queue_prestore_t queue_prestore)
{
    m_queues[queue_prestore.queue] = queue_prestore;
}

void
QueueRegistration::remove_queue(hsa_queue_t* id)
{
    m_queues.erase(id);
}

QueueRegistration*
get_queue_registration()
{
    static auto*& registration = common::static_object<QueueRegistration>::construct();
    return registration;
}

void
queue_registration_init(HsaApiTable* table)
{
    CHECK_NOTNULL(get_queue_registration())->init(*table->core_, *table->amd_ext_);
}

}  // namespace prestore
}  // namespace rocprofiler

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_prestore_export_all_queues(rocprofiler::prestore::queue_prestore_export_t* queues,
                                       uint64_t*                                       num_queues)
{
    if(!queues && num_queues)
    {
        *num_queues =
            CHECK_NOTNULL(rocprofiler::prestore::get_queue_registration())->get_all_queues().size();
        return ROCPROFILER_STATUS_SUCCESS;
    }

    CHECK_NOTNULL(queues);
    CHECK_NOTNULL(num_queues);
    auto q     = CHECK_NOTNULL(rocprofiler::prestore::get_queue_registration())->get_all_queues();
    auto q_out = reinterpret_cast<rocprofiler::prestore::queue_prestore_export_t*>(queues);
    if(*num_queues < q.size())
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    for(const auto& qe : q)
    {
        q_out->agent = qe.second.agent;
        q_out->queue = qe.second.queue;
        ++q_out;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

int
rocprofiler_prestore_set_write_interceptor(hsa_queue_t* queue, write_interceptor_t func, void* data)
{
    auto& qrs = CHECK_NOTNULL(rocprofiler::prestore::get_queue_registration())->get_all_queues();
    auto  qr_pair = qrs.find(queue);
    if(qr_pair == qrs.end())
    {
        ROCP_ERROR << "couldn't find registration to set write interceptor for queue " << queue;
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    qr_pair->second.user_write_interceptor_func = func;
    qr_pair->second.user_write_interceptor_data = data;
    return 0;
}

ROCPROFILER_EXTERN_C_FINI
