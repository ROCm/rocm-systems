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

#include "queue_registration.h"
#include "queue_registration.hpp"
#include "table.hpp"

#include "lib/common/static_object.hpp"

#include <mutex>

namespace
{
using callback_t = void (*)(hsa_status_t status, hsa_queue_t* source, void* data);

struct queue_entry_t
{
    hsa_agent_t         agent                       = hsa_agent_t{};
    write_interceptor_t user_write_interceptor_func = nullptr;
    void*               user_write_interceptor_data = nullptr;
};

using queue_collection_t = std::unordered_map<hsa_queue_t*, queue_entry_t>;

struct queue_registration_t
{
    // guards access to both queues collection
    std::mutex         queues_mutex;
    queue_collection_t queues;

    decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn) hsa_amd_queue_intercept_create_fn =
        nullptr;
    decltype(AmdExtTable::hsa_amd_profiling_set_profiler_enabled_fn)
        hsa_amd_profiling_set_profiler_enabled_fn = nullptr;
    decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn) hsa_amd_queue_intercept_register_fn =
        nullptr;
    decltype(CoreApiTable::hsa_status_string_fn) hsa_status_string_fn = nullptr;
};

queue_registration_t*
get_queue_registration()
{
    static auto*& registration =
        rocprofiler::common::static_object<queue_registration_t>::construct();
    return registration;
}

std::string_view
get_hsa_status_string(hsa_status_t _status)
{
    auto* registration = CHECK_NOTNULL(get_queue_registration());

    const char* _status_msg = nullptr;
    return (CHECK_NOTNULL(registration->hsa_status_string_fn)(_status, &_status_msg) ==
                HSA_STATUS_SUCCESS &&
            _status_msg)
               ? std::string_view{_status_msg}
               : std::string_view{"(unknown HSA error)"};
}

#define ROCP_ATTACH_HSA_TABLE_CALL(SEVERITY, EXPR)                                                 \
    auto ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__) = (EXPR);                            \
    ROCP_##SEVERITY##_IF(ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__) !=                   \
                         HSA_STATUS_SUCCESS)                                                       \
        << #EXPR << " returned non-zero status code "                                              \
        << ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__)                                    \
        << " :: " << get_hsa_status_string(ROCPROFILER_VARIABLE(rocp_hsa_table_call_, __LINE__))   \
        << " "

// This is the attach library's WriteInterceptor that is provided to HSA.
// Since the interceptor function cannot be changed later, this shim is provided immediately upon
// queue creation. This shim's user data is a reference to the queue_entry_t for this queue, which
// will then by cast and used to call the user write interceptor if it is non-null.
void
write_interceptor(const void*                             packets,
                  uint64_t                                pkt_count,
                  uint64_t                                unused,
                  void*                                   data,
                  hsa_amd_queue_intercept_packet_writer_t writer)
{
    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a valid pointer";
    const auto* entry = static_cast<const queue_entry_t*>(data);

    if(entry->user_write_interceptor_func)
    {
        entry->user_write_interceptor_func(
            packets, pkt_count, unused, entry->user_write_interceptor_data, writer);
    }
    else
    {
        writer(packets, pkt_count);
    }
}

// HSA Intercept Functions (create_queue/destroy_queue)
hsa_status_t
create_queue(hsa_agent_t        agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             callback_t         callback,
             void*              data,
             uint32_t           private_segment_size,
             uint32_t           group_segment_size,
             hsa_queue_t**      queue)
{
    auto* registration = CHECK_NOTNULL(get_queue_registration());

    // Create new queue in HSA
    hsa_queue_t* new_queue = nullptr;
    ROCP_FATAL_IF(!registration->hsa_amd_queue_intercept_create_fn ||
                  !registration->hsa_amd_profiling_set_profiler_enabled_fn ||
                  !registration->hsa_amd_queue_intercept_register_fn ||
                  !registration->hsa_status_string_fn)
        << "Queue registration was not initialized before create queue was called!";

    ROCP_ATTACH_HSA_TABLE_CALL(FATAL,
                               registration->hsa_amd_queue_intercept_create_fn(agent,
                                                                               size,
                                                                               type,
                                                                               callback,
                                                                               data,
                                                                               private_segment_size,
                                                                               group_segment_size,
                                                                               &new_queue))
        << "Could not create intercept queue";

    ROCP_ATTACH_HSA_TABLE_CALL(
        FATAL, registration->hsa_amd_profiling_set_profiler_enabled_fn(new_queue, true))
        << "Could not setup intercept profiler";

    // Create and insert our queue's data entry now, as we need to provide a reference to it for the
    // write_interceptor
    queue_entry_t entry{};
    entry.agent = agent;

    {
        std::lock_guard lg(registration->queues_mutex);
        ROCP_FATAL_IF(registration->queues.count(new_queue) > 0)
            << "Queue registration already contains an entry for new queue handle " << new_queue;
        registration->queues.insert({new_queue, entry});
    }
    auto* write_interceptor_data = &(registration->queues.at(new_queue));

    // Pass queue_entry_t* as user data, used to directly call the user's write interceptor
    ROCP_ATTACH_HSA_TABLE_CALL(FATAL,
                               registration->hsa_amd_queue_intercept_register_fn(
                                   new_queue, write_interceptor, write_interceptor_data))
        << "Could not register interceptor";

    *queue = new_queue;

    ROCP_INFO << "created attach queue for HSA agent handle " << agent.handle;

    auto* attach_table = rocprofiler::attach::get_dispatch_table();
    if(attach_table->rocprofiler_attach_notify_new_queue)
    {
        attach_table->rocprofiler_attach_notify_new_queue(new_queue, agent, nullptr);
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    auto* registration = get_queue_registration();
    if(registration)
    {
        std::lock_guard lg(registration->queues_mutex);
        size_t          erase_count = registration->queues.erase(hsa_queue);
        ROCP_WARNING_IF(erase_count == 0)
            << "Destroy queue was called for a handle that was not in queues: " << hsa_queue;
    }
    return HSA_STATUS_SUCCESS;
}

int
iterate_all_queues(rocprof_attach_queue_iterator_t func, void* user_data)
{
    auto* registration = CHECK_NOTNULL(get_queue_registration());

    std::lock_guard lg(registration->queues_mutex);
    for(const auto& qr_pair : registration->queues)
    {
        func(qr_pair.first, qr_pair.second.agent, user_data);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

int
set_write_interceptor(hsa_queue_t* queue, write_interceptor_t func, void* data)
{
    auto* registration = CHECK_NOTNULL(get_queue_registration());
    auto  qr_pair      = registration->queues.find(queue);
    if(qr_pair == registration->queues.end())
    {
        ROCP_ERROR << "couldn't find registration to set write interceptor for queue " << queue;
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    qr_pair->second.user_write_interceptor_func = func;
    qr_pair->second.user_write_interceptor_data = data;
    return 0;
}

}  // namespace

namespace rocprofiler
{
namespace attach
{
void
queue_registration_init(HsaApiTable* table)
{
    ROCP_TRACE << "Initializing Queue Registration";
    auto* registration = CHECK_NOTNULL(get_queue_registration());

    CoreApiTable& core_table = *table->core_;

    core_table.hsa_queue_create_fn  = create_queue;
    core_table.hsa_queue_destroy_fn = destroy_queue;

    registration->hsa_amd_queue_intercept_create_fn =
        *table->amd_ext_->hsa_amd_queue_intercept_create_fn;
    registration->hsa_amd_profiling_set_profiler_enabled_fn =
        *table->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn;
    registration->hsa_amd_queue_intercept_register_fn =
        *table->amd_ext_->hsa_amd_queue_intercept_register_fn;
    registration->hsa_status_string_fn = *table->core_->hsa_status_string_fn;
}

}  // namespace attach
}  // namespace rocprofiler

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_attach_iterate_all_queues(rocprof_attach_queue_iterator_t func, void* data)
{
    return iterate_all_queues(func, data);
}

int
rocprofiler_attach_set_write_interceptor(hsa_queue_t* queue, write_interceptor_t func, void* data)
{
    return set_write_interceptor(queue, func, data);
}

ROCPROFILER_EXTERN_C_FINI
