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

#include "queue_registration_controller.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"

void
init_logging()
{
    rocprofiler::common::init_logging("ROCPROFILER");
}

// ensure that logging is always initialized when library is loaded
bool init_logging_at_load = (init_logging(), true);

extern "C" {
int
rocprofiler_queue_set_api_table(
    const char* name,
    uint64_t    lib_version,
    uint64_t    lib_instance,
    void**      tables,
    uint64_t    num_tables) 
{
    ROCP_TRACE << "rocprofiler_queue_set_api_table called for api " << name;
    (void)lib_version; // unused
    (void)lib_instance; // unused

    if (std::string_view{name} != "hsa")
    {
        ROCP_ERROR << "rocprofiler_queue_set_api_table was called with a table other than HSA";
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    ROCP_ERROR_IF(num_tables > 1)
        << "rocprofiler expected HSA library to pass 1 API table, not " << num_tables;

    auto* hsa_api_table = static_cast<HsaApiTable*>(*tables);

    rocprofiler::hsa::queue_registration_controller_init(hsa_api_table);

    return 0;
}

int
rocprofiler_queue_export_all_queue_registrations(
    void* queue_registrations,
    uint64_t* num_queue_registrations)
{
    if (!queue_registrations && num_queue_registrations)
    {
        *num_queue_registrations = CHECK_NOTNULL(rocprofiler::hsa::get_queue_registration_controller())->get_all_queue_registrations().size();
        return 0;
    }

    CHECK_NOTNULL(queue_registrations);
    CHECK_NOTNULL(num_queue_registrations);
    auto qrs = CHECK_NOTNULL(rocprofiler::hsa::get_queue_registration_controller())->get_all_queue_registrations();
    auto qrs_out = reinterpret_cast<rocprofiler::hsa::queue_registration_export_t*>(queue_registrations);
    if (*num_queue_registrations < qrs.size())
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    for (const auto& qr : qrs)
    {
        qrs_out->agent = qr.second.agent;
        qrs_out->queue = qr.second.queue;
        ++qrs_out;
    }

    return 0;
}

int rocprofiler_queue_set_write_interceptor(
    hsa_queue_t* queue,
    rocprofiler::hsa::write_interceptor_t func,
    void* data
)
{
    auto& qrs = CHECK_NOTNULL(rocprofiler::hsa::get_queue_registration_controller())->get_all_queue_registrations();
    auto qr_pair = qrs.find(queue);
    if (qr_pair == qrs.end())
    {
        ROCP_ERROR << "couldn't find registration to set write interceptor for queue " << queue;
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    qr_pair->second.user_write_interceptor_func = func;
    qr_pair->second.user_write_interceptor_data = data;
    return 0;
}

int rocprofiler_queue_export_all_code_object_registrations(
    hsa_executable_t* executables,
    uint64_t* num_executables)
{
    if (!executables && num_executables)
    {
        *num_executables = CHECK_NOTNULL(rocprofiler::hsa::get_queue_registration_controller())->get_all_code_object_registrations().size();
        return 0;
    }

    CHECK_NOTNULL(executables);
    CHECK_NOTNULL(num_executables);
    auto cos = CHECK_NOTNULL(rocprofiler::hsa::get_queue_registration_controller())->get_all_code_object_registrations();
    auto cos_out = executables;
    if (*num_executables < cos.size())
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    std::copy(cos.begin(), cos.end(), cos_out);

    return 0;
}

int rocprofiler_queue_get_version()
{
    constexpr int ROCPROFILER_QUEUE_VERSION = 1;
    return ROCPROFILER_QUEUE_VERSION;
}
}

namespace rocprofiler {
namespace hsa {

void write_interceptor
    (const void* packets,
    uint64_t     pkt_count,
    uint64_t     unused,
    void*        data,
    hsa_amd_queue_intercept_packet_writer_t writer)
{
    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";
    auto queue = static_cast<hsa_queue_t*>(data);

    auto& queue_map = CHECK_NOTNULL(get_queue_registration_controller())->get_all_queue_registrations();
    auto queue_registration_pair = queue_map.find(queue);
    ROCP_FATAL_IF(queue_registration_pair == queue_map.end()) << "WriteInterceptor was not passed a valid queue";
    auto& queue_registration = queue_registration_pair->second;

    if (queue_registration.user_write_interceptor_func)
    {
        queue_registration.user_write_interceptor_func(
            packets,
            pkt_count,
            unused,
            queue_registration.user_write_interceptor_data,
            writer);
    } else {
        writer(packets, pkt_count);
    }
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
    auto* controller = CHECK_NOTNULL(get_queue_registration_controller());

    auto new_queue_registration = create_queue_registration(
        agent,
        size,
        type,
        callback,
        data,
        private_segment_size,
        group_segment_size,
        controller->get_core_table(),
        controller->get_ext_table(),
        write_interceptor);

    *queue = new_queue_registration.queue;
    controller->add_queue(new_queue_registration);
    ROCP_INFO << "created queue registration for HSA agent handle " << agent.handle;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    if(get_queue_registration_controller())
    {
        get_queue_registration_controller()->destroy_queue(hsa_queue);
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    auto qrc = CHECK_NOTNULL(get_queue_registration_controller());
    auto status = qrc->get_hsa_executable_freeze_fn()(executable, options);
    if (status)
    {
        return status;
    }
    qrc->add_code_object(executable);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    auto qrc = CHECK_NOTNULL(get_queue_registration_controller());
    auto status = qrc->get_hsa_executable_destroy_fn()(executable);
    if (status)
    {
        return status;
    }
    qrc->destroy_code_object(executable);
    return HSA_STATUS_SUCCESS;
}

// Initializes the QueueRegistrationController. This must be delayed until
// HSA has been inited.
void QueueRegistrationController::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    m_core_table = core_table;
    m_ext_table = ext_table;

    core_table.hsa_queue_create_fn  = hsa::create_queue;
    core_table.hsa_queue_destroy_fn = hsa::destroy_queue;

    m_hsa_executable_freeze_fn = core_table.hsa_executable_freeze_fn;
    core_table.hsa_executable_freeze_fn = hsa::executable_freeze;
    m_hsa_executable_destroy_fn = core_table.hsa_executable_destroy_fn;
    core_table.hsa_executable_destroy_fn = hsa::executable_destroy;
}

// Called to add a queue that was created by the user program
void QueueRegistrationController::add_queue(queue_registration_t queue_registration)
{
    m_queues[queue_registration.queue] = queue_registration;
}

void QueueRegistrationController::destroy_queue(hsa_queue_t* id)
{
    m_queues.erase(id);
}

void QueueRegistrationController::add_code_object(hsa_executable_t executable)
{
    ROCP_TRACE << "adding executable " << executable.handle;
    m_code_objects.emplace_back(executable);
}

void QueueRegistrationController::destroy_code_object(hsa_executable_t executable)
{
    ROCP_TRACE << "removing executable " << executable.handle;
    auto pred = [&](const hsa_executable_t& a)
    {
        return a.handle == executable.handle;
    };
    auto itr = std::find_if(m_code_objects.begin(), m_code_objects.end(), pred);
    if (itr == m_code_objects.end())
    {
        ROCP_INFO << "destroy_code_object could not find " << executable.handle;
        return;
    }
    m_code_objects.erase(itr);
}

QueueRegistrationController*
get_queue_registration_controller()
{
    static auto*& controller = common::static_object<QueueRegistrationController>::construct();
    return controller;
}

void
queue_registration_controller_init(HsaApiTable* table)
{
    CHECK_NOTNULL(get_queue_registration_controller())->init(*table->core_, *table->amd_ext_);
}

}}
