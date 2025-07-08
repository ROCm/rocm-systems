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
}

void
init_logging()
{
    rocprofiler::common::init_logging("ROCPROFILER");
}

// ensure that logging is always initialized when library is loaded
bool init_logging_at_load = (init_logging(), true);

namespace rocprofiler {
namespace hsa {

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

    auto new_queue_registration = std::make_shared<QueueRegistration>(
        agent,
        size,
        type,
        callback,
        data,
        private_segment_size,
        group_segment_size,
        controller->get_core_table(),
        controller->get_ext_table(),
        queue);

    controller->add_queue(*queue, new_queue_registration);
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

// Initializes the QueueRegistrationController. This must be delayed until
// HSA has been inited.
void QueueRegistrationController::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    _core_table = core_table;
    _ext_table = ext_table;

    core_table.hsa_queue_create_fn  = hsa::create_queue;
    core_table.hsa_queue_destroy_fn = hsa::destroy_queue;
}

// Called to add a queue that was created by the user program
void QueueRegistrationController::add_queue(hsa_queue_t* id, std::shared_ptr<QueueRegistration> queue_registration)
{
    _queues[id] = queue_registration;
}

void QueueRegistrationController::destroy_queue(hsa_queue_t* id)
{
    _queues.erase(id);
}

const std::unordered_map<hsa_queue_t*, std::shared_ptr<QueueRegistration>> QueueRegistrationController::get_all_registrations()
{
    return _queues;
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
