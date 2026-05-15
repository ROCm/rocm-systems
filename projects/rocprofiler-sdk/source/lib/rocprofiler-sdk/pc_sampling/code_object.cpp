// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/code_object.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/container/operators.hpp"
#    include "lib/common/logging.hpp"
#    include "lib/common/static_object.hpp"
#    include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"

#    include <rocprofiler-sdk/fwd.h>
#    include <rocprofiler-sdk/pc_sampling.h>
#    include <rocprofiler-sdk/cxx/operators.hpp>

#    include <glog/logging.h>
#    include <hsa/hsa.h>
#    include <hsa/hsa_api_trace.h>
#    include <hsa/hsa_ven_amd_loader.h>

namespace rocprofiler
{
namespace pc_sampling
{
namespace code_object
{
namespace
{
auto&
get_freeze_function()
{
    static decltype(::hsa_executable_freeze)* _v = nullptr;
    return _v;
}

auto&
get_destroy_function()
{
    static decltype(::hsa_executable_destroy)* _v = nullptr;
    return _v;
}

/**
 * @brief Flush internal PC sampling buffers and generate a marker record
 * for the code object load/unload event.
 *
 * By using the @p code_object, the function finds the corresponding agent.
 * Then, it drains internal (ROCr + 2nd level trap) buffers of this agent
 * and places all samples in the SDK PC sampling buffer.
 * Finally, it places the marker record representing code object load/unload event
 * in the SDK PC sampling buffer.
 *
 * @param [in] phase       - loading/unloading phase
 * @param [in] code_object - loaded/unloaded code object.
 */
void
flush_pc_sampling_buffers(const rocprofiler::code_object::hsa::code_object& code_object)
{
    auto agent_id = code_object.rocp_data.rocp_agent;
    if(!is_pc_sample_service_configured(agent_id)) return;

    // The PC sampling service is configured on the agent,
    // so flush its PC sampling buffer
    // TODO: Creating a function that gives the buffer_id based on the agent_id?
    const auto* pcs_service     = get_configured_pc_sampling_service().load();
    const auto* agent_session   = pcs_service->agent_sessions.at(agent_id).get();
    auto        agent_buffer_id = agent_session->buffer_id;

    // flush internal PC sampling buffers
    flush_internal_agent_buffers(agent_buffer_id);
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    // Call underlying function
    hsa_status_t status = CHECK_NOTNULL(get_freeze_function())(executable, options);
    if(status != HSA_STATUS_SUCCESS) return status;

    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable == executable)
            {
                const auto& code_object_rocp = code_object.rocp_data;
                CodeobjTableTranslatorSynchronized::Get()->insert(
                    address_range_t{code_object_rocp.load_base,
                                    code_object_rocp.load_size,
                                    code_object_rocp.code_object_id});
                flush_pc_sampling_buffers(code_object);
            }
        });

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable == executable)
            {
                flush_pc_sampling_buffers(code_object);
                const auto& code_object_rocp = code_object.rocp_data;
                CodeobjTableTranslatorSynchronized::Get()->remove(
                    address_range_t{code_object_rocp.load_base,
                                    code_object_rocp.load_size,
                                    code_object_rocp.code_object_id});
            }
        });

    // Call underlying function
    return CHECK_NOTNULL(get_destroy_function())(executable);
}
}  // namespace

void
initialize(HsaApiTable* table)
{
    (void) table;
    auto& core_table = *table->core_;

    get_freeze_function()                = CHECK_NOTNULL(core_table.hsa_executable_freeze_fn);
    get_destroy_function()               = CHECK_NOTNULL(core_table.hsa_executable_destroy_fn);
    core_table.hsa_executable_freeze_fn  = executable_freeze;
    core_table.hsa_executable_destroy_fn = executable_destroy;
    LOG_IF(FATAL, get_freeze_function() == core_table.hsa_executable_freeze_fn)
        << "infinite recursion";
    LOG_IF(FATAL, get_destroy_function() == core_table.hsa_executable_destroy_fn)
        << "infinite recursion";
}

void
finalize()
{
    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            flush_pc_sampling_buffers(code_object);
        });
}

}  // namespace code_object
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
