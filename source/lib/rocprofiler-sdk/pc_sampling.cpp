// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include "lib/common/environment.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

namespace
{
#define ROCPROFILER_INSTRUCTION_TYPE_STRING(CODE)                                                  \
    template <>                                                                                    \
    struct instruction_type_string<CODE>                                                           \
    {                                                                                              \
        static constexpr auto name = #CODE;                                                        \
    };

#define ROCPROFILER_NO_ISSUE_REASON_STRING(CODE)                                                   \
    template <>                                                                                    \
    struct no_issue_reason_string<CODE>                                                            \
    {                                                                                              \
        static constexpr auto name = #CODE;                                                        \
    };

template <size_t Idx>
struct instruction_type_string;

template <size_t Idx>
struct no_issue_reason_string;

ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NONE);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS_DIRECT);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_EXPORT);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST);
ROCPROFILER_INSTRUCTION_TYPE_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU);

ROCPROFILER_NO_ISSUE_REASON_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY);
ROCPROFILER_NO_ISSUE_REASON_STRING(ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT);
ROCPROFILER_NO_ISSUE_REASON_STRING(
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT);

template <size_t Idx, size_t... Tail>
const char*
get_instruction_type_name(rocprofiler_pc_sampling_instruction_type_t instruction_type,
                          std::index_sequence<Idx, Tail...>)
{
    if(instruction_type == Idx) return instruction_type_string<Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_instruction_type_name(instruction_type, std::index_sequence<Tail...>{});
    return nullptr;
}

template <size_t Idx, size_t... Tail>
const char*
get_no_issue_reason_name(rocprofiler_pc_sampling_instruction_not_issued_reason_t no_issue_reason,
                         std::index_sequence<Idx, Tail...>)
{
    if(no_issue_reason == Idx) return no_issue_reason_string<Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_no_issue_reason_name(no_issue_reason, std::index_sequence<Tail...>{});
    return nullptr;
}

/**
 * @brief The functions checks if the `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` is set.
 * If so, it will enable PC sampling API. Otherwise, the API is reported
 * as not implemented.
 *
 * The PC sampling is in experimental phase and its usage may hang the machine
 * requiring the reboot. By enabling the `ROCPROFILER_PC_SAMPLING_BETA_ENABLED`,
 * user accepts all consequences of using early implementation of PC sampling API.
 */
bool
is_pc_sampling_explicitly_enabled()
{
    auto pc_sampling_enabled =
        rocprofiler::common::get_env("ROCPROFILER_PC_SAMPLING_BETA_ENABLED", false);

    if(!pc_sampling_enabled)
        ROCP_INFO << "PC sampling unavailable. The feature is implicitly disabled. "
                  << "To use it on a supported architecture, "
                  << "set ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON in the environment";

    return pc_sampling_enabled;
}
}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_id_t           agent_id,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id,
                                          int /*flags*/)
{
    if(!is_pc_sampling_explicitly_enabled()) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // checking if the registered context exists
    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // checking if the buffer is registered
    auto const* buff = rocprofiler::buffer::get_buffer(buffer_id);
    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    return rocprofiler::pc_sampling::configure_pc_sampling_service(
        ctx, agent, method, unit, interval, buffer_id);
#else
    (void) context_id;
    (void) agent_id;
    (void) method;
    (void) unit;
    (void) interval;
    (void) buffer_id;

    ROCP_INFO << "PC sampling unavailable. The feature depends on the latest HSA runtime.";

    // ROCr runtime is missing PC sampling.
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
#endif
}

rocprofiler_status_t
rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t                                agent_id,
    rocprofiler_available_pc_sampling_configurations_cb_t cb,
    void*                                                 user_data)
{
    if(!is_pc_sampling_explicitly_enabled()) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    auto status = rocprofiler::pc_sampling::ioctl::ioctl_query_pcs_configs(agent, configs);
    return (status == ROCPROFILER_STATUS_SUCCESS) ? cb(configs.data(), configs.size(), user_data)
                                                  : status;
#else
    (void) agent_id;
    (void) cb;
    (void) user_data;

    ROCP_INFO << "PC sampling unavailable. The feature depends on the latest HSA runtime.";

    // ROCr runtime is missing PC sampling.
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
#endif
}

const char*
rocprofiler_get_pc_sampling_instruction_type_name(
    rocprofiler_pc_sampling_instruction_type_t instruction_type)
{
    return get_instruction_type_name(
        instruction_type,
        std::make_index_sequence<ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LAST>{});
}

const char*
rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t not_issued_reason)
{
    return get_no_issue_reason_name(
        not_issued_reason,
        std::make_index_sequence<ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST>{});
}
}
