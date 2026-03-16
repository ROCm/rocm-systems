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

#include <cstring>
#include <unordered_set>

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

// Arbiter state field name infrastructure (same pattern as instruction_type_string)
#define ROCPROFILER_ARBITER_STATE_FIELD_STRING(CODE)                                               \
    template <>                                                                                    \
    struct arbiter_state_field_string<CODE>                                                        \
    {                                                                                              \
        static constexpr auto name = #CODE;                                                        \
    };

template <size_t Idx>
struct arbiter_state_field_string;

ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_NONE);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MATRIX);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS_DIRECT);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM_TEX);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_FLAT);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG_MISC);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MATRIX);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS_DIRECT);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM_TEX);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_FLAT);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG_MISC);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_DUAL_ISSUE_VALU);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED0);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED1);
ROCPROFILER_ARBITER_STATE_FIELD_STRING(ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED2);

template <size_t Idx, size_t... Tail>
const char*
get_arbiter_state_field_name(rocprofiler_pc_sampling_arbiter_state_field_id_t field_id,
                             std::index_sequence<Idx, Tail...>)
{
    if(field_id == Idx) return arbiter_state_field_string<Idx>::name;
    if constexpr(sizeof...(Tail) > 0)
        return get_arbiter_state_field_name(field_id, std::index_sequence<Tail...>{});
    return nullptr;
}

/**
 * @brief Helper to check if a record kind is one of the new valid version kinds (V0-V5).
 */
bool
is_valid_version_record_kind(rocprofiler_pc_sampling_record_kind_t kind)
{
    return kind >= ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE &&
           kind <= ROCPROFILER_PC_SAMPLING_RECORD_V5_SAMPLE;
}

/**
 * @brief Validates the record_kinds array passed to rocprofiler_configure_pc_sampling_service_v2.
 *
 * Rules:
 * - record_kinds must not be null
 * - num_record_kinds must be > 0
 * - No duplicates
 * - At most one valid version kind (V0-V5)
 * - INVALID_SAMPLE can appear independently alongside one valid version
 * - NONE, LAST, and old kinds (HOST_TRAP_V0, STOCHASTIC_V0) are not allowed
 */
rocprofiler_status_t
validate_record_kinds(const rocprofiler_pc_sampling_record_kind_t* record_kinds,
                      size_t                                       num_record_kinds)
{
    if(!record_kinds || num_record_kinds == 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto seen                = std::unordered_set<rocprofiler_pc_sampling_record_kind_t>{};
    int  valid_version_count = 0;

    for(size_t i = 0; i < num_record_kinds; i++)
    {
        auto kind = record_kinds[i];

        // Check for duplicates
        if(!seen.insert(kind).second) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        if(kind == ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE)
        {
            // INVALID_SAMPLE is allowed
            continue;
        }

        if(is_valid_version_record_kind(kind))
        {
            valid_version_count++;
            // At most one valid version kind allowed
            if(valid_version_count > 1) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
            continue;
        }

        // NONE, LAST, old kinds (HOST_TRAP_V0, STOCHASTIC_V0), or out-of-range values
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

// Arbiter state fields supported by GFX9 (MI200/MI300/MI350)
const rocprofiler_pc_sampling_arbiter_state_field_id_t gfx9_arbiter_fields[] = {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MATRIX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_FLAT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MATRIX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_FLAT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_DUAL_ISSUE_VALU,
};
constexpr size_t gfx9_arbiter_fields_count =
    sizeof(gfx9_arbiter_fields) / sizeof(gfx9_arbiter_fields[0]);

// Arbiter state fields supported by GFX12
const rocprofiler_pc_sampling_arbiter_state_field_id_t gfx12_arbiter_fields[] = {
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG_MISC,
};
constexpr size_t gfx12_arbiter_fields_count =
    sizeof(gfx12_arbiter_fields) / sizeof(gfx12_arbiter_fields[0]);

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

rocprofiler_status_t
rocprofiler_configure_pc_sampling_service_v2(
    rocprofiler_context_id_t                     context_id,
    rocprofiler_agent_id_t                       agent_id,
    rocprofiler_pc_sampling_method_t             method,
    rocprofiler_pc_sampling_unit_t               unit,
    uint64_t                                     interval,
    rocprofiler_buffer_id_t                      buffer_id,
    const rocprofiler_pc_sampling_record_kind_t* record_kinds,
    size_t                                       num_record_kinds,
    uint32_t                                     flags)
{
    if(!is_pc_sampling_explicitly_enabled()) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    // flags must be 0 (reserved for future use)
    if(flags != 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // Validate the record_kinds array
    auto validate_status = validate_record_kinds(record_kinds, num_record_kinds);
    if(validate_status != ROCPROFILER_STATUS_SUCCESS) return validate_status;

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

    return rocprofiler::pc_sampling::configure_pc_sampling_service_v2(
        ctx, agent, method, unit, interval, buffer_id, record_kinds, num_record_kinds);
#else
    (void) context_id;
    (void) agent_id;
    (void) method;
    (void) unit;
    (void) interval;
    (void) buffer_id;
    (void) record_kinds;
    (void) num_record_kinds;

    ROCP_INFO << "PC sampling unavailable. The feature depends on the latest HSA runtime.";

    // ROCr runtime is missing PC sampling.
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
#endif
}

rocprofiler_status_t
rocprofiler_get_pc_sampling_instruction_type_name_(
    rocprofiler_pc_sampling_instruction_type_t instruction_type,
    const char**                               name,
    uint64_t*                                  name_len)
{
    if(!name || !name_len) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const char* n = rocprofiler_get_pc_sampling_instruction_type_name(instruction_type);
    if(!n) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *name     = n;
    *name_len = std::strlen(n);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_get_pc_sampling_instruction_not_issued_reason_name_(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t not_issued_reason,
    const char**                                            name,
    uint64_t*                                               name_len)
{
    if(!name || !name_len) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const char* n =
        rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(not_issued_reason);
    if(!n) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *name     = n;
    *name_len = std::strlen(n);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_get_pc_sampling_arbiter_state_field_name(
    rocprofiler_pc_sampling_arbiter_state_field_id_t field_id,
    const char**                                     name,
    uint64_t*                                        name_len)
{
    if(!name || !name_len) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const char* n = get_arbiter_state_field_name(
        field_id, std::make_index_sequence<ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_LAST>{});
    if(!n) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *name     = n;
    *name_len = std::strlen(n);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_query_pc_sampling_arbiter_fields(rocprofiler_agent_id_t                      agent_id,
                                             rocprofiler_pc_sampling_arbiter_fields_cb_t cb,
                                             void*                                       user_data)
{
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // Determine architecture from agent's gfx_target_version
    auto gfxip_major = (agent->gfx_target_version / 10000) % 100;

    const rocprofiler_pc_sampling_arbiter_state_field_id_t* fields     = nullptr;
    size_t                                                  num_fields = 0;

    if(gfxip_major == 9)
    {
        fields     = gfx9_arbiter_fields;
        num_fields = gfx9_arbiter_fields_count;
    }
    else if(gfxip_major == 12)
    {
        fields     = gfx12_arbiter_fields;
        num_fields = gfx12_arbiter_fields_count;
    }
    else
    {
        // Architecture does not support arbiter state fields, deliver empty array
        return cb(nullptr, 0, user_data);
    }

    return cb(fields, num_fields, user_data);
}

rocprofiler_status_t
rocprofiler_pc_sampling_get_arbiter_state_fields(
    uint32_t                                                arbiter_state,
    const rocprofiler_pc_sampling_arbiter_state_field_id_t* field_ids,
    size_t                                                  num_fields,
    rocprofiler_pc_sampling_arbiter_field_values_cb_t       cb,
    void*                                                   user_data)
{
    if(num_fields == 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // Extract bit values from arbiter_state.
    // The arbiter_state is packed such that bit position N corresponds to
    // rocprofiler_pc_sampling_arbiter_state_field_id_t value N.
    // Each field is a single bit.
    auto values = std::vector<uint32_t>(num_fields);
    for(size_t i = 0; i < num_fields; i++)
    {
        auto field_id = static_cast<uint32_t>(field_ids[i]);
        if(field_id >= 32)
        {
            values[i] = 0;
            continue;
        }
        values[i] = (arbiter_state >> field_id) & 1u;
    }

    return cb(field_ids, values.data(), num_fields, user_data);
}
}
