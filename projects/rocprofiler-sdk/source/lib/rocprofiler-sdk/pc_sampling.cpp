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
#include <optional>
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

// Snapshot ext_data field name infrastructure (same pattern as instruction_type_string)
#define ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(CODE)                                             \
    template <>                                                                                    \
    struct snapshot_ext_field_string<CODE>                                                      \
    {                                                                                              \
        static constexpr auto name = #CODE;                                                        \
    };

template <size_t Idx>
struct snapshot_ext_field_string;

ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_NONE);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_MATRIX);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS_DIRECT);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_FLAT);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_MATRIX);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS_DIRECT);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_FLAT);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_DUAL_ISSUE_VALU);
ROCPROFILER_SNAPSHOT_EXT_FIELD_STRING(
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_LOCK_CONTENTION);

template <size_t Idx, size_t... Tail>
const char*
get_snapshot_ext_field_name(rocprofiler_pc_sampling_snapshot_ext_field_id_t field_id,
                               std::index_sequence<Idx, Tail...>)
{
    if(field_id == Idx) return snapshot_ext_field_string<Idx>::name;
    if constexpr(sizeof...(Tail) > 0)
        return get_snapshot_ext_field_name(field_id, std::index_sequence<Tail...>{});
    return nullptr;
}

/**
 * @brief Helper to check if a record kind is one of the supported version kinds (V0-V4).
 */
bool
is_valid_version_record_kind(rocprofiler_pc_sampling_record_kind_t kind)
{
    return kind == ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE;
}

/**
 * @brief Helper to check if a record kind can only be produced by the stochastic method.
 *
 * V2 and V4 carry stochastic-only fields that host-trap cannot populate, so host-trap
 * cannot produce them. V0/V1/V3 can be produced by either method.
 *
 * TODO: if a host-trap-only record kind is ever introduced, add a symmetric
 * is_host_trap_only_record_kind() here and tighten the validator/inference accordingly.
 */
bool
is_stochastic_only_record_kind(rocprofiler_pc_sampling_record_kind_t kind)
{
    return kind == ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE;
}

/**
 * @brief Validates the record_kinds array passed to rocprofiler_pc_sampling_configure_service_v2.
 *
 * Rules:
 * - record_kinds must not be null
 * - num_record_kinds must be > 0
 * - No duplicates
 * - At most one valid version kind (V0-V4)
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

// Snapshot ext_data fields supported by GFX9 (MI200/MI300/MI350)
const rocprofiler_pc_sampling_snapshot_ext_field_id_t gfx9_snapshot_ext_fields[] = {
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_MATRIX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_FLAT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_MATRIX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_FLAT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_DUAL_ISSUE_VALU,
};
constexpr size_t gfx9_snapshot_ext_fields_count =
    sizeof(gfx9_snapshot_ext_fields) / sizeof(gfx9_snapshot_ext_fields[0]);

// Snapshot ext_data fields supported by GFX12
const rocprofiler_pc_sampling_snapshot_ext_field_id_t gfx12_snapshot_ext_fields[] = {
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC,
};
constexpr size_t gfx12_snapshot_ext_fields_count =
    sizeof(gfx12_snapshot_ext_fields) / sizeof(gfx12_snapshot_ext_fields[0]);

// Snapshot ext_data fields supported by GFX1250 (GFX12 fields + LOCK_CONTENTION)
const rocprofiler_pc_sampling_snapshot_ext_field_id_t gfx1250_snapshot_ext_fields[] = {
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS_DIRECT,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_LOCK_CONTENTION,
};
constexpr size_t gfx1250_snapshot_ext_fields_count =
    sizeof(gfx1250_snapshot_ext_fields) / sizeof(gfx1250_snapshot_ext_fields[0]);

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

/**
 * @brief Validate API flags for conflicting or invalid combinations.
 */
rocprofiler_status_t
validate_api_flags(rocprofiler_pc_sampling_api_flags_t flags)
{
    auto f = static_cast<uint32_t>(flags);

    // Check for bits beyond the known flags
    constexpr uint32_t all_known = ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_HOST_TRAP |
                                   ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_STOCHASTIC |
                                   ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_HOST_TRAP |
                                   ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_STOCHASTIC;
    if(f & ~all_known) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // Conflicting REQUIRE flags
    if((f & ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_HOST_TRAP) &&
       (f & ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_STOCHASTIC))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // Conflicting PREFER flags
    if((f & ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_HOST_TRAP) &&
       (f & ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_STOCHASTIC))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // REQUIRE + opposite PREFER: REQUIRE takes priority, PREFER is ignored.
    // This is not an error — the hard constraint simply overrides the soft hint.

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Cross-cutting validation of record kinds against method flags.
 *
 * Rejects REQUIRE_HOST_TRAP combined with a stochastic-only record kind (V2, V4):
 * host-trap cannot produce those records, so the request is internally inconsistent.
 * The reverse (REQUIRE_STOCHASTIC + host-trap-layout kinds V1/V3) is NOT rejected, since
 * stochastic can produce records in the V1/V3 layout with stochastic-only fields unset.
 *
 * Pure function with no side effects. Assumes flags and record_kinds have already passed
 * validate_api_flags() and validate_record_kinds() respectively.
 */
rocprofiler_status_t
validate_record_kinds_against_flags(const rocprofiler_pc_sampling_record_kind_t* record_kinds,
                                    size_t                                       num_record_kinds,
                                    rocprofiler_pc_sampling_api_flags_t          flags)
{
    auto f = static_cast<uint32_t>(flags);

    if(f & ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_HOST_TRAP)
    {
        for(size_t i = 0; i < num_record_kinds; i++)
        {
            if(is_stochastic_only_record_kind(record_kinds[i]))
                return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Infer the preferred PC sampling method from record_kinds and flags.
 *
 * Precondition: the caller must have already run validate_api_flags(),
 * validate_record_kinds(), and validate_record_kinds_against_flags() on these inputs. This
 * function therefore assumes the inputs are internally consistent -- in particular, that
 * REQUIRE_HOST_TRAP is never combined with a stochastic-only record kind (V2/V4) -- and does
 * not re-check that case.
 *
 * Returns the method to use and whether it is a hard requirement. The record-kind heuristic
 * mirrors the compatibility matrix documented in pc_sampling.h:
 * - V0 records           -> prefer stochastic, fall back to host-trap
 * - V1 records           -> prefer host-trap, fall back to stochastic
 * - V2 records           -> stochastic
 * - V3 records           -> prefer host-trap, fall back to stochastic
 * - V4 records           -> stochastic
 * - INVALID_SAMPLE only  -> no method implied; defaults to stochastic (overridable by flags)
 * - REQUIRE flags override the record-kind heuristic
 * - PREFER flags act as soft hints (PREFER_HOST_TRAP is silently honored as stochastic when
 *   the requested record kinds are stochastic-only)
 *
 * @param[in] record_kinds  - array of record kinds
 * @param[in] num_record_kinds - size of array
 * @param[in] flags - API flags
 * @param[out] out_method - the inferred method
 * @param[out] is_required - true if REQUIRE flag is active (no fallback)
 * @return ROCPROFILER_STATUS_SUCCESS or error
 */
rocprofiler_status_t
infer_method_from_record_kinds_and_flags(
    const rocprofiler_pc_sampling_record_kind_t* record_kinds,
    size_t                                       num_record_kinds,
    rocprofiler_pc_sampling_api_flags_t          flags,
    rocprofiler_pc_sampling_method_t*            out_method,
    bool*                                        is_required)
{
    auto f = static_cast<uint32_t>(flags);

    // Determine method from record kinds (find the valid version, if any).
    auto method_from_records = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    for(size_t i = 0; i < num_record_kinds; i++)
    {
        auto kind = record_kinds[i];
        if(kind == ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE ||
           kind == ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE)
        {
            // V1 and V3 are host-trap record kinds
            method_from_records = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
            break;
        }
        else if(kind == ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE ||
                kind == ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE)
        {
            // V2 and V4 are stochastic record kinds
            method_from_records = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
            break;
        }
        // V0 and INVALID_SAMPLE don't imply a method
    }

    // Apply flags over the record-kind heuristic
    *is_required = false;

    if(f & ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_HOST_TRAP)
    {
        *out_method  = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
        *is_required = true;
    }
    else if(f & ROCPROFILER_PC_SAMPLING_API_FLAG_REQUIRE_STOCHASTIC)
    {
        *out_method  = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
        *is_required = true;
    }
    else if(f & ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_HOST_TRAP)
    {
        // Soft preference for host-trap. When the requested record kinds are stochastic-only
        // (V2, V4), host-trap cannot produce them, so the hint is silently ignored and
        // stochastic is used. This mirrors PREFER_STOCHASTIC being satisfied by stochastic-only
        // record kinds. The hard-requirement counterpart (REQUIRE_HOST_TRAP + V2/V4) is
        // rejected upstream by validate_record_kinds_against_flags().
        *out_method = (method_from_records == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
                          ? ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC
                          : ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    }
    else if(f & ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_STOCHASTIC)
    {
        *out_method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    }
    else if(method_from_records != ROCPROFILER_PC_SAMPLING_METHOD_NONE)
    {
        // Record kind implies a specific method
        *out_method = method_from_records;
    }
    else
    {
        // V0 or INVALID_SAMPLE only, no flags: default to prefer stochastic
        *out_method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
/**
 * @brief Check whether a record kind is supported on the given agent's architecture.
 *
 * V3 and V4 record kinds require GFX12+ (gfxip_major == 12).
 * All other record kinds (V0, V1, V2, INVALID_SAMPLE) are universally supported.
 */
bool
is_record_kind_supported_on_agent(rocprofiler_pc_sampling_record_kind_t kind,
                                  const rocprofiler_agent_t*            agent)
{
    if(kind == ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE ||
       kind == ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE)
    {
        auto gfxip_major = (agent->gfx_target_version / 10000) % 100;
        return gfxip_major == 12;
    }
    return true;
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
rocprofiler_pc_sampling_configure_service_v2(
    rocprofiler_context_id_t                     context_id,
    rocprofiler_agent_id_t                       agent_id,
    rocprofiler_pc_sampling_unit_t               unit,
    uint64_t                                     interval,
    rocprofiler_buffer_id_t                      buffer_id,
    const rocprofiler_pc_sampling_record_kind_t* record_kinds,
    size_t                                       num_record_kinds,
    rocprofiler_pc_sampling_api_flags_t          flags)
{
    if(!is_pc_sampling_explicitly_enabled()) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    // Validate flags
    auto flags_status = validate_api_flags(flags);
    if(flags_status != ROCPROFILER_STATUS_SUCCESS) return flags_status;

    // Validate the record_kinds array
    auto validate_status = validate_record_kinds(record_kinds, num_record_kinds);
    if(validate_status != ROCPROFILER_STATUS_SUCCESS) return validate_status;

    // Cross-cutting validation (flags x records): reject REQUIRE_HOST_TRAP + V2/V4
    auto cross_status =
        validate_record_kinds_against_flags(record_kinds, num_record_kinds, flags);
    if(cross_status != ROCPROFILER_STATUS_SUCCESS) return cross_status;

    // Infer the sampling method from record_kinds + flags
    auto method      = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    auto is_required = false;
    auto infer_status =
        infer_method_from_record_kinds_and_flags(record_kinds, num_record_kinds, flags,
                                                 &method, &is_required);
    if(infer_status != ROCPROFILER_STATUS_SUCCESS) return infer_status;

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // Reject V3/V4 record kinds on non-GFX12 agents
    for(size_t i = 0; i < num_record_kinds; i++)
    {
        if(!is_record_kind_supported_on_agent(record_kinds[i], agent))
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // checking if the registered context exists
    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // checking if the buffer is registered
    auto const* buff = rocprofiler::buffer::get_buffer(buffer_id);
    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    // Note on INVALID-only requests ({ INVALID_SAMPLE } with no valid version kind): they carry no
    // method preference of their own, so method selection falls entirely to the flags / stochastic
    // default + fallback below, exactly like any flagless request. There is intentionally no
    // special-casing here -- the carrier-kind decision (parse as V0, deliver only invalids) happens
    // downstream in PCSamplingParserContext::parse() from the requested-record-kinds set.

    // Try with inferred method first
    auto status = rocprofiler::pc_sampling::configure_pc_sampling_service_v2(
        ctx, agent, method, unit, interval, buffer_id, record_kinds, num_record_kinds);

    // A stochastic-only record kind (V2, V4) can only be parsed under the stochastic method.
    // Host-trap cannot produce those records, so there is no valid fallback: retrying with
    // host-trap would configure a session whose samples the parser cannot decode. In that case
    // we let the genuine failure from the stochastic attempt (e.g. NOT_AVAILABLE when the device
    // lacks stochastic) propagate instead of falling back.
    bool stochastic_only = false;
    for(size_t i = 0; i < num_record_kinds; i++)
    {
        if(is_stochastic_only_record_kind(record_kinds[i]))
        {
            stochastic_only = true;
            break;
        }
    }

    // If the preferred method failed and it's not a hard requirement, try the fallback
    if(status != ROCPROFILER_STATUS_SUCCESS && !is_required && !stochastic_only)
    {
        auto fallback = (method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
                            ? ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP
                            : ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
        status = rocprofiler::pc_sampling::configure_pc_sampling_service_v2(
            ctx, agent, fallback, unit, interval, buffer_id, record_kinds, num_record_kinds);
    }

    return status;
#else
    (void) context_id;
    (void) agent_id;
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
rocprofiler_pc_sampling_get_instruction_type_name_v2(
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
rocprofiler_pc_sampling_get_instruction_not_issued_reason_name_v2(
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
rocprofiler_pc_sampling_query_agent_configurations_v2(
    rocprofiler_agent_id_t                                   agent_id,
    const rocprofiler_pc_sampling_record_kind_t*             record_kinds,
    size_t                                                   num_record_kinds,
    rocprofiler_pc_sampling_api_flags_t                      flags,
    rocprofiler_available_pc_sampling_configurations_v2_cb_t cb,
    void*                                                    user_data)
{
    if(!is_pc_sampling_explicitly_enabled()) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    // Validate flags
    auto flags_status = validate_api_flags(flags);
    if(flags_status != ROCPROFILER_STATUS_SUCCESS) return flags_status;

    // Validate the record_kinds array
    auto validate_status = validate_record_kinds(record_kinds, num_record_kinds);
    if(validate_status != ROCPROFILER_STATUS_SUCCESS) return validate_status;

    // Cross-cutting validation (flags x records): reject REQUIRE_HOST_TRAP + V2/V4
    auto cross_status =
        validate_record_kinds_against_flags(record_kinds, num_record_kinds, flags);
    if(cross_status != ROCPROFILER_STATUS_SUCCESS) return cross_status;

    // Infer preferred method
    auto method      = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    auto is_required = false;
    auto infer_status =
        infer_method_from_record_kinds_and_flags(record_kinds, num_record_kinds, flags,
                                                 &method, &is_required);
    if(infer_status != ROCPROFILER_STATUS_SUCCESS) return infer_status;

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // V3/V4 record kinds are not supported on non-GFX12 agents: return empty configs
    // so the caller can fall through to a lower record kind (e.g. V2).
    for(size_t i = 0; i < num_record_kinds; i++)
    {
        if(!is_record_kind_supported_on_agent(record_kinds[i], agent))
            return cb(nullptr, 0, user_data);
    }

    // Query all v1 configs from the ioctl layer
    std::vector<rocprofiler_pc_sampling_configuration_t> v1_configs;
    auto status = rocprofiler::pc_sampling::ioctl::ioctl_query_pcs_configs(agent, v1_configs);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    // Note on INVALID-only requests ({ INVALID_SAMPLE } with no valid version kind): they are
    // filtered like a method-agnostic (V0-style) request -- preference flags are respected and,
    // absent flags, all configs are returned. There is intentionally no special-casing here; the
    // INVALID-only carrier-kind decision is a parse-time concern, not a configuration-query one.

    // A stochastic-only record kind (V2, V4) can only be produced by the stochastic method;
    // host-trap configs are not a valid substitute. Treat the method filter as hard in that
    // case, so when no stochastic config exists the caller receives an empty list (a valid but
    // unsatisfiable request) rather than misleading host-trap configs.
    bool stochastic_only = false;
    for(size_t i = 0; i < num_record_kinds; i++)
    {
        if(is_stochastic_only_record_kind(record_kinds[i]))
        {
            stochastic_only = true;
            break;
        }
    }

    // PREFER vs REQUIRE semantics (mirrors rocprofiler_pc_sampling_configure_v2):
    //   - REQUIRE (or stochastic-only records): only return configs that match the inferred
    //              method (drop the rest); an empty list means the request is unsatisfiable.
    //   - PREFER:  if at least one config matches the inferred method, return only those
    //              (preferred-method configs only); if none match, fall back to returning
    //              all configs so the caller can still pick the available method.
    //   - NONE:    no filtering (return all configs).
    bool filter_by_method = false;
    if(method != ROCPROFILER_PC_SAMPLING_METHOD_NONE)
    {
        if(is_required || stochastic_only)
        {
            filter_by_method = true;
        }
        else
        {
            // PREFER: filter only if a preferred-method config actually exists.
            for(auto const& cfg : v1_configs)
            {
                if(cfg.method == method)
                {
                    filter_by_method = true;
                    break;
                }
            }
        }
    }

    // Translate v1 -> v2, applying the method filter when set.
    std::vector<rocprofiler_pc_sampling_configuration_v2_t> v2_configs;
    v2_configs.reserve(v1_configs.size());
    for(auto const& cfg : v1_configs)
    {
        if(filter_by_method && cfg.method != method) continue;

        auto v2_cfg          = rocprofiler_pc_sampling_configuration_v2_t{};
        v2_cfg.size          = sizeof(rocprofiler_pc_sampling_configuration_v2_t);
        v2_cfg.unit          = cfg.unit;
        v2_cfg.min_interval  = cfg.min_interval;
        v2_cfg.max_interval  = cfg.max_interval;
        v2_cfg.flags =
            (cfg.flags == ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2)
                ? ROCPROFILER_PC_SAMPLING_CONFIGURATION_V2_FLAG_INTERVAL_POW2
                : ROCPROFILER_PC_SAMPLING_CONFIGURATION_V2_FLAG_NONE;
        v2_configs.push_back(v2_cfg);
    }

    // The callback receives a `const T**` array of pointers (matches the convention used by
    // ::rocprofiler_query_available_agents_cb_t). Build the pointer array from v2_configs.
    std::vector<const rocprofiler_pc_sampling_configuration_v2_t*> v2_config_ptrs;
    v2_config_ptrs.reserve(v2_configs.size());
    for(auto const& cfg : v2_configs)
        v2_config_ptrs.push_back(&cfg);

    return cb(v2_config_ptrs.data(), v2_config_ptrs.size(), user_data);
#else
    (void) agent_id;
    (void) record_kinds;
    (void) num_record_kinds;
    (void) cb;
    (void) user_data;

    ROCP_INFO << "PC sampling unavailable. The feature depends on the latest HSA runtime.";

    // ROCr runtime is missing PC sampling.
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
#endif
}

rocprofiler_status_t
rocprofiler_pc_sampling_get_snapshot_ext_field_name(
    rocprofiler_pc_sampling_snapshot_ext_field_id_t field_id,
    const char**                                       name,
    uint64_t*                                          name_len)
{
    if(!name || !name_len) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const char* n = get_snapshot_ext_field_name(
        field_id,
        std::make_index_sequence<ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_LAST>{});
    if(!n) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *name     = n;
    *name_len = std::strlen(n);
    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Helper to check if a record kind contains snapshot_information.
 * Host-trap record kinds (V1, V3) do not have snapshot_information.
 */
static bool
record_kind_has_snapshot_information(rocprofiler_pc_sampling_record_kind_t record_kind)
{
    switch(record_kind)
    {
        case ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE:
        case ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE:
            return true;
        default:
            return false;
    }
}

rocprofiler_status_t
rocprofiler_pc_sampling_query_snapshot_ext_fields(
    rocprofiler_agent_id_t                            agent_id,
    rocprofiler_pc_sampling_record_kind_t             record_kind,
    rocprofiler_pc_sampling_snapshot_ext_fields_cb_t  cb,
    void*                                             user_data)
{
    if(!record_kind_has_snapshot_information(record_kind))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // Determine architecture from agent's gfx_target_version
    auto gfxip_major = (agent->gfx_target_version / 10000) % 100;
    auto gfxip_minor = (agent->gfx_target_version / 100) % 100;

    const rocprofiler_pc_sampling_snapshot_ext_field_id_t* fields     = nullptr;
    size_t                                                 num_fields = 0;

    if(gfxip_major == 9)
    {
        fields     = gfx9_snapshot_ext_fields;
        num_fields = gfx9_snapshot_ext_fields_count;
    }
    else if(gfxip_major == 12 && gfxip_minor == 5)
    {
        fields     = gfx1250_snapshot_ext_fields;
        num_fields = gfx1250_snapshot_ext_fields_count;
    }
    else if(gfxip_major == 12)
    {
        fields     = gfx12_snapshot_ext_fields;
        num_fields = gfx12_snapshot_ext_fields_count;
    }
    else
    {
        // Architecture does not support snapshot ext_data fields, deliver empty array
        return cb(nullptr, 0, user_data);
    }

    return cb(fields, num_fields, user_data);
}

/**
 * @brief Helper to get the ext_data from a record based on its kind.
 * Returns the ext_data value from the snapshot_information embedded in the record,
 * or nullopt if the record kind does not contain snapshot_information.
 */
static std::optional<uint32_t>
get_ext_data_from_record(rocprofiler_pc_sampling_record_kind_t record_kind, const void* record)
{
    switch(record_kind)
    {
        case ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE:
        {
            const auto* rec =
                static_cast<const rocprofiler_pc_sampling_record_v2_t*>(record);
            return rec->snapshot_information.ext_data;
        }
        case ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE:
        {
            const auto* rec =
                static_cast<const rocprofiler_pc_sampling_record_v4_t*>(record);
            return rec->snapshot_information.ext_data;
        }
        default:
            return std::nullopt;
    }
}

rocprofiler_status_t
rocprofiler_pc_sampling_extract_snapshot_ext_field_values(
    rocprofiler_pc_sampling_record_kind_t                  record_kind,
    const void*                                            record,
    const rocprofiler_pc_sampling_snapshot_ext_field_id_t*  field_ids,
    size_t                                                 num_fields,
    rocprofiler_pc_sampling_snapshot_ext_field_values_cb_t  cb,
    void*                                                  user_data)
{
    if(num_fields == 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto ext_data_opt = get_ext_data_from_record(record_kind, record);
    if(!ext_data_opt.has_value())
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto ext_data = ext_data_opt.value();

    // Extract bit values from ext_data.
    // The ext_data is packed such that bit position N corresponds to
    // rocprofiler_pc_sampling_snapshot_ext_field_id_t value N.
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
        values[i] = (ext_data >> field_id) & 1u;
    }

    return cb(field_ids, values.data(), num_fields, user_data);
}
}
