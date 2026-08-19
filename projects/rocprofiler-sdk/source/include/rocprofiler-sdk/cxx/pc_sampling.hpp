// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
//

#pragma once

#include <rocprofiler-sdk/pc_sampling.h>

#include <string_view>

namespace rocprofiler
{
namespace sdk
{
namespace pc_sampling
{
namespace
{
inline bool
has_prefix(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

inline bool
is_conditional_branch(std::string_view instruction)
{
    return has_prefix(instruction, "s_cbranch");
}

inline rocprofiler_pc_sampling_instruction_type_t
classify_instruction(std::string_view instruction)
{
    if(has_prefix(instruction, "v_wmma")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX;
    if(has_prefix(instruction, "v_dual_"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU;
    if(has_prefix(instruction, "v_")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU;
    if(has_prefix(instruction, "global_") || has_prefix(instruction, "buffer_"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX;
    if(has_prefix(instruction, "flat_")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT;
    if(has_prefix(instruction, "ds_")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS;
    if(has_prefix(instruction, "s_nop") || has_prefix(instruction, "s_sleep"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_monitor_sleep"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_wait")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_barrier_wait"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_setprio"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_delay_alu"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_sethalt"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_setkill"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_singleuse_vdst"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_round_mode"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_denorm_mode"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_version"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_clause")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_icache_inv"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(is_conditional_branch(instruction))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN;
    if(has_prefix(instruction, "s_branch"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN;
    if(has_prefix(instruction, "s_barrier_signal"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER;
    if(has_prefix(instruction, "s_swap_pc") || has_prefix(instruction, "s_set_pc"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP;
    if(has_prefix(instruction, "s_sendmsg")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE;
    if(has_prefix(instruction, "s_wakeup"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR;
    return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER;
}

inline rocprofiler_status_t
verify_issued_instruction_type(rocprofiler_pc_sampling_instruction_type_t expected_inst_type,
                                bool                                        is_conditional_branch_,
                                rocprofiler_pc_sampling_instruction_type_t actual_inst_type)
{
    if(is_conditional_branch_)
    {
        return (actual_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN ||
                actual_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN)
                   ? ROCPROFILER_STATUS_SUCCESS
                   : ROCPROFILER_STATUS_ERROR;
    }

    return actual_inst_type == expected_inst_type ? ROCPROFILER_STATUS_SUCCESS
                                                   : ROCPROFILER_STATUS_ERROR;
}

inline rocprofiler_status_t
verify_not_issued_reason(rocprofiler_pc_sampling_instruction_type_t              expected_inst_type,
                          rocprofiler_pc_sampling_instruction_not_issued_reason_t actual_reason)
{
    if(expected_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST)
    {
        return actual_reason == ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN
                   ? ROCPROFILER_STATUS_ERROR
                   : ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

/**
 * @brief Verifies that a PC sampling record is internally consistent (e.g. the sampled
 * instruction type agrees with what the disassembly reports). The primary template covers
 * record kinds without any known verification rules and rejects the call as invalid; add an
 * overload for a new ::rocprofiler_pc_sampling_record_kind_t to opt it into verification.
 */
template <typename PcSamplingRecordT>
inline rocprofiler_status_t
verify_sample(const PcSamplingRecordT&, std::string_view, uint32_t)
{
    return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
}

inline rocprofiler_status_t
verify_sample(const rocprofiler_pc_sampling_record_host_trap_v0_t&, std::string_view, uint32_t)
{
    return ROCPROFILER_STATUS_SUCCESS;
}

inline rocprofiler_status_t
verify_sample(const rocprofiler_pc_sampling_record_stochastic_v0_t& record,
              std::string_view                                    instruction,
              uint32_t                                            gfx_target_version)
{
    if(gfx_target_version / 100 != 1205) return ROCPROFILER_STATUS_SUCCESS;

    if(record.wave_issued != 0)
    {
        auto expected_inst_type = classify_instruction(instruction);
        return verify_issued_instruction_type(
            expected_inst_type,
            is_conditional_branch(instruction),
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(record.inst_type));
    }

    return verify_not_issued_reason(
        classify_instruction(instruction),
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            record.snapshot.reason_not_issued));
}
}  // namespace pc_sampling
}  // namespace sdk
}  // namespace rocprofiler
