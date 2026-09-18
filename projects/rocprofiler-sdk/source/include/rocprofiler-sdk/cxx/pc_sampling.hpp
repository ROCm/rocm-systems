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

#include <string>
#include <string_view>
#include <unordered_map>

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

/**
 * @brief Extracts the opcode (mnemonic) from a disassembled instruction.
 *
 * The disassembler emits instructions as `<mnemonic>[<whitespace><operands>]` with leading
 * whitespace already stripped, so the mnemonic is the leading run of non-whitespace characters.
 * Instructions without operands (e.g. `s_endpgm`) yield the whole string.
 */
inline std::string_view
extract_opcode(std::string_view instruction)
{
    return instruction.substr(0, instruction.find_first_of(" \t"));
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
    if(has_prefix(instruction, "s_sendmsg"))
        return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE;
    if(has_prefix(instruction, "s_wakeup")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST;
    if(has_prefix(instruction, "s_")) return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR;
    return ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER;
}

inline rocprofiler_status_t
verify_issued_instruction_type(rocprofiler_pc_sampling_instruction_type_t expected_inst_type,
                               rocprofiler_pc_sampling_instruction_type_t actual_inst_type,
                               std::string_view                           instruction)
{
    if(is_conditional_branch(instruction))
    {
        return (actual_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN ||
                actual_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN)
                   ? ROCPROFILER_STATUS_SUCCESS
                   : ROCPROFILER_STATUS_ERROR;
    }

    if(expected_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST &&
       !has_prefix(instruction, "s_wakeup"))
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    return actual_inst_type == expected_inst_type ? ROCPROFILER_STATUS_SUCCESS
                                                  : ROCPROFILER_STATUS_ERROR;
}

inline rocprofiler_status_t
verify_not_issued_reason(rocprofiler_pc_sampling_instruction_type_t              expected_inst_type,
                         rocprofiler_pc_sampling_instruction_not_issued_reason_t actual_reason,
                         std::string_view                                        instruction)
{
    if(expected_inst_type == ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST &&
       !has_prefix(instruction, "s_wakeup"))
    {
        return actual_reason ==
                       ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN
                   ? ROCPROFILER_STATUS_ERROR
                   : ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

/**
 * @brief Memoizes opcode classification across ::verify_sample calls.
 *
 * Classification is a pure function of the opcode (mnemonic), so the result can be reused for
 * every instruction sharing it: `s_mov_b32 s0, s1` and `s_mov_b32 s2, s3` are one entry. The ISA
 * has thousands of opcodes, but a sampling run only ever sees the hundreds actually hit by the
 * profiled kernels, and hotspots skew samples toward the costliest instructions — so their
 * opcodes dominate the cache and it fills up quickly. Rarely-sampled opcodes, sometimes seen only
 * once, mostly fall back to the prefix-matching path in ::classify_instruction instead.
 *
 * Passing a cache is optional; the three-argument ::verify_sample overload classifies the
 * instruction directly and keeps no state. Use this overload when profiling shows the per-sample
 * classification to be hot.
 *
 * This class exposes no public lookup API — only the four-argument ::verify_sample overload for
 * ::rocprofiler_pc_sampling_record_stochastic_v0_t is a friend, and it is the sole caller of the
 * private `classify()` method. Opcode classification rules stay solely in the anonymous namespace
 * above; `classify()` only memoizes their result.
 *
 * @warning Not thread safe, and deliberately so: PC sampling buffer callbacks may run
 * concurrently, and synchronizing this cache would likely cost more than the classification it
 * replaces. Construct one instance per thread (or per buffer callback) rather than sharing.
 *
 * @note Entries are keyed on the opcode alone, not on the GFX target. This is safe only because
 * ::verify_sample currently verifies a single target and returns early for all others. Keying
 * would have to include the target before a second one is supported.
 */
class verification_cache
{
public:
    verification_cache()                                     = default;
    ~verification_cache()                                    = default;
    verification_cache(const verification_cache&)            = delete;
    verification_cache& operator=(const verification_cache&) = delete;
    verification_cache(verification_cache&&)                 = default;
    verification_cache& operator=(verification_cache&&)      = default;

private:
    /**
     * @brief Returns the classification of @p instruction's opcode, computing and storing it on
     * first use. Delegates to ::classify_instruction on a cache miss; this method only adds the
     * opcode-keyed memoization.
     */
    rocprofiler_pc_sampling_instruction_type_t classify(std::string_view instruction)
    {
        // C++17 unordered_map has no heterogeneous lookup (that is C++20), so both paths build a
        // std::string key. Opcodes are short enough for the small-string optimization, so this is
        // a construct-and-hash rather than an allocation.
        auto opcode = extract_opcode(instruction);
        if(auto itr = m_entries.find(std::string{opcode}); itr != m_entries.end())
            return itr->second;

        return m_entries.emplace(std::string{opcode}, classify_instruction(instruction))
            .first->second;
    }

    std::unordered_map<std::string, rocprofiler_pc_sampling_instruction_type_t> m_entries = {};

    friend rocprofiler_status_t verify_sample(
        const rocprofiler_pc_sampling_record_stochastic_v0_t& record,
        std::string_view                                      instruction,
        uint32_t                                               gfx_target_version,
        verification_cache&                                   cache);
};

/**
 * @brief Verifies that a PC sampling record is internally consistent (e.g. the sampled
 * instruction type agrees with what the disassembly reports). The primary template covers
 * record kinds without any known verification rules and rejects the call as invalid; add an
 * overload for a new ::rocprofiler_pc_sampling_record_kind_t to opt it into verification.
 *
 * @param [in] record the PC sampling record to verify, e.g. an instance of
 * ::rocprofiler_pc_sampling_record_host_trap_v0_t or
 * ::rocprofiler_pc_sampling_record_stochastic_v0_t
 * @param [in] instruction_string the disassembled instruction text matching the sampled PC
 * @param [in] gfx_target_version the GFX IP version of the agent where the sample was taken.
 * Can be extracted from ::rocprofiler_agent_t.
 */
template <typename PcSamplingRecordT>
inline rocprofiler_status_t
verify_sample(const PcSamplingRecordT& /*record*/,
              std::string_view /*instruction_string*/,
              uint32_t /*gfx_target_version*/)
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
              std::string_view                                      instruction,
              uint32_t                                              gfx_target_version)
{
    if(gfx_target_version / 100 != 1205) return ROCPROFILER_STATUS_SUCCESS;

    if(record.wave_issued != 0)
    {
        auto expected_inst_type = classify_instruction(instruction);
        return verify_issued_instruction_type(
            expected_inst_type,
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(record.inst_type),
            instruction);
    }

    return verify_not_issued_reason(
        classify_instruction(instruction),
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            record.snapshot.reason_not_issued),
        instruction);
}

/**
 * @brief Verifies a PC sampling record, reusing @p cache to avoid re-classifying opcodes.
 *
 * Behaves identically to the three-argument overload; only the cost of repeated classification
 * differs. See ::rocprofiler::sdk::pc_sampling::verification_cache for the threading contract.
 *
 * @param [in] record the PC sampling record to verify
 * @param [in] instruction_string the disassembled instruction text matching the sampled PC
 * @param [in] gfx_target_version the GFX IP version of the agent where the sample was taken
 * @param [in,out] cache opcode classification cache owned by the caller
 */
template <typename PcSamplingRecordT>
inline rocprofiler_status_t
verify_sample(const PcSamplingRecordT& /*record*/,
              std::string_view /*instruction_string*/,
              uint32_t /*gfx_target_version*/,
              verification_cache& /*cache*/)
{
    return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
}

inline rocprofiler_status_t
verify_sample(const rocprofiler_pc_sampling_record_host_trap_v0_t&,
              std::string_view,
              uint32_t,
              verification_cache&)
{
    return ROCPROFILER_STATUS_SUCCESS;
}

inline rocprofiler_status_t
verify_sample(const rocprofiler_pc_sampling_record_stochastic_v0_t& record,
              std::string_view                                      instruction,
              uint32_t                                              gfx_target_version,
              verification_cache&                                   cache)
{
    if(gfx_target_version / 100 != 1205) return ROCPROFILER_STATUS_SUCCESS;

    if(record.wave_issued != 0)
    {
        auto expected_inst_type = cache.classify(instruction);
        return verify_issued_instruction_type(
            expected_inst_type,
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(record.inst_type),
            instruction);
    }

    return verify_not_issued_reason(
        cache.classify(instruction),
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            record.snapshot.reason_not_issued),
        instruction);
}
}  // namespace pc_sampling
}  // namespace sdk
}  // namespace rocprofiler
