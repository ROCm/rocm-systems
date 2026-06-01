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

#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx11.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx12.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx1250.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx9.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx950.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/parser_types.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.h"

#include <rocprofiler-sdk/pc_sampling.h>

#include <array>
#include <cstdint>
#include <cstring>

#define LUTOVERLOAD(sname, rocp_prefix) this->operator[](GFX::sname) = rocp_prefix##_##sname
#define LUTOVERLOAD_INST(sname)         LUTOVERLOAD(sname, ROCPROFILER_PC_SAMPLING_INSTRUCTION)
#define LUTOVERLOAD_INST_NOT_ISSUED(sname)                                                         \
    LUTOVERLOAD(sname, ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED)

template <typename GFX>
struct gfx_inst_lut : public std::array<int, 32>
{
    gfx_inst_lut()
    {
        std::memset(data(), 0, size() * sizeof(int));
        LUTOVERLOAD_INST(TYPE_VALU);
        LUTOVERLOAD_INST(TYPE_MATRIX);
        LUTOVERLOAD_INST(TYPE_SCALAR);
        LUTOVERLOAD_INST(TYPE_TEX);
        LUTOVERLOAD_INST(TYPE_LDS);
        LUTOVERLOAD_INST(TYPE_LDS_DIRECT);
        LUTOVERLOAD_INST(TYPE_FLAT);
        LUTOVERLOAD_INST(TYPE_EXPORT);
        LUTOVERLOAD_INST(TYPE_MESSAGE);
        LUTOVERLOAD_INST(TYPE_BARRIER);
        LUTOVERLOAD_INST(TYPE_BRANCH_NOT_TAKEN);
        LUTOVERLOAD_INST(TYPE_BRANCH_TAKEN);
        LUTOVERLOAD_INST(TYPE_JUMP);
        LUTOVERLOAD_INST(TYPE_OTHER);
        LUTOVERLOAD_INST(TYPE_NO_INST);
        LUTOVERLOAD_INST(TYPE_DUAL_VALU);
    }
};

template <typename GFX>
struct gfx_reason_lut : public std::array<int, 32>
{
    gfx_reason_lut()
    {
        std::memset(data(), 0, size() * sizeof(int));
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_NO_INSTRUCTION_AVAILABLE);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_ALU_DEPENDENCY);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_WAITCNT);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_INTERNAL_INSTRUCTION);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_BARRIER_WAIT);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_ARBITER_NOT_WIN);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_ARBITER_WIN_EX_STALL);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_OTHER_WAIT);
        LUTOVERLOAD_INST_NOT_ISSUED(REASON_SLEEP_WAIT);
    }
};

template <typename GFX>
inline int
translate_inst(int in)
{
    static gfx_inst_lut<GFX> lut;
    return lut[in & 0x1F];
}

template <typename GFX>
inline int
translate_reason(int in)
{
    static gfx_reason_lut<GFX> lut;
    return lut[in & 0x1F];
}

#undef LUTOVERLOAD_INST_NOT_ISSUED
#undef LUTOVERLOAD_INST
#undef LUTOVERLOAD

#define EXTRACT_BITS(val, bit_end, bit_start)                                                      \
    ((val >> bit_start) & ((1U << (bit_end - bit_start + 1)) - 1))

template <typename GFX, typename PcSamplingRecordT, typename SType>
inline void
copyChipletId(PcSamplingRecordT& record, const SType& sample)
{
    // extract chiplet record
    record.hw_id.chiplet = sample.chiplet_and_wave_id >> 8;
}

template <typename GFX, typename HwIdT>
inline void
copyHwId(HwIdT& hw_id, const uint32_t hsa_hw_id);

template <>
inline void
copyHwId<GFX9, rocprofiler_pc_sampling_hw_id_v0_t>(rocprofiler_pc_sampling_hw_id_v0_t& hw_id,
                                                   const uint32_t                      hw_id_reg)
{
    // 3:0 -> wave_id
    hw_id.wave_id = EXTRACT_BITS(hw_id_reg, 3, 0);
    // 5:4 -> simd_id
    hw_id.simd_id = EXTRACT_BITS(hw_id_reg, 5, 4);
    // 7:6 -> pipe_id;
    hw_id.pipe_id = EXTRACT_BITS(hw_id_reg, 7, 6);
    // 11:8 -> cu_id
    hw_id.cu_or_wgp_id = EXTRACT_BITS(hw_id_reg, 11, 8);
    // 12 -> sa_id
    hw_id.shader_array_id = EXTRACT_BITS(hw_id_reg, 12, 12);
    // 15:13 -> se_id
    hw_id.shader_engine_id = EXTRACT_BITS(hw_id_reg, 15, 13);
    // 19:16 -> tg_id
    hw_id.workgroup_id = EXTRACT_BITS(hw_id_reg, 19, 16);
    // 23:20 -> vm_id
    hw_id.vm_id = EXTRACT_BITS(hw_id_reg, 23, 20);
    // 26:24 -> queue_id
    hw_id.queue_id = EXTRACT_BITS(hw_id_reg, 26, 24);
    // 29:27 -> state_id (ignored)
    // 31:30 -> me_id
    hw_id.microengine_id = EXTRACT_BITS(hw_id_reg, 31, 30);
}

template <>
inline void
copyHwId<GFX12, rocprofiler_pc_sampling_hw_id_v0_t>(rocprofiler_pc_sampling_hw_id_v0_t& hw_id,
                                                    const uint32_t                      hw_id_reg)
{
    // 4:0 -> wave_id
    hw_id.wave_id = EXTRACT_BITS(hw_id_reg, 4, 0);
    // 8:5 -> queue_id
    hw_id.queue_id = EXTRACT_BITS(hw_id_reg, 8, 5);
    // 13:10 -> wgp_id
    hw_id.cu_or_wgp_id = EXTRACT_BITS(hw_id_reg, 13, 10);
    // 15:14 -> simd_id
    hw_id.simd_id = EXTRACT_BITS(hw_id_reg, 15, 14);
    // 16 -> sa_id
    hw_id.shader_array_id = EXTRACT_BITS(hw_id_reg, 16, 16);
    // 17 -> me_id
    hw_id.microengine_id = EXTRACT_BITS(hw_id_reg, 17, 17);
    // 19:18 -> se_id
    hw_id.shader_engine_id = EXTRACT_BITS(hw_id_reg, 19, 18);
    // 21:20 -> pipe_id
    hw_id.pipe_id = EXTRACT_BITS(hw_id_reg, 21, 20);
    // 27:23 -> wg_id
    hw_id.workgroup_id = EXTRACT_BITS(hw_id_reg, 27, 23);
    // 31:28 -> vm_id
    hw_id.vm_id = EXTRACT_BITS(hw_id_reg, 31, 28);
}

// --- copyHwId specializations for hw_id_v1_t (unpacked uint8_t fields) ---

template <>
inline void
copyHwId<GFX9, rocprofiler_pc_sampling_hw_id_v1_t>(rocprofiler_pc_sampling_hw_id_v1_t& hw_id,
                                                   const uint32_t                      hw_id_reg)
{
    hw_id.wave_id          = EXTRACT_BITS(hw_id_reg, 3, 0);
    hw_id.simd_id          = EXTRACT_BITS(hw_id_reg, 5, 4);
    hw_id.pipe_id          = EXTRACT_BITS(hw_id_reg, 7, 6);
    hw_id.cu_or_wgp_id     = EXTRACT_BITS(hw_id_reg, 11, 8);
    hw_id.shader_array_id  = EXTRACT_BITS(hw_id_reg, 12, 12);
    hw_id.shader_engine_id = EXTRACT_BITS(hw_id_reg, 15, 13);
    hw_id.workgroup_id     = EXTRACT_BITS(hw_id_reg, 19, 16);
    hw_id.vm_id            = EXTRACT_BITS(hw_id_reg, 23, 20);
    hw_id.queue_id         = EXTRACT_BITS(hw_id_reg, 26, 24);
    hw_id.microengine_id   = EXTRACT_BITS(hw_id_reg, 31, 30);
}

template <>
inline void
copyHwId<GFX12, rocprofiler_pc_sampling_hw_id_v1_t>(rocprofiler_pc_sampling_hw_id_v1_t& hw_id,
                                                    const uint32_t                      hw_id_reg)
{
    hw_id.wave_id          = EXTRACT_BITS(hw_id_reg, 4, 0);
    hw_id.queue_id         = EXTRACT_BITS(hw_id_reg, 8, 5);
    hw_id.cu_or_wgp_id     = EXTRACT_BITS(hw_id_reg, 13, 10);
    hw_id.simd_id          = EXTRACT_BITS(hw_id_reg, 15, 14);
    hw_id.shader_array_id  = EXTRACT_BITS(hw_id_reg, 16, 16);
    hw_id.microengine_id   = EXTRACT_BITS(hw_id_reg, 17, 17);
    hw_id.shader_engine_id = EXTRACT_BITS(hw_id_reg, 19, 18);
    hw_id.pipe_id          = EXTRACT_BITS(hw_id_reg, 21, 20);
    hw_id.workgroup_id     = EXTRACT_BITS(hw_id_reg, 27, 23);
    hw_id.vm_id            = EXTRACT_BITS(hw_id_reg, 31, 28);
}

template <typename PcSamplingRecordT, typename SType>
inline PcSamplingRecordT
copySampleHeader(const SType& sample)
{
    // value-init zeroes out all record fields
    PcSamplingRecordT ret{};

    // Decode fields common for all host-trap and stochastic on all architectures.
    ret.size          = sizeof(PcSamplingRecordT);
    ret.wave_in_group = sample.chiplet_and_wave_id & 0x3F;

    ret.exec_mask      = sample.exec_mask;
    ret.workgroup_id.x = sample.workgroup_id_x;
    ret.workgroup_id.y = sample.workgroup_id_y;
    ret.workgroup_id.z = sample.workgroup_id_z;

    ret.timestamp = sample.timestamp;

    return ret;
}

template <typename GFX, typename PcSamplingRecordT, rocprofiler_pc_sampling_method_t Method>
inline PcSamplingRecordT
copySample(const void* sample);

/**
 * @brief Host trap V0 sample for GFX9
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_host_trap_v0_t>(sample_);
    copyChipletId<GFX9>(ret, sample_);
    copyHwId<GFX9>(ret.hw_id, sample_.hw_id);
    // copyHwId<GFX9>(&ret, sample);
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);

    // Extracting data from the perf_snapshot_data register
    auto perf_snapshot_data = sample_.perf_snapshot_data;
    // The sample is valid iff neither of perf_snapshot_data.valid and perf_snapshot_data.error == 0
    // is one
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0) &
                                   ~EXTRACT_BITS(perf_snapshot_data, 26, 26));
    if(!valid)
    {
        // To reduce refactoring of the PC sampling parser, we agreed to internally represent
        // invalid samples with `rocprofiler_pc_sampling_record_stochastic_v0_t` with size 0.
        // Eventually, those records are replaced with rocprofiler_pc_sampling_record_invalid_t
        // and placed into the SDK buffer consumed by the end tool.
        rocprofiler_pc_sampling_record_stochastic_v0_t invalid{};
        invalid.size = 0;
        // No need to further process invalid samples
        return invalid;
    }

    auto ret = copySampleHeader<rocprofiler_pc_sampling_record_stochastic_v0_t>(sample_);
    copyChipletId<GFX9>(ret, sample_);
    copyHwId<GFX9>(ret.hw_id, sample_.hw_id);

    // no memory counters on GFX9
    ret.flags.has_memory_counter = false;

    // wave issued an instruction
    ret.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    // type of issued instruction, valid only if `ret.wave_issued` is true.
    ret.inst_type = translate_inst<GFX9>(EXTRACT_BITS(perf_snapshot_data, 6, 3));
    // two VALU instructions issued in this cycles
    ret.snapshot.dual_issue_valu = EXTRACT_BITS(perf_snapshot_data, 2, 2);
    // reason for not issuing an instruction, valid only if `ret.wave_issued` is false
    ret.snapshot.reason_not_issued = translate_reason<GFX9>(EXTRACT_BITS(perf_snapshot_data, 9, 7));

    // arbiter state information
    uint16_t arb_state                    = EXTRACT_BITS(perf_snapshot_data, 25, 10);
    ret.snapshot.arb_state_issue_valu     = EXTRACT_BITS(arb_state, 7, 7);
    ret.snapshot.arb_state_issue_matrix   = EXTRACT_BITS(arb_state, 6, 6);
    ret.snapshot.arb_state_issue_lds      = EXTRACT_BITS(arb_state, 3, 3);
    ret.snapshot.arb_state_issue_scalar   = EXTRACT_BITS(arb_state, 5, 5);
    ret.snapshot.arb_state_issue_vmem_tex = EXTRACT_BITS(arb_state, 4, 4);
    ret.snapshot.arb_state_issue_flat     = EXTRACT_BITS(arb_state, 2, 2);
    ret.snapshot.arb_state_issue_exp      = EXTRACT_BITS(arb_state, 1, 1);
    ret.snapshot.arb_state_issue_misc     = EXTRACT_BITS(arb_state, 0, 0);

    ret.snapshot.arb_state_stall_valu     = EXTRACT_BITS(arb_state, 15, 15);
    ret.snapshot.arb_state_stall_matrix   = EXTRACT_BITS(arb_state, 14, 14);
    ret.snapshot.arb_state_stall_lds      = EXTRACT_BITS(arb_state, 11, 11);
    ret.snapshot.arb_state_stall_scalar   = EXTRACT_BITS(arb_state, 13, 13);
    ret.snapshot.arb_state_stall_vmem_tex = EXTRACT_BITS(arb_state, 12, 12);
    ret.snapshot.arb_state_stall_flat     = EXTRACT_BITS(arb_state, 10, 10);
    ret.snapshot.arb_state_stall_exp      = EXTRACT_BITS(arb_state, 9, 9);
    ret.snapshot.arb_state_stall_misc     = EXTRACT_BITS(arb_state, 8, 8);

    // Extracting data from the perf_snapshot_data1 register
    // Active waves on CU at the moment of sampling
    ret.wave_count = EXTRACT_BITS(sample_.perf_snapshot_data1, 5, 0);

    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX950, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX950, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(sample);
}

/**
 * @brief Host trap V0 sample for GFX11
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_host_trap_v0_t>(sample_);
    // TODO: decode other fields.
    return ret;
}

// TODO: implement stochastic for GFX11
template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_stochastic_v0_t>(sample_);
    // TODO: decode other fields
    // TODO: implement logic for manipulating stochastic related fields
    // ret.wave_count      = sample_.perf_snapshot_data1 & 0x3F;
    return ret;
}

/**
 * @brief Host trap V0 sample for GFX12
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_host_trap_v0_t>(sample_);
    copyHwId<GFX12>(ret.hw_id, sample_.hw_id);
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);

    // Extracting data from the perf_snapshot_data register
    auto perf_snapshot_data = sample_.perf_snapshot_data;
    // The sample is valid if perf_snapshot_data.valid == 1
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // To reduce refactoring of the PC sampling parser, we agreed to internally represent
        // invalid samples with `rocprofiler_pc_sampling_record_stochastic_v0_t` with size 0.
        // Eventually, those records are replaced with rocprofiler_pc_sampling_record_invalid_t
        // and placed into the SDK buffer consumed by the end tool.
        rocprofiler_pc_sampling_record_stochastic_v0_t invalid{};
        invalid.size = 0;
        // No need to further process invalid samples
        return invalid;
    }

    auto ret = copySampleHeader<rocprofiler_pc_sampling_record_stochastic_v0_t>(sample_);
    copyHwId<GFX12>(ret.hw_id, sample_.hw_id);

    // wave issued an instruction
    ret.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    // type of issued instruction, valid only if `ret.wave_issued` is true.
    ret.inst_type = translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    // reason for not issuing an instruction, valid only if `ret.wave_issued` is false
    ret.snapshot.reason_not_issued =
        translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));

    // arbiter state information
    auto     perf_snapshot_data1 = sample_.perf_snapshot_data1;
    uint16_t arb_state           = EXTRACT_BITS(perf_snapshot_data1, 21, 6);

    ret.snapshot.arb_state_issue_brmsg      = EXTRACT_BITS(arb_state, 0, 0);
    ret.snapshot.arb_state_issue_exp        = EXTRACT_BITS(arb_state, 1, 1);
    ret.snapshot.arb_state_issue_lds_direct = EXTRACT_BITS(arb_state, 2, 2);
    ret.snapshot.arb_state_issue_lds        = EXTRACT_BITS(arb_state, 3, 3);
    ret.snapshot.arb_state_issue_vmem_tex   = EXTRACT_BITS(arb_state, 4, 4);
    ret.snapshot.arb_state_issue_scalar     = EXTRACT_BITS(arb_state, 5, 5);
    ret.snapshot.arb_state_issue_valu       = EXTRACT_BITS(arb_state, 6, 6);

    ret.snapshot.arb_state_stall_brmsg      = EXTRACT_BITS(arb_state, 8, 8);
    ret.snapshot.arb_state_stall_exp        = EXTRACT_BITS(arb_state, 9, 9);
    ret.snapshot.arb_state_stall_lds_direct = EXTRACT_BITS(arb_state, 10, 10);
    ret.snapshot.arb_state_stall_lds        = EXTRACT_BITS(arb_state, 11, 11);
    ret.snapshot.arb_state_stall_vmem_tex   = EXTRACT_BITS(arb_state, 12, 12);
    ret.snapshot.arb_state_stall_scalar     = EXTRACT_BITS(arb_state, 13, 13);
    ret.snapshot.arb_state_stall_valu       = EXTRACT_BITS(arb_state, 14, 14);

    ret.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Memory counters exist on GFX12.
    ret.flags.has_memory_counter = true;
    // Extracting memory counters from the perf_snapshot_data2 register
    auto perf_snapshot_data2       = sample_.perf_snapshot_data2;
    ret.memory_counters.load_cnt   = EXTRACT_BITS(perf_snapshot_data2, 5, 0);
    ret.memory_counters.store_cnt  = EXTRACT_BITS(perf_snapshot_data2, 11, 6);
    ret.memory_counters.bvh_cnt    = EXTRACT_BITS(perf_snapshot_data2, 14, 12);
    ret.memory_counters.sample_cnt = EXTRACT_BITS(perf_snapshot_data2, 20, 15);
    ret.memory_counters.ds_cnt     = EXTRACT_BITS(perf_snapshot_data2, 26, 21);
    ret.memory_counters.km_cnt     = EXTRACT_BITS(perf_snapshot_data2, 31, 27);

    return ret;
}

// =====================================================================================
// New API (v2) record type specializations: V0, V1, V2
// =====================================================================================

/**
 * @brief Helper to copy common fields shared by all v2 record types (V0/V1/V2).
 * V0/V1/V2 records do NOT have a `size` member.
 */
template <typename PcSamplingRecordT, typename SType>
inline PcSamplingRecordT
copySampleCommon(const SType& sample)
{
    // value-init zeroes out all record fields
    PcSamplingRecordT ret{};
    ret.exec_mask = sample.exec_mask;
    ret.timestamp = sample.timestamp;
    return ret;
}

/**
 * @brief Helper to pack GFX9 arbiter state bits into the canonical ext_data uint32_t layout
 * where bit N corresponds to rocprofiler_pc_sampling_snapshot_ext_field_id_t value N.
 *
 * GFX9 arb_state register layout (from perf_snapshot_data bits [25:10]):
 *   bit 0 = ISSUE_MISC, bit 1 = ISSUE_EXP, bit 2 = ISSUE_FLAT, bit 3 = ISSUE_LDS,
 *   bit 4 = ISSUE_VMEM_TEX, bit 5 = ISSUE_SCALAR, bit 6 = ISSUE_MATRIX, bit 7 = ISSUE_VALU
 *   bit 8 = STALL_MISC, bit 9 = STALL_EXP, bit 10 = STALL_FLAT, bit 11 = STALL_LDS,
 *   bit 12 = STALL_VMEM_TEX, bit 13 = STALL_SCALAR, bit 14 = STALL_MATRIX, bit 15 = STALL_VALU
 */
inline uint32_t
pack_snapshot_ext_data_gfx9(uint16_t arb_state, bool dual_issue_valu)
{
    uint32_t result = 0;
    // Map HW bits to canonical field IDs
    auto set_bit = [&](rocprofiler_pc_sampling_snapshot_ext_field_id_t field, int hw_bit) {
        if(arb_state & (1u << hw_bit)) result |= (1u << field);
    };
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU, 7);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_MATRIX, 6);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS, 3);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR, 5);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX, 4);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_FLAT, 2);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP, 1);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC, 0);

    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU, 15);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_MATRIX, 14);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS, 11);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR, 13);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX, 12);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_FLAT, 10);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP, 9);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC, 8);

    if(dual_issue_valu)
        result |= (1u << ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_DUAL_ISSUE_VALU);

    return result;
}

/**
 * @brief Pack GFX12 arbiter state bits into canonical ext_data uint32_t layout.
 *
 * GFX12 arb_state register layout (from perf_snapshot_data1 bits [21:6]):
 *   bit 0 = ISSUE_BRMSG, bit 1 = ISSUE_EXP, bit 2 = ISSUE_LDS_DIRECT, bit 3 = ISSUE_LDS,
 *   bit 4 = ISSUE_VMEM_TEX, bit 5 = ISSUE_SCALAR, bit 6 = ISSUE_VALU,
 *   [bit 7 = reserved],
 *   bit 8 = STALL_BRMSG, bit 9 = STALL_EXP, bit 10 = STALL_LDS_DIRECT, bit 11 = STALL_LDS,
 *   bit 12 = STALL_VMEM_TEX, bit 13 = STALL_SCALAR, bit 14 = STALL_VALU
 */
inline uint32_t
pack_snapshot_ext_data_gfx12(uint16_t arb_state)
{
    uint32_t result  = 0;
    auto     set_bit = [&](rocprofiler_pc_sampling_snapshot_ext_field_id_t field, int hw_bit) {
        if(arb_state & (1u << hw_bit)) result |= (1u << field);
    };
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VALU, 6);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS, 3);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_LDS_DIRECT, 2);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_SCALAR, 5);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_VMEM_TEX, 4);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_EXP, 1);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_ISSUED_BRMSG_MISC, 0);

    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VALU, 14);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS, 11);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_LDS_DIRECT, 10);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_SCALAR, 13);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_VMEM_TEX, 12);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_EXP, 9);
    set_bit(ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_ARBITER_STATE_STALLED_BRMSG_MISC, 8);

    return result;
}

/**
 * @brief Pack GFX1250 ext_data: GFX12 arbiter state bits + LOCK_CONTENTION.
 *
 * @param arb_state       Arbiter state bits from perf_snapshot_data1[24:9]
 * @param lock_contention SAMPLING_LOCK_ERR bit (perf_snapshot_data bit 14)
 */
inline uint32_t
pack_snapshot_ext_data_gfx1250(uint16_t arb_state, bool lock_contention)
{
    uint32_t result = pack_snapshot_ext_data_gfx12(arb_state);
    if(lock_contention)
        result |= (1u << ROCPROFILER_PC_SAMPLING_SNAPSHOT_EXT_FIELD_ID_LOCK_CONTENTION);
    return result;
}

// --- copySample specializations for V0 (minimal, all architectures) ---

template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_host_trap_v1*>(sample);
    return copySampleCommon<rocprofiler_pc_sampling_record_v0_t>(s);
}

template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX950, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_host_trap_v1*>(sample);
    return copySampleCommon<rocprofiler_pc_sampling_record_v0_t>(s);
}

template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_host_trap_v1*>(sample);
    return copySampleCommon<rocprofiler_pc_sampling_record_v0_t>(s);
}

// --- copySample specializations for V1 (host-trap with hw_id, workgroup, wave_in_group) ---

template <>
inline rocprofiler_pc_sampling_record_v1_t
copySample<GFX9, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s     = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret   = copySampleCommon<rocprofiler_pc_sampling_record_v1_t>(s);
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    copyHwId<GFX9>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_v1_t
copySample<GFX950, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_v1_t
copySample<GFX11, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s            = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret          = copySampleCommon<rocprofiler_pc_sampling_record_v1_t>(s);
    ret.wave_in_group        = s.chiplet_and_wave_id & 0x3F;
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    // TODO: decode hw_id for GFX11 when hw_id register layout is available
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_v1_t
copySample<GFX12, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s     = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret   = copySampleCommon<rocprofiler_pc_sampling_record_v1_t>(s);
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    return ret;
}

// --- copySample specializations for V2 (stochastic with snapshot_information) ---

template <>
inline rocprofiler_pc_sampling_record_v2_t
copySample<GFX9, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_snapshot_v1*>(sample);

    auto perf_snapshot_data = s.perf_snapshot_data;
    auto valid              = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0) &
                                   ~EXTRACT_BITS(perf_snapshot_data, 26, 26));
    if(!valid)
    {
        // Invalid sample: value-initialize all fields to zero.
        // is_invalid_sample<rocprofiler_pc_sampling_record_v2_t>() relies on
        // timestamp == 0 as the invalid sentinel (see parser/correlation.hpp).
        return rocprofiler_pc_sampling_record_v2_t{};
    }

    auto ret          = copySampleCommon<rocprofiler_pc_sampling_record_v2_t>(s);
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    copyHwId<GFX9>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;

    // Populate snapshot_information
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type =
        translate_inst<GFX9>(EXTRACT_BITS(perf_snapshot_data, 6, 3));
    ret.snapshot_information.no_issue_reason =
        translate_reason<GFX9>(EXTRACT_BITS(perf_snapshot_data, 9, 7));
    ret.snapshot_information.wave_count = EXTRACT_BITS(s.perf_snapshot_data1, 5, 0);

    // Pack arbiter state into canonical layout
    uint16_t arb_state                     = EXTRACT_BITS(perf_snapshot_data, 25, 10);
    bool     dual_issue                    = EXTRACT_BITS(perf_snapshot_data, 2, 2);
    ret.snapshot_information.ext_data = pack_snapshot_ext_data_gfx9(arb_state, dual_issue);

    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_v2_t
copySample<GFX950, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_v2_t
copySample<GFX11, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& s            = *static_cast<const perf_sample_snapshot_v1*>(sample);
    auto        ret          = copySampleCommon<rocprofiler_pc_sampling_record_v2_t>(s);
    ret.wave_in_group        = s.chiplet_and_wave_id & 0x3F;
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    // TODO: decode snapshot_information fields for GFX11
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_v2_t
copySample<GFX12, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_snapshot_v1*>(sample);

    auto perf_snapshot_data = s.perf_snapshot_data;
    auto valid              = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // See GFX9 specialization above for the zero-init / timestamp == 0 sentinel rationale.
        return rocprofiler_pc_sampling_record_v2_t{};
    }

    auto ret          = copySampleCommon<rocprofiler_pc_sampling_record_v2_t>(s);
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;

    // Populate snapshot_information
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type =
        translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    ret.snapshot_information.no_issue_reason =
        translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));

    auto perf_snapshot_data1            = s.perf_snapshot_data1;
    ret.snapshot_information.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Pack arbiter state into canonical layout
    uint16_t arb_state                = EXTRACT_BITS(perf_snapshot_data1, 21, 6);
    ret.snapshot_information.ext_data = pack_snapshot_ext_data_gfx12(arb_state);

    return ret;
}

// --- copySample specializations for GFX1250 V0, V1, V2 (new API) ---

template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX1250, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    return copySample<GFX12, rocprofiler_pc_sampling_record_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_v1_t
copySample<GFX1250, rocprofiler_pc_sampling_record_v1_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s     = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret   = copySampleCommon<rocprofiler_pc_sampling_record_v1_t>(s);
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    // GFX1250 has chiplets
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    // ROCr uses same hw_id struct for both GFX12 and GFX1250
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_v2_t
copySample<GFX1250, rocprofiler_pc_sampling_record_v2_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_snapshot_v1*>(sample);

    auto perf_snapshot_data = s.perf_snapshot_data;
    // The sample is valid if perf_snapshot_data.valid == 1
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // See GFX9 specialization above for the zero-init / timestamp == 0 sentinel rationale.
        return rocprofiler_pc_sampling_record_v2_t{};
    }

    auto ret          = copySampleCommon<rocprofiler_pc_sampling_record_v2_t>(s);
    // GFX1250 has chiplets
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    // ROCr uses same hw_id struct for both GFX12 and GFX1250
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;

    // Populate snapshot_information
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type =
        translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    ret.snapshot_information.no_issue_reason =
        translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));

    auto perf_snapshot_data1            = s.perf_snapshot_data1;
    ret.snapshot_information.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Pack arbiter state + lock contention into canonical layout
    // GFX1250 arb_state is at bits [24:9] of perf_snapshot_data1 (GFX12 uses [21:6])
    uint16_t arb_state      = EXTRACT_BITS(perf_snapshot_data1, 24, 9);
    bool lock_contention    = EXTRACT_BITS(perf_snapshot_data, 14, 14);
    ret.snapshot_information.ext_data = pack_snapshot_ext_data_gfx1250(arb_state, lock_contention);

    // Sanity check: the wave_id of snapshot_data should match hw_id.wave_id. A mismatch
    // indicates a malformed HW record; drop the sample (zero-init makes is_invalid_sample()
    // filter it) rather than killing the entire process with ROCP_FATAL.
    auto sampled_wave_id = EXTRACT_BITS(perf_snapshot_data, 13, 9);
    if(sampled_wave_id != ret.hw_id.wave_id)
    {
        ROCP_ERROR << "sampled_wave_id: " << sampled_wave_id
                   << " mismatches the hw_id.wave_id: " << ret.hw_id.wave_id
                   << "; marking sample invalid";
        return rocprofiler_pc_sampling_record_v2_t{};
    }

    return ret;
}

// --- copySample specializations for GFX1250 V3 (host-trap with cluster info) ---

template <>
inline rocprofiler_pc_sampling_record_v3_t
copySample<GFX1250, rocprofiler_pc_sampling_record_v3_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    const auto& s     = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret   = copySampleCommon<rocprofiler_pc_sampling_record_v3_t>(s);
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    // GFX1250 has chiplets
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    // ROCr uses same hw_id struct for both GFX12 and GFX1250
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;
    // Cluster fields are zero-initialized by copySampleCommon (value-init)
    // and will be populated when cluster support is available in the hardware.
    return ret;
}

// --- copySample specialization for GFX1250 V4 (stochastic with cluster + memory counters) ---

template <>
inline rocprofiler_pc_sampling_record_v4_t
copySample<GFX1250, rocprofiler_pc_sampling_record_v4_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    const auto& s = *static_cast<const perf_sample_snapshot_v1*>(sample);

    auto perf_snapshot_data = s.perf_snapshot_data;
    // The sample is valid if perf_snapshot_data.valid == 1
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // See GFX9 specialization above for the zero-init / timestamp == 0 sentinel rationale.
        return rocprofiler_pc_sampling_record_v4_t{};
    }

    auto ret          = copySampleCommon<rocprofiler_pc_sampling_record_v4_t>(s);
    // GFX1250 has chiplets
    ret.hw_id.chiplet = s.chiplet_and_wave_id >> 8;
    ret.wave_in_group = s.chiplet_and_wave_id & 0x3F;
    // ROCr uses same hw_id struct for both GFX12 and GFX1250
    copyHwId<GFX12>(ret.hw_id, s.hw_id);
    ret.workgroup_position.x = s.workgroup_id_x;
    ret.workgroup_position.y = s.workgroup_id_y;
    ret.workgroup_position.z = s.workgroup_id_z;

    // Cluster fields are zero-initialized by copySampleCommon (value-init)
    // and will be populated when cluster support is available in the hardware.

    // Populate snapshot_information
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type =
        translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    ret.snapshot_information.no_issue_reason =
        translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));

    auto perf_snapshot_data1            = s.perf_snapshot_data1;
    ret.snapshot_information.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Pack arbiter state + lock contention into canonical layout
    // GFX1250 arb_state is at bits [24:9] of perf_snapshot_data1 (GFX12 uses [21:6])
    uint16_t arb_state      = EXTRACT_BITS(perf_snapshot_data1, 24, 9);
    bool lock_contention    = EXTRACT_BITS(perf_snapshot_data, 14, 14);
    ret.snapshot_information.ext_data = pack_snapshot_ext_data_gfx1250(arb_state, lock_contention);

    // Populate memory counters from perf_snapshot_data2 register
    auto perf_snapshot_data2         = s.perf_snapshot_data2;
    ret.memory_counters.load_count   = EXTRACT_BITS(perf_snapshot_data2, 5, 0);
    ret.memory_counters.store_count  = EXTRACT_BITS(perf_snapshot_data2, 11, 6);
    ret.memory_counters.bvh_count    = EXTRACT_BITS(perf_snapshot_data2, 14, 12);
    ret.memory_counters.sample_count = EXTRACT_BITS(perf_snapshot_data2, 20, 15);
    ret.memory_counters.ds_count     = EXTRACT_BITS(perf_snapshot_data2, 26, 21);
    ret.memory_counters.km_count     = EXTRACT_BITS(perf_snapshot_data2, 31, 27);
    // GFX1250-specific counters from perf_snapshot_data and perf_snapshot_data1
    ret.memory_counters.async_count  = EXTRACT_BITS(perf_snapshot_data, 25, 20);
    ret.memory_counters.tensor_count = EXTRACT_BITS(perf_snapshot_data, 31, 26);
    ret.memory_counters.xnack_count  = EXTRACT_BITS(perf_snapshot_data1, 31, 26);

    // Sanity check: the wave_id of snapshot_data should match hw_id.wave_id. A mismatch
    // indicates a malformed HW record; drop the sample (zero-init makes is_invalid_sample()
    // filter it) rather than killing the entire process with ROCP_FATAL.
    auto sampled_wave_id = EXTRACT_BITS(perf_snapshot_data, 13, 9);
    if(sampled_wave_id != ret.hw_id.wave_id)
    {
        ROCP_ERROR << "sampled_wave_id: " << sampled_wave_id
                   << " mismatches the hw_id.wave_id: " << ret.hw_id.wave_id
                   << "; marking sample invalid";
        return rocprofiler_pc_sampling_record_v4_t{};
    }

    return ret;
}

// =====================================================================================
// STOCHASTIC specializations for the host-trap-layout record kinds (V0, V1, V3)
//
// V0/V1/V3 carry no stochastic-only fields, so under the stochastic method they decode
// exactly like their host-trap counterparts -- the common leading fields (pc, exec_mask,
// workgroup, chiplet_and_wave_id, hw_id, timestamp) sit at byte-identical offsets in
// perf_sample_host_trap_v1 and perf_sample_snapshot_v1. The only extra work is honoring the
// hardware validity bit: stochastic hardware can emit invalid samples, so we zero-init the
// record (making is_invalid_sample<RecordT, STOCHASTIC>() detect it via timestamp == 0) and
// otherwise delegate to the host-trap specialization. The per-GFX validity bit mirrors the
// corresponding V2 specialization above.
// =====================================================================================

/**
 * @brief Per-GFX hardware validity test for a stochastic snapshot sample. Mirrors the
 * valid-bit logic used by the V2 copySample specializations.
 */
template <typename GFX>
inline bool
is_stochastic_snapshot_valid(const perf_sample_snapshot_v1& s);

template <>
inline bool
is_stochastic_snapshot_valid<GFX9>(const perf_sample_snapshot_v1& s)
{
    return static_cast<bool>(EXTRACT_BITS(s.perf_snapshot_data, 0, 0) &
                             ~EXTRACT_BITS(s.perf_snapshot_data, 26, 26));
}

template <>
inline bool
is_stochastic_snapshot_valid<GFX950>(const perf_sample_snapshot_v1& s)
{
    return is_stochastic_snapshot_valid<GFX9>(s);
}

template <>
inline bool
is_stochastic_snapshot_valid<GFX11>(const perf_sample_snapshot_v1& /*s*/)
{
    // GFX11 stochastic decoding is not yet implemented (see V2 GFX11 specialization, which
    // performs no validity check); treat every sample as valid for now.
    return true;
}

template <>
inline bool
is_stochastic_snapshot_valid<GFX12>(const perf_sample_snapshot_v1& s)
{
    return static_cast<bool>(EXTRACT_BITS(s.perf_snapshot_data, 0, 0));
}

template <>
inline bool
is_stochastic_snapshot_valid<GFX1250>(const perf_sample_snapshot_v1& s)
{
    return static_cast<bool>(EXTRACT_BITS(s.perf_snapshot_data, 0, 0));
}

#define STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFXIP, RECORD_T)                                          \
    template <>                                                                                     \
    inline RECORD_T copySample<GFXIP, RECORD_T, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(         \
        const void* sample)                                                                         \
    {                                                                                               \
        const auto& s = *static_cast<const perf_sample_snapshot_v1*>(sample);                       \
        if(!is_stochastic_snapshot_valid<GFXIP>(s)) return RECORD_T{};                              \
        return copySample<GFXIP, RECORD_T, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);       \
    }

// V0 stochastic (all architectures)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX9, rocprofiler_pc_sampling_record_v0_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX950, rocprofiler_pc_sampling_record_v0_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX11, rocprofiler_pc_sampling_record_v0_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX12, rocprofiler_pc_sampling_record_v0_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX1250, rocprofiler_pc_sampling_record_v0_t)

// V1 stochastic (all architectures)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX9, rocprofiler_pc_sampling_record_v1_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX950, rocprofiler_pc_sampling_record_v1_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX11, rocprofiler_pc_sampling_record_v1_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX12, rocprofiler_pc_sampling_record_v1_t)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX1250, rocprofiler_pc_sampling_record_v1_t)

// V3 stochastic (GFX1250 only)
STOCHASTIC_PASSTHROUGH_COPYSAMPLE(GFX1250, rocprofiler_pc_sampling_record_v3_t)

#undef STOCHASTIC_PASSTHROUGH_COPYSAMPLE

// =====================================================================================
// correct_pc_address
// =====================================================================================

/**
 * @brief Host trap V0 sample for GFX1250
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX1250, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(const void* sample)
{
    // Host-trap samples are same for GFX12 and GFX1250
    auto ret = copySample<GFX12, rocprofiler_pc_sampling_record_host_trap_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>(sample);
    // Only difference are chiplets that exist on GFX1250
    copyChipletId<GFX1250>(ret, *static_cast<const perf_sample_snapshot_v1*>(sample));
    return ret;
}

/**
 * @brief Stochastic sample for GFX1250
 */
template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX1250, rocprofiler_pc_sampling_record_stochastic_v0_t, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>(const void* sample)
{
    // Differences compared to the GFX12:
    // - SAMPLING_LOCK_ERR introduced in the GFX1250 and means:
    //   A wave tried taking a snapshot but was unable to as the previous snapshot had not yet been
    //   read.
    // - ARB_STATE_* begins at different offset
    // - XCNT, ASYNC_CNT, TENSOR_CNT introduced in GFX1250.
    // - GFX1250 uses chiplets
    // Due to all differences and potential adaptation to the future subfamily members,
    // we decided to have a separate implementation of copySample for GFX1250 stochastic.

    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);

    // Extracting data from the perf_snapshot_data register
    auto perf_snapshot_data = sample_.perf_snapshot_data;
    // The sample is valid  if perf_snapshot_data.valid == 1
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // To reduce refactoring of the PC sampling parser, we agreed to internally represent
        // invalid samples with `rocprofiler_pc_sampling_record_stochastic_v0_t` with size 0.
        // Eventually, those records are replaced with rocprofiler_pc_sampling_record_invalid_t
        // and placed into the SDK buffer consumed by the end tool.
        rocprofiler_pc_sampling_record_stochastic_v0_t invalid{};
        invalid.size = 0;
        // No need to further process invalid samples
        return invalid;
    }

    auto ret = copySampleHeader<rocprofiler_pc_sampling_record_stochastic_v0_t>(sample_);
    // GFX1250 has chiplets
    copyChipletId<GFX1250>(ret, sample_);
    // ROCr uses same hw_id struct for both GFX12 and GFX1250.
    copyHwId<GFX12>(ret.hw_id, sample_.hw_id);

    // wave issued an instruction
    ret.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    // TODO: we need to handle new instruction type I think
    // type of issued instruction, valid only if `ret.wave_issued` is true.
    ret.inst_type = translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    // reason for not issuing an instruction, valid only if `ret.wave_issued` is false
    ret.snapshot.reason_not_issued =
        translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));

    // Sampling lock error indicates that reading this sample took to long
    // so at least one wave was locked out from taking a sample.
    ret.snapshot.sampling_lock_error = EXTRACT_BITS(perf_snapshot_data, 14, 14);

    // arbiter state information
    auto     perf_snapshot_data1 = sample_.perf_snapshot_data1;
    uint16_t arb_state           = EXTRACT_BITS(perf_snapshot_data1, 24, 9);

    ret.snapshot.arb_state_issue_brmsg      = EXTRACT_BITS(arb_state, 0, 0);
    ret.snapshot.arb_state_issue_exp        = EXTRACT_BITS(arb_state, 1, 1);
    ret.snapshot.arb_state_issue_lds_direct = EXTRACT_BITS(arb_state, 2, 2);
    ret.snapshot.arb_state_issue_lds        = EXTRACT_BITS(arb_state, 3, 3);
    ret.snapshot.arb_state_issue_vmem_tex   = EXTRACT_BITS(arb_state, 4, 4);
    ret.snapshot.arb_state_issue_scalar     = EXTRACT_BITS(arb_state, 5, 5);
    ret.snapshot.arb_state_issue_valu       = EXTRACT_BITS(arb_state, 6, 6);

    ret.snapshot.arb_state_stall_brmsg      = EXTRACT_BITS(arb_state, 8, 8);
    ret.snapshot.arb_state_stall_exp        = EXTRACT_BITS(arb_state, 9, 9);
    ret.snapshot.arb_state_stall_lds_direct = EXTRACT_BITS(arb_state, 10, 10);
    ret.snapshot.arb_state_stall_lds        = EXTRACT_BITS(arb_state, 11, 11);
    ret.snapshot.arb_state_stall_vmem_tex   = EXTRACT_BITS(arb_state, 12, 12);
    ret.snapshot.arb_state_stall_scalar     = EXTRACT_BITS(arb_state, 13, 13);
    ret.snapshot.arb_state_stall_valu       = EXTRACT_BITS(arb_state, 14, 14);

    ret.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Memory counters exist on GFX1250.
    ret.flags.has_memory_counter = true;
    // Extracting memory counters from the perf_snapshot_data2 register
    auto perf_snapshot_data2       = sample_.perf_snapshot_data2;
    ret.memory_counters.load_cnt   = EXTRACT_BITS(perf_snapshot_data2, 5, 0);
    ret.memory_counters.store_cnt  = EXTRACT_BITS(perf_snapshot_data2, 11, 6);
    ret.memory_counters.bvh_cnt    = EXTRACT_BITS(perf_snapshot_data2, 14, 12);
    ret.memory_counters.sample_cnt = EXTRACT_BITS(perf_snapshot_data2, 20, 15);
    ret.memory_counters.ds_cnt     = EXTRACT_BITS(perf_snapshot_data2, 26, 21);
    ret.memory_counters.km_cnt     = EXTRACT_BITS(perf_snapshot_data2, 31, 27);
    // counters available in perf_snapshot_data
    ret.memory_counters.async_cnt  = EXTRACT_BITS(perf_snapshot_data, 25, 20);
    ret.memory_counters.tensor_cnt = EXTRACT_BITS(perf_snapshot_data, 31, 26);
    // counters available in perf_snapshot_data1
    ret.memory_counters.xnack_cnt = EXTRACT_BITS(perf_snapshot_data1, 31, 26);

    // Sanity check: the wave_id of snapshot_data should match hw_id.wave_id. A mismatch
    // indicates a malformed HW record; drop the sample (size == 0 makes is_invalid_sample()
    // filter it) rather than killing the entire process with ROCP_FATAL.
    auto sampled_wave_id = EXTRACT_BITS(perf_snapshot_data, 13, 9);
    if(sampled_wave_id != ret.hw_id.wave_id)
    {
        ROCP_ERROR << "sampled_wave_id: " << sampled_wave_id
                   << " mismatches the hw_id.wave_id: " << ret.hw_id.wave_id
                   << "; marking sample invalid";
        rocprofiler_pc_sampling_record_stochastic_v0_t invalid{};
        invalid.size = 0;
        return invalid;
    }

    return ret;
}

/**
 * @brief The default implementation assumes no correction is needed.
 */
template <typename GFX, typename PcSamplingRecordT>
inline rocprofiler_address_t
correct_pc_address(const perf_sample_snapshot_v1* sample)
{
    return rocprofiler_address_t{.value = sample->pc};
}

/**
 * @brief GFX950 specific implementation of the PC address correction.
 */
template <>
inline rocprofiler_address_t
correct_pc_address<GFX950, rocprofiler_pc_sampling_record_stochastic_v0_t>(
    const perf_sample_snapshot_v1* sample)
{
    // If mid_macro bit is 1, then reg spec says we need to subtract 2 dwords from the PC address.
    auto mid_macro = static_cast<bool>(EXTRACT_BITS(sample->perf_snapshot_data1, 31, 31));
    if(mid_macro)
    {
        return rocprofiler_address_t{.value = sample->pc - 2 * sizeof(uint32_t)};
    }
    else
    {
        return rocprofiler_address_t{.value = sample->pc};
    }
}

/**
 * @brief GFX950 PC address correction for V2 records (stochastic on GFX950).
 */
template <>
inline rocprofiler_address_t
correct_pc_address<GFX950, rocprofiler_pc_sampling_record_v2_t>(
    const perf_sample_snapshot_v1* sample)
{
    auto mid_macro = static_cast<bool>(EXTRACT_BITS(sample->perf_snapshot_data1, 31, 31));
    if(mid_macro)
    {
        return rocprofiler_address_t{.value = sample->pc - 2 * sizeof(uint32_t)};
    }
    else
    {
        return rocprofiler_address_t{.value = sample->pc};
    }
}

#undef EXTRACT_BITS
