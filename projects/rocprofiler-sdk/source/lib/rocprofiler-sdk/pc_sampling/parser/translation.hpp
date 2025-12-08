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

template <typename PcSamplingRecordT, typename SType>
inline PcSamplingRecordT
copySampleHeader(const SType& sample)
{
    PcSamplingRecordT ret;
    // zero out all record fields
    std::memset(&ret, 0, sizeof(PcSamplingRecordT));

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

template <typename GFX, typename PcSamplingRecordT>
inline PcSamplingRecordT
copySample(const void* sample);

/**
 * @brief Host trap V0 sample for GFX9
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_host_trap_v0_t>(const void* sample)
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
copySample<GFX9, rocprofiler_pc_sampling_record_stochastic_v0_t>(const void* sample)
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
copySample<GFX950, rocprofiler_pc_sampling_record_host_trap_v0_t>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_host_trap_v0_t>(sample);
}

template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX950, rocprofiler_pc_sampling_record_stochastic_v0_t>(const void* sample)
{
    return copySample<GFX9, rocprofiler_pc_sampling_record_stochastic_v0_t>(sample);
}

/**
 * @brief Host trap V0 sample for GFX11
 */
template <>
inline rocprofiler_pc_sampling_record_host_trap_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_host_trap_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_host_trap_v0_t>(sample_);
    // TODO: decode other fields.
    return ret;
}

// TODO: implement stochastic for GFX11
template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_stochastic_v0_t>(const void* sample)
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
copySample<GFX12, rocprofiler_pc_sampling_record_host_trap_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_host_trap_v0_t>(sample_);
    copyHwId<GFX12>(ret.hw_id, sample_.hw_id);
    return ret;
}

template <>
inline rocprofiler_pc_sampling_record_stochastic_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_stochastic_v0_t>(const void* sample)
{
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

// =============================================================================
// Direct encoding functions for unified v0 record format
// These encode directly from hardware sample data without intermediate structures
// =============================================================================

/**
 * @brief Encode hw_id directly from GFX9 hardware sample into compact format
 */
inline uint64_t
encode_hw_id_from_gfx9_sample(uint64_t chiplet_and_wave_id, uint32_t hw_id_reg)
{
    uint64_t encoded = 0;

    // Set hardware version (lower 6 bits)
    encoded |= (ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX9 & 0x3F);

    // Extract chiplet from chiplet_and_wave_id
    uint8_t chiplet = chiplet_and_wave_id >> 8;
    encoded |= (static_cast<uint64_t>(chiplet) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_CHIPLET);

    // Extract and encode fields from hw_id_reg
    uint8_t wave_id = EXTRACT_BITS(hw_id_reg, 3, 0);
    encoded |= (static_cast<uint64_t>(wave_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_WAVE_ID);

    uint8_t simd_id = EXTRACT_BITS(hw_id_reg, 5, 4);
    encoded |= (static_cast<uint64_t>(simd_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_SIMD_ID);

    uint8_t pipe_id = EXTRACT_BITS(hw_id_reg, 7, 6);
    encoded |= (static_cast<uint64_t>(pipe_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_PIPE_ID);

    uint8_t cu_id = EXTRACT_BITS(hw_id_reg, 11, 8);
    encoded |= (static_cast<uint64_t>(cu_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_CU_ID);

    uint8_t shader_array_id = EXTRACT_BITS(hw_id_reg, 12, 12);
    encoded |= (static_cast<uint64_t>(shader_array_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_SHADER_ARRAY_ID);

    uint8_t shader_engine_id = EXTRACT_BITS(hw_id_reg, 15, 13);
    encoded |= (static_cast<uint64_t>(shader_engine_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_SHADER_ENGINE_ID);

    uint8_t workgroup_id = EXTRACT_BITS(hw_id_reg, 19, 16);
    encoded |= (static_cast<uint64_t>(workgroup_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_WORKGROUP_ID);

    uint8_t vm_id = EXTRACT_BITS(hw_id_reg, 23, 20);
    encoded |= (static_cast<uint64_t>(vm_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_VM_ID);

    uint8_t queue_id = EXTRACT_BITS(hw_id_reg, 26, 24);
    encoded |= (static_cast<uint64_t>(queue_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_QUEUE_ID);

    uint8_t microengine_id = EXTRACT_BITS(hw_id_reg, 31, 30);
    encoded |= (static_cast<uint64_t>(microengine_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX9_OFFSET_MICROENGINE_ID);

    return encoded;
}

/**
 * @brief Encode hw_id directly from GFX12 hardware sample into compact format
 */
inline uint64_t
encode_hw_id_from_gfx12_sample(uint64_t chiplet_and_wave_id, uint32_t hw_id_reg)
{
    uint64_t encoded = 0;

    // Set hardware version (lower 6 bits)
    encoded |= (ROCPROFILER_PC_SAMPLING_HARDWARE_VERSION_GFX12 & 0x3F);

    // Extract chiplet from chiplet_and_wave_id
    uint8_t chiplet = chiplet_and_wave_id >> 8;
    encoded |= (static_cast<uint64_t>(chiplet) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_CHIPLET);

    // Extract and encode fields from hw_id_reg
    uint8_t wave_id = EXTRACT_BITS(hw_id_reg, 4, 0);
    encoded |= (static_cast<uint64_t>(wave_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_WAVE_ID);

    uint8_t queue_id = EXTRACT_BITS(hw_id_reg, 8, 5);
    encoded |= (static_cast<uint64_t>(queue_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_QUEUE_ID);

    uint8_t wgp_id = EXTRACT_BITS(hw_id_reg, 13, 10);
    encoded |= (static_cast<uint64_t>(wgp_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_WGP_ID);

    uint8_t simd_id = EXTRACT_BITS(hw_id_reg, 15, 14);
    encoded |= (static_cast<uint64_t>(simd_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_SIMD_ID);

    uint8_t shader_array_id = EXTRACT_BITS(hw_id_reg, 16, 16);
    encoded |= (static_cast<uint64_t>(shader_array_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_SHADER_ARRAY_ID);

    uint8_t microengine_id = EXTRACT_BITS(hw_id_reg, 17, 17);
    encoded |= (static_cast<uint64_t>(microengine_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_MICROENGINE_ID);

    uint8_t shader_engine_id = EXTRACT_BITS(hw_id_reg, 19, 18);
    encoded |= (static_cast<uint64_t>(shader_engine_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_SHADER_ENGINE_ID);

    uint8_t pipe_id = EXTRACT_BITS(hw_id_reg, 21, 20);
    encoded |= (static_cast<uint64_t>(pipe_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_PIPE_ID);

    uint8_t workgroup_id = EXTRACT_BITS(hw_id_reg, 27, 23);
    encoded |= (static_cast<uint64_t>(workgroup_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_WORKGROUP_ID);

    uint8_t vm_id = EXTRACT_BITS(hw_id_reg, 31, 28);
    encoded |= (static_cast<uint64_t>(vm_id) << ROCPROFILER_PC_SAMPLING_HW_ID_GFX12_OFFSET_VM_ID);

    return encoded;
}

/**
 * @brief Encode arbiter_state directly from GFX9 hardware sample into compact format
 */
inline uint32_t
encode_arbiter_state_from_gfx9_sample(uint32_t perf_snapshot_data)
{
    uint32_t encoded = 0;
    uint16_t arb_state = EXTRACT_BITS(perf_snapshot_data, 25, 10);

    // Issue bits
    encoded |= (EXTRACT_BITS(arb_state, 7, 7) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_VALU);
    encoded |= (EXTRACT_BITS(arb_state, 6, 6) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MATRIX);
    encoded |= (EXTRACT_BITS(arb_state, 3, 3) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_LDS);
    encoded |= (EXTRACT_BITS(arb_state, 5, 5) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_SCALAR);
    encoded |= (EXTRACT_BITS(arb_state, 4, 4) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_TEX);
    encoded |= (EXTRACT_BITS(arb_state, 2, 2) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_FLAT);
    encoded |= (EXTRACT_BITS(arb_state, 1, 1) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_EXP);
    encoded |= (EXTRACT_BITS(arb_state, 0, 0) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_ISSUE_MISC);

    // Stall bits
    encoded |= (EXTRACT_BITS(arb_state, 15, 15) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_VALU);
    encoded |= (EXTRACT_BITS(arb_state, 14, 14) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MATRIX);
    encoded |= (EXTRACT_BITS(arb_state, 11, 11) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_LDS);
    encoded |= (EXTRACT_BITS(arb_state, 13, 13) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_SCALAR);
    encoded |= (EXTRACT_BITS(arb_state, 12, 12) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_TEX);
    encoded |= (EXTRACT_BITS(arb_state, 10, 10) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_FLAT);
    encoded |= (EXTRACT_BITS(arb_state, 9, 9) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_EXP);
    encoded |= (EXTRACT_BITS(arb_state, 8, 8) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_STALL_MISC);

    // Dual issue (from different location in perf_snapshot_data)
    uint8_t dual_issue_valu = EXTRACT_BITS(perf_snapshot_data, 2, 2);
    encoded |= (dual_issue_valu << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX9_FIELD_OFFSET_DUAL_ISSUE_VALU);

    return encoded;
}

/**
 * @brief Encode arbiter_state directly from GFX12 hardware sample into compact format
 */
inline uint32_t
encode_arbiter_state_from_gfx12_sample(uint32_t perf_snapshot_data1)
{
    uint32_t encoded = 0;
    uint16_t arb_state = EXTRACT_BITS(perf_snapshot_data1, 21, 6);

    // Issue bits
    encoded |= (EXTRACT_BITS(arb_state, 6, 6) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VALU);
    encoded |= (EXTRACT_BITS(arb_state, 3, 3) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS);
    encoded |= (EXTRACT_BITS(arb_state, 2, 2) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_LDS_DIRECT);
    encoded |= (EXTRACT_BITS(arb_state, 5, 5) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_SCALAR);
    encoded |= (EXTRACT_BITS(arb_state, 4, 4) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_VMEM);
    encoded |= (EXTRACT_BITS(arb_state, 1, 1) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_EXP);
    encoded |= (EXTRACT_BITS(arb_state, 0, 0) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_ISSUE_BRMSG);

    // Stall bits
    encoded |= (EXTRACT_BITS(arb_state, 14, 14) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VALU);
    encoded |= (EXTRACT_BITS(arb_state, 11, 11) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS);
    encoded |= (EXTRACT_BITS(arb_state, 10, 10) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_LDS_DIRECT);
    encoded |= (EXTRACT_BITS(arb_state, 13, 13) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_SCALAR);
    encoded |= (EXTRACT_BITS(arb_state, 12, 12) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_VMEM);
    encoded |= (EXTRACT_BITS(arb_state, 9, 9) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_EXP);
    encoded |= (EXTRACT_BITS(arb_state, 8, 8) << ROCPROFILER_PC_SAMPLING_ARBITER_STATE_GFX12_FIELD_OFFSET_STALL_BRMSG);

    return encoded;
}

// =============================================================================
// copySample implementations for unified v0 record format
// These directly encode from hardware samples
// =============================================================================

/**
 * @brief GFX9 Host Trap → Unified v0 Record
 */
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);

    rocprofiler_pc_sampling_record_v0_t ret;
    std::memset(&ret, 0, sizeof(ret));

    // Set size
    ret.size = sizeof(rocprofiler_pc_sampling_record_v0_t);

    // Common fields
    ret.exec_mask = sample_.exec_mask;
    ret.workgroup_position.x = sample_.workgroup_id_x;
    ret.workgroup_position.y = sample_.workgroup_id_y;
    ret.workgroup_position.z = sample_.workgroup_id_z;
    ret.timestamp = sample_.timestamp;
    ret.wave_in_group = sample_.chiplet_and_wave_id & 0x3F;

    // Directly encode hw_id from hardware sample
    ret.hw_id.value = encode_hw_id_from_gfx9_sample(sample_.chiplet_and_wave_id, sample_.hw_id);

    // Host trap doesn't have snapshot information, so no flags set
    ret.flags.value = 0;

    // dispatch_id and correlation_id will be filled by add_upcoming_samples
    return ret;
}

/**
 * @brief GFX9 Stochastic → Unified v0 Record
 */
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX9, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);

    // Check validity first
    auto perf_snapshot_data = sample_.perf_snapshot_data;
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0) &
                                   ~EXTRACT_BITS(perf_snapshot_data, 26, 26));
    if(!valid)
    {
        // Invalid sample - mark with flag
        rocprofiler_pc_sampling_record_v0_t invalid{};
        invalid.size = sizeof(rocprofiler_pc_sampling_record_v0_t);
        invalid.flags.value = ROCPROFILER_PC_SAMPLING_RECORD_FLAG_IS_INVALID;
        return invalid;
    }

    rocprofiler_pc_sampling_record_v0_t ret;
    std::memset(&ret, 0, sizeof(ret));

    // Set size
    ret.size = sizeof(rocprofiler_pc_sampling_record_v0_t);

    // Common fields
    ret.exec_mask = sample_.exec_mask;
    ret.workgroup_position.x = sample_.workgroup_id_x;
    ret.workgroup_position.y = sample_.workgroup_id_y;
    ret.workgroup_position.z = sample_.workgroup_id_z;
    ret.timestamp = sample_.timestamp;
    ret.wave_in_group = sample_.chiplet_and_wave_id & 0x3F;

    // Directly encode hw_id from hardware sample
    ret.hw_id.value = encode_hw_id_from_gfx9_sample(sample_.chiplet_and_wave_id, sample_.hw_id);

    // Snapshot information (stochastic-specific)
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type = translate_inst<GFX9>(EXTRACT_BITS(perf_snapshot_data, 6, 3));
    ret.snapshot_information.no_issue_reason = translate_reason<GFX9>(EXTRACT_BITS(perf_snapshot_data, 9, 7));
    ret.snapshot_information.wave_count = EXTRACT_BITS(sample_.perf_snapshot_data1, 5, 0);

    // Directly encode arbiter_state from hardware sample
    ret.snapshot_information.arbiter_state = encode_arbiter_state_from_gfx9_sample(perf_snapshot_data);

    // Set flags - has snapshot information
    ret.flags.value = ROCPROFILER_PC_SAMPLING_RECORD_FLAG_HAS_SNAPSHOT_INFORMATION;
    // No memory counters on GFX9

    return ret;
}

/**
 * @brief GFX12 Host Trap → Unified v0 Record
 */
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_host_trap_v1*>(sample);

    rocprofiler_pc_sampling_record_v0_t ret;
    std::memset(&ret, 0, sizeof(ret));

    // Set size
    ret.size = sizeof(rocprofiler_pc_sampling_record_v0_t);

    // Common fields
    ret.exec_mask = sample_.exec_mask;
    ret.workgroup_position.x = sample_.workgroup_id_x;
    ret.workgroup_position.y = sample_.workgroup_id_y;
    ret.workgroup_position.z = sample_.workgroup_id_z;
    ret.timestamp = sample_.timestamp;
    ret.wave_in_group = sample_.chiplet_and_wave_id & 0x3F;

    // Directly encode hw_id from hardware sample
    ret.hw_id.value = encode_hw_id_from_gfx12_sample(sample_.chiplet_and_wave_id, sample_.hw_id);

    // Host trap doesn't have snapshot information, so no flags set
    ret.flags.value = 0;

    return ret;
}

/**
 * @brief GFX12 Stochastic → Unified v0 Record
 */
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX12, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    const auto& sample_ = *static_cast<const perf_sample_snapshot_v1*>(sample);

    // Check validity first
    auto perf_snapshot_data = sample_.perf_snapshot_data;
    auto valid = static_cast<bool>(EXTRACT_BITS(perf_snapshot_data, 0, 0));
    if(!valid)
    {
        // Invalid sample - mark with flag
        rocprofiler_pc_sampling_record_v0_t invalid{};
        invalid.size = sizeof(rocprofiler_pc_sampling_record_v0_t);
        invalid.flags.value = ROCPROFILER_PC_SAMPLING_RECORD_FLAG_IS_INVALID;
        return invalid;
    }

    rocprofiler_pc_sampling_record_v0_t ret;
    std::memset(&ret, 0, sizeof(ret));

    // Set size
    ret.size = sizeof(rocprofiler_pc_sampling_record_v0_t);

    // Common fields
    ret.exec_mask = sample_.exec_mask;
    ret.workgroup_position.x = sample_.workgroup_id_x;
    ret.workgroup_position.y = sample_.workgroup_id_y;
    ret.workgroup_position.z = sample_.workgroup_id_z;
    ret.timestamp = sample_.timestamp;
    ret.wave_in_group = sample_.chiplet_and_wave_id & 0x3F;

    // Directly encode hw_id from hardware sample
    ret.hw_id.value = encode_hw_id_from_gfx12_sample(sample_.chiplet_and_wave_id, sample_.hw_id);

    // Snapshot information (stochastic-specific)
    auto perf_snapshot_data1 = sample_.perf_snapshot_data1;
    ret.snapshot_information.wave_issued = EXTRACT_BITS(perf_snapshot_data, 1, 1);
    ret.snapshot_information.instruction_type = translate_inst<GFX12>(EXTRACT_BITS(perf_snapshot_data, 5, 2));
    ret.snapshot_information.no_issue_reason = translate_reason<GFX12>(EXTRACT_BITS(perf_snapshot_data, 8, 6));
    ret.snapshot_information.wave_count = EXTRACT_BITS(perf_snapshot_data1, 5, 0);

    // Directly encode arbiter_state from hardware sample
    ret.snapshot_information.arbiter_state = encode_arbiter_state_from_gfx12_sample(perf_snapshot_data1);

    // Set flags - has snapshot information and memory counters
    ret.flags.value = ROCPROFILER_PC_SAMPLING_RECORD_FLAG_HAS_SNAPSHOT_INFORMATION |
                      ROCPROFILER_PC_SAMPLING_RECORD_FLAG_HAS_MEMORY_COUNTERS;

    // Memory counters (GFX12 specific)
    auto perf_snapshot_data2 = sample_.perf_snapshot_data2;
    ret.memory_counters.gfx12.load_cnt   = EXTRACT_BITS(perf_snapshot_data2, 5, 0);
    ret.memory_counters.gfx12.store_cnt  = EXTRACT_BITS(perf_snapshot_data2, 11, 6);
    ret.memory_counters.gfx12.bvh_cnt    = EXTRACT_BITS(perf_snapshot_data2, 14, 12);
    ret.memory_counters.gfx12.sample_cnt = EXTRACT_BITS(perf_snapshot_data2, 20, 15);
    ret.memory_counters.gfx12.ds_cnt     = EXTRACT_BITS(perf_snapshot_data2, 26, 21);
    ret.memory_counters.gfx12.km_cnt     = EXTRACT_BITS(perf_snapshot_data2, 31, 27);

    return ret;
}

// GFX950 uses the same implementation as GFX9
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX950, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    // For GFX950, use GFX9 encoding
    // Note: This assumes GFX950 uses same format as GFX9
    // Adjust if GFX950 has different encoding
    return copySample<GFX9, rocprofiler_pc_sampling_record_v0_t>(sample);
}

// GFX11 uses similar implementation to GFX9
template <>
inline rocprofiler_pc_sampling_record_v0_t
copySample<GFX11, rocprofiler_pc_sampling_record_v0_t>(const void* sample)
{
    // For GFX11, use GFX9 encoding for now
    // TODO: Verify if GFX11 needs different encoding
    return copySample<GFX9, rocprofiler_pc_sampling_record_v0_t>(sample);
}

#undef EXTRACT_BITS
