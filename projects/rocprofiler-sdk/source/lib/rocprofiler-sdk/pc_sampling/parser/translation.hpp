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

template <typename GFX9, typename HwIdT>
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

#undef EXTRACT_BITS
