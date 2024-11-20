// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <array>
#include <cstdint>
#include <cstring>

#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx11.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx9.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/parser_types.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.h"
#include "lib/rocprofiler-sdk/pc_sampling/parser/stochastic_records.h"

// TODO: refactor the commented code for stochastic sampling

// template <typename gfx>
// inline rocprofiler_pc_sampling_record_t
// copyStochasticSample(const perf_sample_snapshot_v1& sample);

// template <>
// inline rocprofiler_pc_sampling_record_t
// copyStochasticSample<GFX9>(const perf_sample_snapshot_v1& sample)
// {
//     rocprofiler_pc_sampling_record_t ret = copySampleHeader<perf_sample_snapshot_v1>(sample);
//     ret.flags.valid = sample.perf_snapshot_data & (~sample.perf_snapshot_data >> 26) & 0x1;
//     // Check wave_id matches snapshot_wave_id

//     ret.flags.has_wave_cnt     = true;
//     ret.flags.has_stall_reason = true;

//     ret.wave_count = sample.perf_snapshot_data1 & 0x3F;

//     ret.wave_issued                = sample.perf_snapshot_data >> 1;
//     ret.snapshot.dual_issue_valu   = sample.perf_snapshot_data >> 2;
//     ret.snapshot.inst_type         = sample.perf_snapshot_data >> 3;
//     ret.snapshot.reason_not_issued = (sample.perf_snapshot_data >> 7) & 0x7;
//     ret.snapshot.arb_state_issue   = (sample.perf_snapshot_data >> 10) & 0xFF;
//     ret.snapshot.arb_state_stall   = (sample.perf_snapshot_data >> 18) & 0xFF;
//     ret.reserved                   = 0;
//     return ret;
// }

// template <>
// inline rocprofiler_pc_sampling_record_t
// copyStochasticSample<GFX11>(const perf_sample_snapshot_v1& sample)
// {
//     rocprofiler_pc_sampling_record_t ret = copySampleHeader<perf_sample_snapshot_v1>(sample);
//     ret.flags.valid = sample.perf_snapshot_data & (~sample.perf_snapshot_data >> 23) & 0x1;
//     // Check wave_id matches snapshot_wave_id

//     ret.flags.has_stall_reason = true;

//     ret.wave_issued                = sample.perf_snapshot_data >> 1;
//     ret.snapshot.inst_type         = sample.perf_snapshot_data >> 2;
//     ret.snapshot.reason_not_issued = (sample.perf_snapshot_data >> 6) & 0x7;
//     ret.snapshot.arb_state_issue   = (sample.perf_snapshot_data >> 9) & 0x7F;
//     ret.snapshot.arb_state_stall   = (sample.perf_snapshot_data >> 16) & 0x7F;
//     ret.snapshot.dual_issue_valu   = false;
//     ret.reserved                   = 0;
//     return ret;
// }

// #define BITSHIFT(sname) out |= ((in >> GFX::sname) & 1) << PCSAMPLE::sname

// template <typename GFX>
// inline int
// translate_arb(int in)
// {
//     size_t out = 0;
//     BITSHIFT(ISSUE_VALU);
//     BITSHIFT(ISSUE_MATRIX);
//     BITSHIFT(ISSUE_LDS);
//     BITSHIFT(ISSUE_LDS_DIRECT);
//     BITSHIFT(ISSUE_SCALAR);
//     BITSHIFT(ISSUE_VMEM_TEX);
//     BITSHIFT(ISSUE_FLAT);
//     BITSHIFT(ISSUE_EXP);
//     BITSHIFT(ISSUE_MISC);
//     BITSHIFT(ISSUE_BRMSG);
//     return out & 0x3FF;
// }

// #undef BITSHIFT

// #define LUTOVERLOAD(sname) this->operator[](GFX::sname) = PCSAMPLE::sname

// template <typename GFX>
// class GFX_REASON_LUT : public std::array<int, 32>
// {
// public:
//     GFX_REASON_LUT()
//     {
//         std::memset(data(), 0, size() * sizeof(int));
//         LUTOVERLOAD(REASON_NOT_AVAILABLE);
//         LUTOVERLOAD(REASON_ALU);
//         LUTOVERLOAD(REASON_WAITCNT);
//         LUTOVERLOAD(REASON_INTERNAL);
//         LUTOVERLOAD(REASON_BARRIER);
//         LUTOVERLOAD(REASON_ARBITER);
//         LUTOVERLOAD(REASON_EX_STALL);
//         LUTOVERLOAD(REASON_OTHER_WAIT);
//         LUTOVERLOAD(REASON_SLEEP);
//     }
// };

// template <typename GFX>
// class GFX_INST_LUT : public std::array<int, 32>
// {
// public:
//     GFX_INST_LUT()
//     {
//         std::memset(data(), 0, size() * sizeof(int));
//         LUTOVERLOAD(TYPE_VALU);
//         LUTOVERLOAD(TYPE_MATRIX);
//         LUTOVERLOAD(TYPE_SCALAR);
//         LUTOVERLOAD(TYPE_TEX);
//         LUTOVERLOAD(TYPE_LDS);
//         LUTOVERLOAD(TYPE_LDS_DIRECT);
//         LUTOVERLOAD(TYPE_FLAT);
//         LUTOVERLOAD(TYPE_EXP);
//         LUTOVERLOAD(TYPE_MESSAGE);
//         LUTOVERLOAD(TYPE_BARRIER);
//         LUTOVERLOAD(TYPE_BRANCH_NOT_TAKEN);
//         LUTOVERLOAD(TYPE_BRANCH_TAKEN);
//         LUTOVERLOAD(TYPE_JUMP);
//         LUTOVERLOAD(TYPE_OTHER);
//         LUTOVERLOAD(TYPE_NO_INST);
//         LUTOVERLOAD(TYPE_DUAL_VALU);
//     }
// };

// template <typename GFX>
// inline int
// translate_reason(int in)
// {
//     static GFX_REASON_LUT<GFX> lut;
//     return lut[in & 0x1F];
// }

// template <typename GFX>
// inline int
// translate_inst(int in)
// {
//     static GFX_INST_LUT<GFX> lut;
//     return lut[in & 0x1F];
// }

// #undef LUTOVERLOAD

// template <bool HostTrap, typename GFX>
// inline rocprofiler_pc_sampling_record_t
// copySample(const void* sample)
// {
//     if(HostTrap) return copyHostTrapSample(*(const perf_sample_host_trap_v1*) sample);

//     rocprofiler_pc_sampling_record_t ret =
//         copyStochasticSample<GFX>(*(const perf_sample_snapshot_v1*) sample);

//     ret.snapshot.inst_type         = translate_inst<GFX>(ret.snapshot.inst_type);
//     ret.snapshot.arb_state_issue   = translate_arb<GFX>(ret.snapshot.arb_state_issue);
//     ret.snapshot.arb_state_stall   = translate_arb<GFX>(ret.snapshot.arb_state_stall);
//     ret.snapshot.reason_not_issued = translate_reason<GFX>(ret.snapshot.reason_not_issued);

//     return ret;
// }

#define EXTRACT_BITS(val, bit_end, bit_start)                                                      \
    (val >> bit_start) & ((1U << (bit_end - bit_start + 1)) - 1)

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

#undef EXTRACT_BITS

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
    auto        ret     = copySampleHeader<rocprofiler_pc_sampling_record_stochastic_v0_t>(sample_);
    copyChipletId<GFX9>(ret, sample_);
    copyHwId<GFX9>(ret.hw_id, sample_.hw_id);
    ret.wave_count = sample_.perf_snapshot_data1 & 0x3F;
    // TODO: implement logic for manipulating stochastic related fields
    return ret;
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
