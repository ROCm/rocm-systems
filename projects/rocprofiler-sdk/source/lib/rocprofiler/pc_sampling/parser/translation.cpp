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

#include "lib/rocprofiler/pc_sampling/parser/translation.hpp"

pcsample_v1_t
copyHostTrapSample(const perf_sample_host_trap_v1& sample)
{
    pcsample_v1_t ret = PCSParserTranslation::copySampleHeader<perf_sample_host_trap_v1>(sample);
    ret.flags.type    = AMD_HOST_TRAP_V1;
    return ret;
}

template <typename SType>
pcsample_v1_t
PCSParserTranslation::copySampleHeader(const SType& sample)
{
    pcsample_v1_t ret;
    ret.flags.type = AMD_SNAPSHOT_V1;

    ret.pc             = sample.pc;
    ret.exec_mask      = sample.exec_mask;
    ret.workgroud_id_x = sample.workgroud_id_x;
    ret.workgroud_id_y = sample.workgroud_id_y;
    ret.workgroud_id_z = sample.workgroud_id_z;

    ret.chiplet   = sample.chiplet_and_wave_id >> 8;
    ret.wave_id   = sample.chiplet_and_wave_id & 0x3F;
    ret.hw_id     = sample.hw_id;
    ret.timestamp = sample.timestamp;
    return ret;
}

template <typename gfx>
pcsample_v1_t
PCSParserTranslation::copyStochasticSample(const perf_sample_snapshot_v1& sample)
{
    (void) sample;
    return {};
};

template <>
pcsample_v1_t
PCSParserTranslation::copyStochasticSample<GFX9>(const perf_sample_snapshot_v1& sample)
{
    pcsample_v1_t ret = copySampleHeader<perf_sample_snapshot_v1>(sample);
    ret.flags.valid   = sample.perf_snapshot_data & (~sample.perf_snapshot_data >> 26) & 0x1;
    // Check wave_id matches snapshot_wave_id

    ret.flags.has_wave_cnt     = true;
    ret.flags.has_stall_reason = true;

    ret.wave_count = sample.perf_snapshot_data1 & 0x3F;

    ret.wave_issued                = sample.perf_snapshot_data >> 1;
    ret.snapshot.dual_issue_valu   = sample.perf_snapshot_data >> 2;
    ret.snapshot.inst_type         = sample.perf_snapshot_data >> 3;
    ret.snapshot.reason_not_issued = (sample.perf_snapshot_data >> 7) & 0x7;
    ret.snapshot.arb_state_issue   = (sample.perf_snapshot_data >> 10) & 0xFF;
    ret.snapshot.arb_state_stall   = (sample.perf_snapshot_data >> 18) & 0xFF;
    return ret;
}

template <>
pcsample_v1_t
PCSParserTranslation::copyStochasticSample<GFX11>(const perf_sample_snapshot_v1& sample)
{
    // TODO: finish this
    return copySampleHeader<perf_sample_snapshot_v1>(sample);
}

template <>
pcsample_v1_t
PCSParserTranslation::copyStochasticSample<gfx_unknown>(const perf_sample_snapshot_v1& sample)
{
    pcsample_v1_t ret = copySampleHeader<perf_sample_snapshot_v1>(sample);
    ret.flags.valid   = sample.perf_snapshot_data & 0x1;
    // Check wave_id matches snapshot_wave_id

    ret.flags.has_wave_cnt     = true;
    ret.flags.has_stall_reason = true;

    ret.wave_issued                = sample.perf_snapshot_data >> 1;
    ret.snapshot.inst_type         = sample.perf_snapshot_data >> 2;
    ret.snapshot.reason_not_issued = (sample.perf_snapshot_data >> 6) & 0x7;

    ret.wave_count               = sample.perf_snapshot_data1 & 0x3F;
    ret.snapshot.arb_state_issue = (sample.perf_snapshot_data1 >> 6) & 0xFF;
    ret.snapshot.arb_state_stall = (sample.perf_snapshot_data1 >> 14) & 0xFF;

    ret.flags.has_memory_counter = true;
    ret.memory_counters.raw      = sample.perf_snapshot_data2;
    return ret;
}
