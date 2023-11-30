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

#include <cstdint>
#include <cstring>

#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx11.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx_unknown.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/gfx9.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/parser_types.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.hpp"

pcsample_v1_t
copyHostTrapSample(const perf_sample_host_trap_v1& sample);

class PCSParserTranslation
{
public:
    template <typename SType>
    static pcsample_v1_t copySampleHeader(const SType& sample);

    template <typename gfx>
    static pcsample_v1_t copyStochasticSample(const perf_sample_snapshot_v1& sample);
};

#define BITSHIFT(sname) out |= ((in >> GFX::sname) & 1) << PCSAMPLE::sname

template <typename GFX>
int
translate_arb(int in)
{
    size_t out = 0;
    BITSHIFT(ISSUE_VALU);
    BITSHIFT(ISSUE_MATRIX);
    BITSHIFT(ISSUE_LDS);
    BITSHIFT(ISSUE_LDS_DIRECT);
    BITSHIFT(ISSUE_SCALAR);
    BITSHIFT(ISSUE_VMEM_TEX);
    BITSHIFT(ISSUE_FLAT);
    BITSHIFT(ISSUE_EXP);
    BITSHIFT(ISSUE_MISC);
    BITSHIFT(ISSUE_BRMSG);
    return out & 0x3FF;
}

#undef BITSHIFT

#define LUTOVERLOAD(sname) this->operator[](GFX::sname) = PCSAMPLE::sname

template <typename GFX>
class GFX_REASON_LUT : public std::array<int, 32>
{
public:
    GFX_REASON_LUT()
    {
        std::memset(data(), 0, size() * sizeof(int));
        LUTOVERLOAD(REASON_NOT_AVAILABLE);
        LUTOVERLOAD(REASON_ALU);
        LUTOVERLOAD(REASON_WAITCNT);
        LUTOVERLOAD(REASON_INTERNAL);
        LUTOVERLOAD(REASON_BARRIER);
        LUTOVERLOAD(REASON_ARBITER);
        LUTOVERLOAD(REASON_EX_STALL);
        LUTOVERLOAD(REASON_OTHER_WAIT);
        LUTOVERLOAD(REASON_SLEEP);
    }
};

template <typename GFX>
class GFX_INST_LUT : public std::array<int, 32>
{
public:
    GFX_INST_LUT()
    {
        std::memset(data(), 0, size() * sizeof(int));
        LUTOVERLOAD(TYPE_VALU);
        LUTOVERLOAD(TYPE_MATRIX);
        LUTOVERLOAD(TYPE_SCALAR);
        LUTOVERLOAD(TYPE_TEX);
        LUTOVERLOAD(TYPE_LDS);
        LUTOVERLOAD(TYPE_LDS_DIRECT);
        LUTOVERLOAD(TYPE_FLAT);
        LUTOVERLOAD(TYPE_EXP);
        LUTOVERLOAD(TYPE_MESSAGE);
        LUTOVERLOAD(TYPE_BARRIER);
        LUTOVERLOAD(TYPE_BRANCH_NOT_TAKEN);
        LUTOVERLOAD(TYPE_BRANCH_TAKEN);
        LUTOVERLOAD(TYPE_JUMP);
        LUTOVERLOAD(TYPE_OTHER);
        LUTOVERLOAD(TYPE_NO_INST);
        LUTOVERLOAD(TYPE_DUAL_VALU);
    }
};

template <typename GFX>
int
translate_reason(int in)
{
    static GFX_REASON_LUT<GFX> lut;
    return lut[in & 0xF];
}

template <typename GFX>
int
translate_inst(int in)
{
    static GFX_INST_LUT<GFX> lut;
    return lut[in & 0xF];
}

#undef LUTOVERLOAD

template <bool HostTrap, typename GFX>
inline pcsample_v1_t
copySample(const void* sample)
{
    if(HostTrap) return copyHostTrapSample(*(const perf_sample_host_trap_v1*) sample);

    pcsample_v1_t ret =
        PCSParserTranslation::copyStochasticSample<GFX>(*(const perf_sample_snapshot_v1*) sample);

    ret.snapshot.inst_type         = translate_inst<GFX>(ret.snapshot.inst_type);
    ret.snapshot.arb_state_issue   = translate_arb<GFX>(ret.snapshot.arb_state_issue);
    ret.snapshot.arb_state_stall   = translate_arb<GFX>(ret.snapshot.arb_state_stall);
    ret.snapshot.reason_not_issued = translate_reason<GFX>(ret.snapshot.reason_not_issued);

    return ret;
}
