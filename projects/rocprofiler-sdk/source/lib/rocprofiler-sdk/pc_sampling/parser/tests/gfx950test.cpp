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

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/gfxtest.hpp"

#include <gtest/gtest.h>
#include <cstddef>

/**
 * @brief This test verifies if the PC address is corrected properly on GFX950 when required.
 */
template <typename PcSamplingRecordT>
class MidMacroPCCorrectionGFX950 : public MidMacroPCCorrection<GFX950, PcSamplingRecordT>
{
public:
    void genPCSample(uint64_t pc, bool mid_macro) override
    {
        // mid_macro exists only for stochastic PC sampling on GFX950
        if constexpr(!std::is_same<PcSamplingRecordT,
                                   rocprofiler_pc_sampling_record_stochastic_v0_t>::value)
        {
            // Invalidate mid_macro
            mid_macro = false;
        }

        // invoking parent class
        MidMacroPCCorrection<GFX950, PcSamplingRecordT>::genPCSample(pc, mid_macro);
    };

    uint64_t calculateExpectedPC(uint64_t pc, bool mid_macro) override
    {
        // According to the regspec, if mid_macro is true, we need to subtract 2 dwords from the PC
        // address.
        return mid_macro ? (pc - 2 * sizeof(uint32_t)) : pc;
    }
};

TEST(pcs_parser, gfx950_test)
{
    MidMacroPCCorrectionGFX950<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    MidMacroPCCorrectionGFX950<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();

    std::cout << "GFX950 Test Done." << std::endl;
}
