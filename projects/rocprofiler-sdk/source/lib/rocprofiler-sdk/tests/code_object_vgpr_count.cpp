// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Pure-logic unit test for the VGPR/AGPR counts reported on kernel symbols. On
// gfx90a/gfx94x/gfx95x the unified register file is split by ACCUM_OFFSET, which is
// COMPUTE_PGM_RSRC3[5:0] (6 bits, granularity 4): arch VGPRs = (ACCUM_OFFSET + 1) * 4
// and AGPRs = (GRANULATED_WORKITEM_VGPR_COUNT + 1) * 8 - arch VGPRs.

#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

namespace code_object = ::rocprofiler::code_object;

namespace
{
code_object::kernel_descriptor_t
make_descriptor(uint32_t accum_offset, uint32_t vgpr_granule)
{
    auto kd              = code_object::kernel_descriptor_t{};
    kd.compute_pgm_rsrc3 = accum_offset;  // ACCUM_OFFSET: bits 5:0
    kd.compute_pgm_rsrc1 = vgpr_granule;  // GRANULATED_WORKITEM_VGPR_COUNT: bits 5:0
    return kd;
}
}  // namespace

TEST(code_object_vgpr_count, unified_register_file_above_128_arch_vgprs)
{
    for(std::string_view name : {"gfx90a", "gfx942", "gfx950"})
    {
        SCOPED_TRACE(name);

        // 256 arch VGPRs, no AGPRs: ACCUM_OFFSET=63 needs bit 5
        auto all_arch = make_descriptor(63, 31);
        EXPECT_EQ(code_object::arch_vgpr_count(name, all_arch), 256u);
        EXPECT_EQ(code_object::accum_vgpr_count(name, all_arch), 0u);

        // 132 arch VGPRs + 124 AGPRs: ACCUM_OFFSET=32 is only bit 5, the smallest value it affects
        auto boundary = make_descriptor(32, 31);
        EXPECT_EQ(code_object::arch_vgpr_count(name, boundary), 132u);
        EXPECT_EQ(code_object::accum_vgpr_count(name, boundary), 124u);

        // 160 arch VGPRs out of 384 total: ACCUM_OFFSET=39 needs bit 5
        auto split = make_descriptor(39, 47);
        EXPECT_EQ(code_object::arch_vgpr_count(name, split), 160u);
        EXPECT_EQ(code_object::accum_vgpr_count(name, split), 224u);
    }
}

TEST(code_object_vgpr_count, unified_register_file_up_to_128_arch_vgprs)
{
    for(std::string_view name : {"gfx90a", "gfx942", "gfx950"})
    {
        SCOPED_TRACE(name);

        // 128 arch VGPRs + 128 AGPRs
        auto half = make_descriptor(31, 31);
        EXPECT_EQ(code_object::arch_vgpr_count(name, half), 128u);
        EXPECT_EQ(code_object::accum_vgpr_count(name, half), 128u);

        // 4 arch VGPRs (minimum) out of 8 total
        auto minimum = make_descriptor(0, 0);
        EXPECT_EQ(code_object::arch_vgpr_count(name, minimum), 4u);
        EXPECT_EQ(code_object::accum_vgpr_count(name, minimum), 4u);
    }
}
