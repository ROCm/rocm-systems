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

#include "lib/output/pc_sampling_pc_correction.hpp"

#include <gtest/gtest.h>

#include <string_view>

using rocprofiler::tool::pc_correction::classify;
using rocprofiler::tool::pc_correction::Kind;

TEST(pc_correction_classify, Classify_NopIsRegular)
{
    EXPECT_EQ(classify("s_nop 0"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_SleepIsRegular)
{
    EXPECT_EQ(classify("s_sleep 1"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_WaitAllVariants)
{
    // All s_wait* variants resolve via the "s_wait" prefix.
    static constexpr std::string_view wait_variants[] = {
        "s_waitcnt 0",
        "s_wait_loadcnt 0",
        "s_wait_storecnt 0",
        "s_wait_kmcnt 0",
        "s_wait_alu 0",
        "s_wait_idle",
    };

    for(auto inst : wait_variants)
    {
        EXPECT_EQ(classify(inst), Kind::REGULAR_INTERNAL) << "inst=" << inst;
    }
}

TEST(pc_correction_classify, Classify_BarrierWaitIsRegular)
{
    EXPECT_EQ(classify("s_barrier_wait -1"), Kind::REGULAR_INTERNAL);
}

TEST(pc_correction_classify, Classify_IcacheInvIsS_icache_inv)
{
    EXPECT_EQ(classify("s_icache_inv"), Kind::S_ICACHE_INV);
}

TEST(pc_correction_classify, Classify_VAluIsExt)
{
    EXPECT_EQ(classify("v_add_f32_e32 v0, v1, v2"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_SAluIsExt)
{
    EXPECT_EQ(classify("s_mov_b32 s0, 1"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_BranchIsExt)
{
    EXPECT_EQ(classify("s_branch 8"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_EndPgmIsExt)
{
    EXPECT_EQ(classify("s_endpgm"), Kind::EXT);
}

TEST(pc_correction_classify, Classify_SetPrioIsExt_Today)
{
    // s_setprio is a candidate internal pending hardware confirmation; until
    // then it must classify as EXT (the safe direction).
    EXPECT_EQ(classify("s_setprio 1"), Kind::EXT);
}
