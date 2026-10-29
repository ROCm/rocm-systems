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


// Pure-logic unit tests for the agent CU bitmap helpers: the un-fold of the
// legacy drm_amdgpu_info_device.cu_bitmap[4][4] (used when the kernel lacks
// AMDGPU_INFO_CU_BITMAP) and the per-WGP activity lookup used by thread trace.

#include "lib/rocprofiler-sdk/agent.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace agent = ::rocprofiler::agent;

TEST(cu_bitmap, unfold_legacy_identity_up_to_four_se)
{
    uint32_t bitmap[4][4] = {};
    for(uint32_t se = 0; se < 4; ++se)
        for(uint32_t sa = 0; sa < 2; ++sa)
            bitmap[se][sa] = 0x100u * se + sa + 1;

    const auto unfolded = agent::unfold_legacy_cu_bitmap(bitmap, 4, 2, 120001);
    ASSERT_TRUE(unfolded);
    EXPECT_EQ(unfolded->num_xcc, 1u);
    EXPECT_EQ(unfolded->num_se, 4u);
    EXPECT_EQ(unfolded->num_sa, 2u);
    EXPECT_EQ(unfolded->cu_bits_per_wgp, 2u);
    for(uint32_t se = 0; se < 4; ++se)
        for(uint32_t sa = 0; sa < 2; ++sa)
            EXPECT_EQ(unfolded->get_mask(0, se, sa), bitmap[se][sa]) << se << " " << sa;
}

// Navi31 (6 SE x 2 SA): SE4/SE5 live in columns 2-3 of rows 0/1, matching the
// kernel's gfx_v11_0_get_cu_info packing.
TEST(cu_bitmap, unfold_legacy_folds_upper_shader_engines)
{
    uint32_t bitmap[4][4] = {};
    for(auto& row : bitmap)
        for(auto& entry : row)
            entry = 0xFFu;
    bitmap[0][2] = 0x3u;
    bitmap[0][3] = 0xCu;
    bitmap[1][2] = 0x30u;
    bitmap[1][3] = 0xC0u;

    const auto unfolded = agent::unfold_legacy_cu_bitmap(bitmap, 6, 2, 110000);
    ASSERT_TRUE(unfolded);
    EXPECT_EQ(unfolded->masks.size(), 12u);
    EXPECT_EQ(unfolded->get_mask(0, 4, 0), 0x3u);
    EXPECT_EQ(unfolded->get_mask(0, 4, 1), 0xCu);
    EXPECT_EQ(unfolded->get_mask(0, 5, 0), 0x30u);
    EXPECT_EQ(unfolded->get_mask(0, 5, 1), 0xC0u);
    EXPECT_EQ(unfolded->get_mask(0, 3, 1), 0xFFu);

    // 8 SE x 1 SA folds into column 2 (stride 2), not column 1.
    bitmap[2][2] = 0x300u;
    const auto eight_se = agent::unfold_legacy_cu_bitmap(bitmap, 8, 1, 120000);
    ASSERT_TRUE(eight_se);
    EXPECT_EQ(eight_se->get_mask(0, 6, 0), 0x300u);
}

// Topologies the 4x4 fold cannot represent are rejected instead of aliasing
// another shader array's mask.
TEST(cu_bitmap, unfold_legacy_rejects_lossy_topologies)
{
    uint32_t bitmap[4][4] = {};
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 6, 4, 110000));   // SEs overlap
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 12, 1, 110000));  // past the tile
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 0, 2, 110000));
    EXPECT_TRUE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 4, 110000));
}

// Other generations use different packings (gfx9_0 folds with stride 1,
// gfx9_4_3 packs per XCC unfolded), so only GFX11/GFX12 are un-folded.
TEST(cu_bitmap, unfold_legacy_only_gfx11_gfx12)
{
    uint32_t bitmap[4][4] = {};
    bitmap[0][0]          = 0xFu;
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 1, 90402));   // gfx942
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 2, 100300));  // gfx1030
    EXPECT_FALSE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 2, 130000));
    EXPECT_TRUE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 2, 110000));
    EXPECT_TRUE(agent::unfold_legacy_cu_bitmap(bitmap, 4, 2, 120501));
}

TEST(cu_bitmap, is_wgp_active)
{
    // gfx1151 pattern: available WGPs {0,1,2,4}, 2 CU bits per WGP.
    auto rdna = agent::cu_bitmap{.num_xcc = 1, .num_se = 1, .num_sa = 2, .masks = {0x33Fu, 0}};
    EXPECT_TRUE(rdna.is_wgp_active(0, 0, 0, 0));
    EXPECT_TRUE(rdna.is_wgp_active(0, 0, 0, 2));
    EXPECT_FALSE(rdna.is_wgp_active(0, 0, 0, 3));
    EXPECT_TRUE(rdna.is_wgp_active(0, 0, 0, 4));
    EXPECT_FALSE(rdna.is_wgp_active(0, 0, 0, 5));
    EXPECT_FALSE(rdna.is_wgp_active(0, 0, 0, 16));  // past the 32-bit window
    EXPECT_FALSE(rdna.is_wgp_active(0, 0, 1, 0));   // no WGP available

    // gfx1250: 1 CU bit per WGP.
    auto gfx1250 = agent::cu_bitmap{
        .num_xcc = 2, .num_se = 1, .num_sa = 1, .cu_bits_per_wgp = 1, .masks = {0x1FFu, 0xFFu}};
    EXPECT_TRUE(gfx1250.is_wgp_active(0, 0, 0, 8));
    EXPECT_FALSE(gfx1250.is_wgp_active(1, 0, 0, 8));
    EXPECT_TRUE(gfx1250.is_wgp_active(1, 0, 0, 7));
}
