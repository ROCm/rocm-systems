// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GFX12 companion to pmc_builder_harvest_tests.cpp (PR #8229). Kept in its own
// translation unit because the gfx11 and gfx12 register headers cannot coexist
// in a single TU. Confirms the WGP read path that skips WGPs that are not
// available and grows wgp_per_sa_ to the physical span) also applies to GFX12, including
// gfx1250's 1-bit-per-WGP bitmap encoding and per-XCC bitmaps.
//
// NOTE: no main() here on purpose (see pmc_builder_tests.cpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

// Include order matters and mirrors what the gfx12 factory gets through
// pm4_factory.h:
//   1. cmd_builder.h first  -> provides ChipletId / pred_exec / CmdBuffer that
//      pmc_builder.h uses.
//   2. pmc_builder.h next    -> parsed BEFORE gfx12_def.h defines the
//      function-like `select_value` macro (from gfx12_primitives.h), which would
//      otherwise expand inside pmc_builder.h's `block_info->select_value(...)`.
//   3. gfx12_def.h last      -> gfx12_cntx_prim (+ the select_value macro, which
//      then only affects the gfx12 block table).
#include "lib/aqlprofile/pm4/cmd_builder.h"
#include "lib/aqlprofile/pm4/pmc_builder.h"

#include "lib/aqlprofile/def/gfx12_def.h"
#include "lib/aqlprofile/pm4/cmd_config.h"
#include "lib/aqlprofile/pm4/gfx12_cmd_builder.h"
#include "lib/aqlprofile/util/hsa_rsrc_factory.h"

namespace
{
using Gfx12PmcBuilder =
    pm4_builder::GpuPmcBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim, false>;

AgentInfo
make_gfx12_agent(uint32_t    xcc_num,
                 uint32_t    se_per_xcc,
                 uint32_t    sarrays_per_se,
                 uint32_t    cu_num,
                 uint32_t    sa_bitmap_value,
                 const char* gfxip = "gfx1200")
{
    AgentInfo info{};
    std::strncpy(info.gfxip, gfxip, sizeof(info.gfxip) - 1);
    std::strncpy(info.name, gfxip, sizeof(info.name) - 1);
    info.se_num               = xcc_num * se_per_xcc;
    info.xcc_num              = xcc_num;
    info.xcc_per_aid          = 1;
    info.shader_arrays_per_se = sarrays_per_se;
    info.cu_num               = cu_num;
    info.cu_bitmap.assign(xcc_num * se_per_xcc * sarrays_per_se, sa_bitmap_value);
    return info;
}

// A single SQ (per-WGP) counter descriptor.
pm4_builder::counters_vector
make_sq_counter()
{
    counter_des_t des{};
    des.id              = 0;
    des.index           = 0;
    des.block_des.id    = SqcCounterBlockInfo.id;
    des.block_des.index = 0;
    des.block_info      = &SqcCounterBlockInfo;

    pm4_builder::counters_vector vec;
    vec.push_back(des);
    return vec;
}

constexpr uint32_t kSentinel = 0xDEADBEEFu;

// Drive a real Read() with an SQ counter into a sentinel-filled buffer and
// decode which WGP slots were left for the GPU (active) vs zero-filled
// (not available). Returns one bitmask per (XCC, SE, SA) block, in read order.
std::vector<uint32_t>
read_active_wgp_masks(Gfx12PmcBuilder& builder, uint32_t block_count)
{
    const uint32_t        wgp_count = static_cast<uint32_t>(builder.GetNumWGPs());
    auto                  counters  = make_sq_counter();
    std::vector<uint32_t> buf(block_count * wgp_count * 2, kSentinel);

    pm4_builder::CmdBuffer cmd_buffer;
    const uint32_t         bytes = builder.Read(&cmd_buffer, counters, buf.data());
    EXPECT_EQ(bytes, buf.size() * sizeof(uint32_t));

    std::vector<uint32_t> active(block_count, 0u);
    for(uint32_t block = 0; block < block_count; ++block)
        for(uint32_t wgp = 0; wgp < wgp_count; ++wgp)
            if(buf[(block * wgp_count + wgp) * 2] != 0u) active[block] |= (1u << wgp);
    return active;
}
}  // namespace

// GFX12 grows wgp_per_sa_ to the physical span so the [0, wgp_per_sa_) read
// bound reaches a WGP after one that is not available, exactly like GFX11.
TEST(PmcBuilderHarvestGfx12Test, WgpGrowth)
{
    // cu_num = 32 -> wgp_per_sa_ = (16) / (2 * 2) = 4; active {0,1,2,4} -> 5.
    AgentInfo       partial = make_gfx12_agent(1, 2, 2, 32, /*value=*/0x33Fu);
    Gfx12PmcBuilder b1(&partial);
    EXPECT_EQ(b1.GetNumWGPs(), 5);

    // Contiguous active {0,1,2,3} -> span 4 (no growth beyond cu_num).
    AgentInfo       contig = make_gfx12_agent(1, 2, 2, 32, /*value=*/0xFFu);
    Gfx12PmcBuilder b2(&contig);
    EXPECT_EQ(b2.GetNumWGPs(), 4);
}

// WGPs that are not available are zero-filled in place instead of being read.
TEST(PmcBuilderHarvestGfx12Test, ZeroFillPlacement)
{
    AgentInfo       info = make_gfx12_agent(1, 2, 2, 32, /*value=*/0x33Fu);
    Gfx12PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 5);

    for(auto mask : read_active_wgp_masks(builder, 4))
        EXPECT_EQ(mask, 0x17u);  // WGPs {0,1,2,4}
}

// Per-XCC bitmaps: each XCC's read uses its own slice of the bitmap.
TEST(PmcBuilderHarvestGfx12Test, PerXccBitmap)
{
    AgentInfo info = make_gfx12_agent(2, 2, 2, /*cu_num=*/64, /*value=*/0xFFu);
    // XCC1, SE1, SA0 loses WGP 1.
    info.cu_bitmap[(1 * 2 + 1) * 2 + 0] = 0xF3u;

    Gfx12PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 4);

    const auto active = read_active_wgp_masks(builder, 8);
    for(uint32_t block = 0; block < 8; ++block)
        EXPECT_EQ(active[block], block == 6 ? 0xDu : 0xFu) << "block " << block;
}

// gfx1250 encodes 1 CU bit per WGP (1 WGP = 1 CU). An asymmetric SA1 with only
// 8 WGPs is described by the bitmap: WGP 8 of SA1 is never read.
TEST(PmcBuilderHarvestGfx12Test, Gfx1250OneBitPerWgp)
{
    // populate_arch_specific_info() sizes gfx1250 for 9 WGPs per SA.
    const uint32_t se = 2, sa = 2;
    AgentInfo      info = make_gfx12_agent(1, se, sa, se * sa * 9 * 2, 0x1FFu, "gfx1250");
    for(uint32_t s = 0; s < se; ++s)
        info.cu_bitmap[s * sa + 1] = 0xFFu;  // SA1: WGPs 0..7
    info.cu_bitmap[0] = 0x1EFu;              // SE0 SA0: WGP 4 not available

    Gfx12PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 9);

    const auto active = read_active_wgp_masks(builder, se * sa);
    EXPECT_EQ(active[0], 0x1EFu);
    EXPECT_EQ(active[1], 0xFFu);
    EXPECT_EQ(active[2], 0x1FFu);
    EXPECT_EQ(active[3], 0xFFu);
}

// Without a bitmap, gfx1250 still skips the missing WGP 8 of SA1.
TEST(PmcBuilderHarvestGfx12Test, Gfx1250NoBitmapAsymmetricFallback)
{
    const uint32_t se = 2, sa = 2;
    AgentInfo      info = make_gfx12_agent(1, se, sa, se * sa * 9 * 2, 0u, "gfx1250");
    info.cu_bitmap.clear();

    Gfx12PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 9);

    const auto active = read_active_wgp_masks(builder, se * sa);
    for(uint32_t block = 0; block < se * sa; ++block)
        EXPECT_EQ(active[block], (block % sa == 1) ? 0xFFu : 0x1FFu) << "block " << block;
}
