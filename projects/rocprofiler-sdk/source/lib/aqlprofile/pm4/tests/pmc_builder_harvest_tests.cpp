// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CPU-only unit tests for the GFX11 harvested-WGP path in GpuPmcBuilder
// (PR #8229). These drive the real builder with a synthetic AgentInfo (no GPU)
// to lock the emitter/reader stride invariant, the SE->SA-column fold, the
// wgp_per_sa_ growth, harvest zero-fill placement, and the UB guards.
//
// Platforms modeled: the harvest cases use the gfx1151 (Strix Halo) pattern
// active WGPs {0,1,2,4} with WGP 3 fused off (from the field investigation); the
// fold case uses Navi31 (6 SE), per the reviewer's request.
//
// NOTE: no main() here on purpose. pmc_builder_tests.cpp already provides one
// for this test executable; defining another would be a duplicate symbol.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

// Include order matters and mirrors what the gfx11 factory gets through
// pm4_factory.h:
//   1. cmd_builder.h first  -> provides ChipletId / pred_exec / CmdBuffer that
//      pmc_builder.h uses.
//   2. pmc_builder.h next    -> parsed BEFORE gfx11_def.h defines the
//      function-like `select_value` macro (from gfx11_primitives.h), which would
//      otherwise expand inside pmc_builder.h's `block_info->select_value(...)`.
//   3. gfx11_def.h last      -> gfx11_cntx_prim / SqCounterBlockInfo (+ the
//      select_value macro, which then only affects the gfx11 block table).
#include "lib/aqlprofile/pm4/cmd_builder.h"
#include "lib/aqlprofile/pm4/pmc_builder.h"

#include "lib/aqlprofile/def/gfx11_def.h"
#include "lib/aqlprofile/pm4/cmd_config.h"
#include "lib/aqlprofile/pm4/gfx11_cmd_builder.h"
#include "lib/aqlprofile/util/hsa_rsrc_factory.h"

namespace
{
// Non-concurrent GFX11 builder, matching the default gfx11_factory path.
using Gfx11PmcBuilder =
    pm4_builder::GpuPmcBuilder<pm4_builder::Gfx11CmdBuilder, gfx11_cntx_prim, false>;

// Build a synthetic GFX11 AgentInfo. gfxip must start with "gfx11" so the
// builder can acquire a (sienna_cichlid fallback) IP offset table on a box with
// no matching GPU. The concrete gfxip/name (e.g. "gfx1151" Strix Halo, "gfx1100"
// Navi31) does not change the harvest logic - it is topology + cu_bitmap driven -
// but naming the part keeps each test's intent clear.
AgentInfo
make_gfx11_agent(uint32_t    se_num,
                 uint32_t    sarrays_per_se,
                 uint32_t    cu_num,
                 uint32_t    sa_bitmap_value,
                 const char* gfxip = "gfx1100")
{
    AgentInfo info{};
    std::strncpy(info.gfxip, gfxip, sizeof(info.gfxip) - 1);
    std::strncpy(info.name, gfxip, sizeof(info.name) - 1);
    info.se_num               = se_num;
    info.xcc_num              = 1;
    info.xcc_per_aid          = 1;
    info.shader_arrays_per_se = sarrays_per_se;
    info.cu_num               = cu_num;

    // Uniformly seed the DRM cu_bitmap window covered by this topology.
    constexpr uint32_t kSeLim = std::extent_v<decltype(aqlprofile_cu_bitmap_t::bits), 0>;
    constexpr uint32_t kSaLim = std::extent_v<decltype(aqlprofile_cu_bitmap_t::bits), 1>;
    for(uint32_t se = 0; se < kSeLim; ++se)
        for(uint32_t sa = 0; sa < kSaLim; ++sa)
            info.cu_bitmap.bits[se][sa] = sa_bitmap_value;
    return info;
}

// A single SQ (per-WGP) counter descriptor: WgpAttr drives the harvest read
// loop; index 0 selects the first entry of the SQ register table.
pm4_builder::counters_vector
make_sq_counter()
{
    counter_des_t des{};
    des.id              = 0;
    des.index           = 0;
    des.block_des.id    = SqCounterBlockInfo.id;
    des.block_des.index = 0;
    des.block_info      = &SqCounterBlockInfo;

    pm4_builder::counters_vector vec;
    vec.push_back(des);
    return vec;
}

constexpr uint32_t kSentinel = 0xDEADBEEFu;

// Per-SA active-CU bitmap for a harvested gfx1151 (Strix Halo) SA - the exact
// pattern from the field investigation: active WGPs {0,1,2,4}, WGP 3 fused off.
//
// Encoding: the DRM cu_bitmap packs 2 CU bits per WGP (both CUs of that WGP), so
// WGP w occupies bits [2w, 2w+1]:
//   WGP0 = 0x003, WGP1 = 0x00C, WGP2 = 0x030, WGP3 = 0x0C0, WGP4 = 0x300
// Setting {0,1,2,4} and clearing 3:
//   0x003 | 0x00C | 0x030 | 0x300 = 0x33F   (WGP3's 0x0C0 stays clear)
// So the highest active WGP index is 4 -> physical span = 5 slots per SA.
constexpr uint32_t kHarvestMask = 0x33Fu;

// Black-box probe of the per-(SE, SA) active-WGP layout: drive a real Read()
// with an SQ (per-WGP) counter into a sentinel-filled buffer, then decode which
// physical WGP slots the reader emitted. The read loop leaves active WGP slots
// for the GPU to fill (sentinel survives) and zero-fills harvested ones, so a
// non-zero low dword marks an active slot. Returns one bitmask per (SE, SA)
// block (bit w set => WGP w active), indexed [se * sa_per_se + sa]. This lets
// tests assert the fold / synth / harvest behavior purely through observable
// output, with no access to builder internals.
std::vector<uint32_t>
read_active_wgp_masks(Gfx11PmcBuilder& builder, uint32_t se_num, uint32_t sa_per_se)
{
    const uint32_t        wgp_count = static_cast<uint32_t>(builder.GetNumWGPs());
    const uint32_t        sa_count  = se_num * sa_per_se;
    auto                  counters  = make_sq_counter();
    std::vector<uint32_t> buf(sa_count * wgp_count * 2, kSentinel);

    pm4_builder::CmdBuffer cmd_buffer;
    builder.Read(&cmd_buffer, counters, buf.data());

    std::vector<uint32_t> active(sa_count, 0u);
    for(uint32_t block = 0; block < sa_count; ++block)
        for(uint32_t wgp = 0; wgp < wgp_count; ++wgp)
            if(buf[(block * wgp_count + wgp) * 2] != 0u) active[block] |= (1u << wgp);
    return active;
}
}  // namespace

// The emitter multiplies its per-SA sample count by GetNumWGPs() (pm4_factory);
// the reader emits exactly wgp_per_sa_ slots per SA. This is the #6940 guard:
// if the two ever diverge the collected buffer aliases. Drive a real Read()
// into an allocated buffer and assert the emitted slot count matches.
TEST(PmcBuilderHarvestTest, EmitterReaderInvariant)
{
    // Harvested gfx1151 (Strix Halo) SA: 2 SE x 2 SA, active WGPs {0,1,2,4} per
    // SA. cu_num = 32 active CUs (16 active WGPs x 2 CUs) -> initial wgp_per_sa_
    // = 4; the harvest span then grows to 5 to reach live WGP 4.
    const uint32_t se_num = 2, sa_per_se = 2, cu_num = 32;
    AgentInfo      info = make_gfx11_agent(se_num, sa_per_se, cu_num, kHarvestMask, "gfx1151");

    Gfx11PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 5);

    auto counters = make_sq_counter();

    const uint32_t        sa_count   = se_num * sa_per_se;
    const uint32_t        slot_count = sa_count * builder.GetNumWGPs();
    std::vector<uint32_t> buf(slot_count * 2, kSentinel);

    pm4_builder::CmdBuffer cmd_buffer;
    const uint32_t         bytes = builder.Read(&cmd_buffer, counters, buf.data());

    // 2 dwords per WGP slot.
    const uint32_t emitted_slots = bytes / sizeof(uint32_t) / 2;
    EXPECT_EQ(emitted_slots, slot_count);
    // Reader stride per SA must equal the emitter multiplier GetNumWGPs().
    EXPECT_EQ(emitted_slots / sa_count, static_cast<uint32_t>(builder.GetNumWGPs()));
}

// GFX11 grows wgp_per_sa_ to the physical span so the [0, wgp_per_sa_) read
// bound reaches a middle-harvested WGP.
TEST(PmcBuilderHarvestTest, WgpGrowthGfx11)
{
    // gfx1151 (Strix Halo) harvest: active {0,1,2,4} -> span 5 even though cu_num
    // implies only 4.
    AgentInfo       harvested = make_gfx11_agent(2, 2, 32, kHarvestMask, "gfx1151");
    Gfx11PmcBuilder b1(&harvested);
    EXPECT_EQ(b1.GetNumWGPs(), 5);

    // Contiguous active {0,1,2,3} -> span 4 (no growth beyond cu_num).
    AgentInfo       contig = make_gfx11_agent(2, 2, 32, 0xFFu, "gfx1151");
    Gfx11PmcBuilder b2(&contig);
    EXPECT_EQ(b2.GetNumWGPs(), 4);
}

// Harvested WGP slots are zero-filled in place; active WGPs land at their
// physical slot with no compaction/remap.
TEST(PmcBuilderHarvestTest, ZeroFillPlacement)
{
    // gfx1151 (Strix Halo) harvest pattern: active {0,1,2,4}, WGP 3 fused off.
    const uint32_t se_num = 2, sa_per_se = 2, cu_num = 32;
    AgentInfo      info = make_gfx11_agent(se_num, sa_per_se, cu_num, kHarvestMask, "gfx1151");

    Gfx11PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 5);

    auto                  counters  = make_sq_counter();
    const uint32_t        wgp_count = builder.GetNumWGPs();
    const uint32_t        sa_count  = se_num * sa_per_se;
    std::vector<uint32_t> buf(sa_count * wgp_count * 2, kSentinel);

    pm4_builder::CmdBuffer cmd_buffer;
    builder.Read(&cmd_buffer, counters, buf.data());

    // For each SA block, the harvested WGP (index 3) low dword is zeroed while
    // active WGPs keep their sentinel (the GPU would fill them at runtime).
    for(uint32_t sa = 0; sa < sa_count; ++sa)
    {
        const uint32_t base = sa * wgp_count * 2;
        for(uint32_t wgp = 0; wgp < wgp_count; ++wgp)
        {
            const uint32_t lo = buf[base + wgp * 2];
            if(wgp == 3)
                EXPECT_EQ(lo, 0u) << "harvested WGP " << wgp << " (SA " << sa
                                  << ") not zero-filled";
            else
                EXPECT_EQ(lo, kSentinel)
                    << "active WGP " << wgp << " (SA " << sa << ") was overwritten";
        }
    }
}

// Logical SEs beyond the bitmap's 4-SE dimension fold into higher SA columns,
// mirroring the kernel gfx_v11_0_get_cu_info packing (validated on Navi31, 6 SE).
TEST(PmcBuilderHarvestTest, FoldingNavi31)
{
    const uint32_t se_num = 6, sa_per_se = 2;
    AgentInfo      info = make_gfx11_agent(se_num, sa_per_se, /*cu_num=*/120, /*value=*/0x33Fu);

    // Distinct single-WGP patterns for the folded SE4/SE5 raw slots ([0..1][2..3]).
    info.cu_bitmap.bits[0][2] = 0x3u;   // WGP0 -> logical SE4, SA0
    info.cu_bitmap.bits[0][3] = 0xCu;   // WGP1 -> logical SE4, SA1
    info.cu_bitmap.bits[1][2] = 0x30u;  // WGP2 -> logical SE5, SA0
    info.cu_bitmap.bits[1][3] = 0xC0u;  // WGP3 -> logical SE5, SA1
    // Raw rows 2-3 cols 2-3 are only reachable by SE6/SE7 (absent here), so they
    // must never be read. Seed them with a distinct WGP4 pattern: if the fold
    // wrongly sourced SE4/SE5 from these rows, the probe below would see WGP4
    // active instead of the expected single WGP.
    info.cu_bitmap.bits[2][2] = 0x300u;
    info.cu_bitmap.bits[2][3] = 0x300u;
    info.cu_bitmap.bits[3][2] = 0x300u;
    info.cu_bitmap.bits[3][3] = 0x300u;

    Gfx11PmcBuilder builder(&info);
    ASSERT_EQ(builder.GetNumWGPs(), 5);  // 0x33F on base SEs spans WGP0..4.

    const auto     active = read_active_wgp_masks(builder, se_num, sa_per_se);
    const auto     at     = [&](uint32_t se, uint32_t sa) { return active[se * sa_per_se + sa]; };
    constexpr auto kBaseMask = 0x17u;  // 0x33F -> active WGPs {0,1,2,4}.

    // Base SEs (0..3) read raw rows 0..3 cols 0..1 (still 0x33F).
    for(uint32_t se = 0; se < 4; ++se)
        for(uint32_t sa = 0; sa < sa_per_se; ++sa)
            EXPECT_EQ(at(se, sa), kBaseMask) << "SE " << se << " SA " << sa;

    // Folded SE4/SE5 pick up the higher SA columns of raw rows 0/1 (single WGP
    // each), proving the fold sources rows 0/1 - not the WGP4 sentinel rows.
    EXPECT_EQ(at(4, 0), 0x1u);  // WGP0
    EXPECT_EQ(at(4, 1), 0x2u);  // WGP1
    EXPECT_EQ(at(5, 0), 0x4u);  // WGP2
    EXPECT_EQ(at(5, 1), 0x8u);  // WGP3
}

// UB / boundary guards in build_sa_cu_mask(): no clz(0), no (1u << 32), and no
// crash on degenerate topologies. Observed through the emitted read layout.
TEST(PmcBuilderHarvestTest, UBGuards)
{
    const uint32_t se_num = 2, sa_per_se = 2, sa_count = se_num * sa_per_se;

    // (a) All-zero bitmap with cu_num > 0 -> synthesize a fully-active mask; no
    //     WGP is zero-filled.
    {
        AgentInfo       info = make_gfx11_agent(se_num, sa_per_se, /*cu_num=*/32, /*value=*/0u);
        Gfx11PmcBuilder builder(&info);
        ASSERT_EQ(builder.GetNumWGPs(), 4);
        const auto     active   = read_active_wgp_masks(builder, se_num, sa_per_se);
        const uint32_t all_wgps = (1u << builder.GetNumWGPs()) - 1u;  // 0xF
        for(uint32_t block = 0; block < sa_count; ++block)
            EXPECT_EQ(active[block], all_wgps) << "block " << block;
    }

    // (b) wgp_per_sa_ spanning the full 32-bit window (>= 16 WGPs) -> synth mask
    //     ~0u (never 1u << 32); every WGP slot stays active.
    {
        AgentInfo       info = make_gfx11_agent(se_num, sa_per_se, /*cu_num=*/128, /*value=*/0u);
        Gfx11PmcBuilder builder(&info);
        ASSERT_GE(builder.GetNumWGPs(), 16);
        const auto     active = read_active_wgp_masks(builder, se_num, sa_per_se);
        const uint32_t all_wgps =
            (builder.GetNumWGPs() >= 32) ? ~0u : ((1u << builder.GetNumWGPs()) - 1u);
        for(uint32_t block = 0; block < sa_count; ++block)
            EXPECT_EQ(active[block], all_wgps) << "block " << block;
    }

    // (c) cu_num == 0 -> no clz(0), no crash; empty mask means every emitted slot
    //     is zero-filled (no active WGP).
    {
        AgentInfo       info = make_gfx11_agent(se_num, sa_per_se, /*cu_num=*/0, /*value=*/0u);
        Gfx11PmcBuilder builder(&info);
        const auto      active = read_active_wgp_masks(builder, se_num, sa_per_se);
        for(uint32_t block = 0; block < sa_count; ++block)
            EXPECT_EQ(active[block], 0u) << "block " << block;
    }
}
