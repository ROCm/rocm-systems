// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GFX12 companion to pmc_builder_harvest_tests.cpp (PR #8229). Kept in its own
// translation unit because the gfx11 and gfx12 register headers cannot coexist
// in a single TU. Confirms the harvest wgp_per_sa_ growth is GFX11-only: on
// GFX12 the WGP count stays cu_num-derived regardless of the DRM cu_bitmap.
//
// NOTE: no main() here on purpose (see pmc_builder_tests.cpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

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
make_gfx12_agent(uint32_t se_num,
                 uint32_t sarrays_per_se,
                 uint32_t cu_num,
                 uint32_t sa_bitmap_value)
{
    AgentInfo info{};
    std::strncpy(info.gfxip, "gfx1200", sizeof(info.gfxip) - 1);
    std::strncpy(info.name, "gfx1200", sizeof(info.name) - 1);
    info.se_num               = se_num;
    info.xcc_num              = 1;
    info.xcc_per_aid          = 1;
    info.shader_arrays_per_se = sarrays_per_se;
    info.cu_num               = cu_num;

    constexpr uint32_t kSeLim = std::extent_v<decltype(aqlprofile_cu_bitmap_t::bits), 0>;
    constexpr uint32_t kSaLim = std::extent_v<decltype(aqlprofile_cu_bitmap_t::bits), 1>;
    for(uint32_t se = 0; se < kSeLim; ++se)
        for(uint32_t sa = 0; sa < kSaLim; ++sa)
            info.cu_bitmap.bits[se][sa] = sa_bitmap_value;
    return info;
}
}  // namespace

// Same harvest pattern that grows GFX11 to span 5 must leave GFX12 at its
// cu_num-derived count (growth is gated on GFXIP_LEVEL == 11).
TEST(PmcBuilderHarvestGfx12Test, WgpGrowthDisabled)
{
    // cu_num = 32 -> wgp_per_sa_ = (16) / (2 * 2) = 4.
    AgentInfo       harvested = make_gfx12_agent(2, 2, 32, /*value=*/0x33Fu);
    Gfx12PmcBuilder b1(&harvested);
    EXPECT_EQ(b1.GetNumWGPs(), 4);

    // Different bitmap, same cu_num -> still cu_num-derived, unchanged.
    AgentInfo       contig = make_gfx12_agent(2, 2, 32, /*value=*/0xFFu);
    Gfx12PmcBuilder b2(&contig);
    EXPECT_EQ(b2.GetNumWGPs(), 4);
}
