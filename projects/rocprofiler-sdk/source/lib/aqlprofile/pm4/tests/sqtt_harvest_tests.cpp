// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "lib/aqlprofile/pm4/cmd_builder.h"
#include "lib/aqlprofile/pm4/sqtt_builder.h"
#include "lib/aqlprofile/def/gfx11_def.h"
#include "lib/aqlprofile/pm4/gfx11_cmd_builder.h"

namespace
{
using Builder = pm4_builder::GpuSqttBuilder<pm4_builder::Gfx11CmdBuilder, gfx11_cntx_prim>;

AgentInfo
w7800()
{
    AgentInfo info{};
    std::strncpy(info.gfxip, "gfx1100", sizeof(info.gfxip) - 1);
    std::strncpy(info.name, "gfx1100", sizeof(info.name) - 1);
    info.se_num  = 6;
    info.xcc_num = info.xcc_per_aid = 1;
    info.shader_arrays_per_se       = 2;
    info.cu_num                     = 70;
    // Physical DRM map from a 70-CU Navi31: SE0 is entirely harvested.
    const uint32_t bitmap[4][4] = {
        {0, 0, 0xfc, 0xff}, {0xfc, 0xff, 0xff, 0xcf}, {0xfc, 0xff, 0, 0}, {0xfc, 0xff, 0, 0}};
    std::memcpy(info.cu_bitmap.bits, bitmap, sizeof(bitmap));
    return info;
}

struct Capture
{
    std::vector<pm4_builder::TraceControl> controls{6};
    pm4_builder::TraceConfig               config{};
    pm4_builder::CmdBuffer                 commands{};
    Capture()
    {
        config.se_mask             = 0x3f;
        config.data_buffer_ptr     = reinterpret_cast<void*>(0x10000000ull);
        config.data_buffer_size    = 16 << 20;
        config.control_buffer_ptr  = controls.data();
        config.control_buffer_size = controls.size() * sizeof(controls[0]);
        config.simd_sel            = 0;
    }
};

// Decode actual SET_UCONFIG_REG packets directed to the SQTT mask register.
// No PM4 packet is submitted to a GPU in these tests.
std::vector<uint32_t>
programmed_values(const AgentInfo&        info,
                  pm4_builder::CmdBuffer& buffer,
                  const Register&         reg = gfx11_cntx_prim::SQ_THREAD_TRACE_MASK_ADDR)
{
    pm4_builder::Gfx11CmdBuilder commands(acquire_ip_offset_table(&info));
    uint32_t                     mask_addr = commands.get_addr(reg);
    const auto*                  data      = static_cast<const uint32_t*>(buffer.Data());
    std::vector<uint32_t>        values;
    for(size_t i = 0; i < buffer.DwSize();)
    {
        size_t length = ((data[i] >> 16) & 0x3fff) + 2;
        if(length < 2 || i + length > buffer.DwSize()) break;
        if(((data[i] >> 8) & 0xff) == PACKET3_SET_UCONFIG_REG && length == 3 &&
           (data[i + 1] & 0xffff) == mask_addr - UCONFIG_SPACE_START)
            values.push_back(data[i + 2]);
        i += length;
    }
    return values;
}
}  // namespace

TEST(SqttHarvestTest, InactiveEngineIsExcludedBeforeBufferAllocation)
{
    auto    info = w7800();
    Builder builder(&info);
    Capture capture;
    builder.Begin(&capture.commands, &capture.config);
    EXPECT_EQ(capture.config.GetSEmask(), 0x3eu);
    EXPECT_EQ(capture.config.GetTargetCU(0), -1);
    EXPECT_EQ(capture.config.GetCapacity(0), 4096u);
    EXPECT_EQ(capture.config.GetCapacity(1), 3305472u);
    auto masks = programmed_values(info, capture.commands);
    ASSERT_EQ(masks.size(), 5u);
    // WGP_SEL is bits 7:4; preserve SE5 WGP0, use WGP1 for SE1-4.
    for(int se = 1; se < 5; ++se)
    {
        EXPECT_EQ(capture.config.GetTargetCU(se), 1);
        EXPECT_EQ((masks[se - 1] >> 4) & 0xf, 1u);
    }
    EXPECT_EQ(capture.config.GetTargetCU(5), 0);
    EXPECT_EQ((masks[4] >> 4) & 0xf, 0u);
}

TEST(SqttHarvestTest, InactiveEngineOnlyRejectsBeforeEmittingPackets)
{
    auto    info = w7800();
    Builder builder(&info);
    Capture capture;
    capture.config.se_mask = 1;
    EXPECT_THROW(builder.Begin(&capture.commands, &capture.config), std::runtime_error);
    EXPECT_EQ(capture.commands.DwSize(), 0u);
}

TEST(SqttHarvestTest, ExplicitActiveTargetIsPreserved)
{
    auto    info = w7800();
    Builder builder(&info);
    Capture capture;
    capture.config.targetCu = 2;
    builder.Begin(&capture.commands, &capture.config);
    auto masks = programmed_values(info, capture.commands);
    ASSERT_EQ(masks.size(), 5u);
    for(int se = 1; se < 6; ++se)
    {
        EXPECT_EQ(capture.config.GetTargetCU(se), 2);
        EXPECT_EQ((masks[se - 1] >> 4) & 0xf, 2u);
    }
}

TEST(SqttHarvestTest, FallsBackToShaderArrayOne)
{
    auto info = w7800();
    std::memset(info.cu_bitmap.bits, 0, sizeof(info.cu_bitmap.bits));
    info.se_num               = 1;
    info.cu_num               = 2;
    info.cu_bitmap.bits[0][1] = 0x30;  // Only SA1 WGP2 is active.
    Builder builder(&info);
    Capture capture;
    capture.config.se_mask = 1;
    builder.Begin(&capture.commands, &capture.config);
    auto masks = programmed_values(info, capture.commands);
    ASSERT_EQ(masks.size(), 1u);
    EXPECT_EQ(capture.config.GetTargetSA(0), 1u);
    EXPECT_EQ(capture.config.GetTargetCU(0), 2);
    EXPECT_EQ((masks[0] >> 4) & 0xf, 2u);
    EXPECT_EQ((masks[0] >> 9) & 1, 1u);
}

TEST(SqttHarvestTest, MissingOrIncompleteTopologyPreservesSelection)
{
    for(bool missing : {false, true})
    {
        auto info = w7800();
        if(missing)
            std::memset(info.cu_bitmap.bits, 0, sizeof(info.cu_bitmap.bits));
        else
            info.cu_bitmap.bits[1][0] = 0;
        Builder builder(&info);
        Capture capture;
        capture.config.se_mask = 1;
        builder.Begin(&capture.commands, &capture.config);
        EXPECT_EQ(capture.config.GetSEmask(), 1u);
        EXPECT_EQ(capture.config.GetTargetCU(0), 0);
    }
}

TEST(SqttHarvestTest, LegacyMaskRetainsPhysicalSelection)
{
    auto    info = w7800();
    Builder builder(&info);
    Capture capture;
    capture.config.se_mask               = 1;
    capture.config.deprecated_mask       = 1;
    capture.config.deprecated_tokenMask  = 1;
    capture.config.deprecated_tokenMask2 = 1;
    builder.Begin(&capture.commands, &capture.config);
    EXPECT_EQ(capture.config.GetSEmask(), 1u);
    EXPECT_EQ(capture.config.GetTargetCU(0), 0);
}

TEST(SqttHarvestTest, EndDoesNotReadInactiveEngineRegisters)
{
    auto    info = w7800();
    Builder builder(&info);
    Capture capture;
    builder.Begin(&capture.commands, &capture.config);
    capture.commands.Clear();
    builder.End(&capture.commands, &capture.config);
    auto selections =
        programmed_values(info, capture.commands, gfx11_cntx_prim::GRBM_GFX_INDEX_ADDR);
    EXPECT_EQ(
        std::count(
            selections.begin(), selections.end(), gfx11_cntx_prim::grbm_se_sh_index_value(0, 0)),
        0);
    for(int se = 1; se < 6; ++se)
        EXPECT_EQ(std::count(selections.begin(),
                             selections.end(),
                             gfx11_cntx_prim::grbm_se_sh_index_value(se, 0)),
                  1);
}
