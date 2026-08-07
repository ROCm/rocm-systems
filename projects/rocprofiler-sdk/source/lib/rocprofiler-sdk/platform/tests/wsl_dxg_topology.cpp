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

// Exercises the WSL/DXG topology mapping without a GPU, a DXG thunk or the HSA
// runtime: every function under test is a pure transform of a KMT node record.

#include "lib/rocprofiler-sdk/platform/wsl/dxg_topology.hpp"

#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/agent.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <set>
#include <vector>

namespace
{
using rocprofiler::platform::wsl::DxgNodeTopology;
using rocprofiler::platform::wsl::match_node_to_adapter;

// The matcher is called once per adapter with a growing consumed set; the tests
// below spell that set out explicitly, so shorten the call.
constexpr auto& match = match_node_to_adapter;

// A gfx1150 (RDNA 3.5) node as the DXG thunk reports it: 1 shader engine, 2
// SIMD arrays per engine, 2 SIMDs per CU.
DxgNodeTopology
make_gfx1150_node()
{
    auto node                      = DxgNodeTopology{};
    node.StructSize                = sizeof(DxgNodeTopology);
    node.AbiVersion                = rocprofiler::platform::wsl::kDxgNodeTopologyAbiVersion;
    node.NodeId                    = 1;  // KMT node 0 is the CPU
    node.NumFComputeCores          = 32;
    node.NumSIMDPerCU              = 2;
    node.NumShaderBanks            = 1;
    node.NumArrays                 = 2;
    node.NumCUPerArray             = 8;
    node.WaveFrontSize             = 32;
    node.MaxWavesPerSIMD           = 16;
    node.NumXcc                    = 1;
    node.EngineIdMajor             = 11;
    node.EngineIdMinor             = 5;
    node.EngineIdStepping          = 0;
    node.EngineIdUCode             = 42;
    node.SdmaUCode                 = 17;
    node.VendorId                  = 0x1002;
    node.DeviceId                  = 0x150e;
    node.LocationId                = 0x0300;
    node.FamilyID                  = 145;
    node.LuidLowPart               = 0xAABBCCDD;
    node.LuidHighPart              = 0x11;
    node.UniqueID                  = 0x0123456789ABCDEFULL;
    node.MaxEngineClockMhzFCompute = 2900;
    return node;
}

struct ForceGfxEnv
{
    explicit ForceGfxEnv(const char* value)
    {
        if(value)
            ::setenv("ROCPROFILER_FORCE_GFX", value, /*overwrite=*/1);
        else
            ::unsetenv("ROCPROFILER_FORCE_GFX");
    }

    ~ForceGfxEnv() { ::unsetenv("ROCPROFILER_FORCE_GFX"); }

    ForceGfxEnv(const ForceGfxEnv&) = delete;
    ForceGfxEnv& operator=(const ForceGfxEnv&) = delete;
};
}  // namespace

TEST(wsl_dxg_topology, derived_compute_topology)
{
    const auto node = make_gfx1150_node();
    auto       info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});

    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.simd_count, 32u);
    EXPECT_EQ(info.simd_per_cu, 2u);
    EXPECT_EQ(info.cu_count, 16u);
    EXPECT_EQ(info.num_shader_banks, 1u);
    EXPECT_EQ(info.simd_arrays_per_engine, 2u);
    // NumArrays is per shader engine, so the total is the product. Publishing
    // NumArrays directly would under-report array_count on any multi-SE part.
    EXPECT_EQ(info.array_count, 2u);
    EXPECT_EQ(info.cu_per_simd_array, 8u);
    EXPECT_EQ(info.cu_per_engine, 16u);
    EXPECT_EQ(info.wave_front_size, 32u);
    EXPECT_EQ(info.max_waves_per_simd, 16u);
    EXPECT_EQ(info.max_waves_per_cu, 32u);
    EXPECT_EQ(info.num_xcc, 1u);
}

TEST(wsl_dxg_topology, array_count_is_a_total_across_shader_engines)
{
    auto node             = make_gfx1150_node();
    node.NumShaderBanks   = 4;
    node.NumArrays        = 2;
    node.NumFComputeCores = 256;
    node.NumCUPerArray    = 16;

    auto info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});
    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.array_count, 8u);
    EXPECT_EQ(info.simd_arrays_per_engine, 2u);
    EXPECT_EQ(info.cu_count, 128u);
    EXPECT_EQ(info.cu_per_engine, 32u);
}

TEST(wsl_dxg_topology, identity_fields_are_copied)
{
    const auto node = make_gfx1150_node();
    auto       info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});

    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.vendor_id, 0x1002u);
    EXPECT_EQ(info.device_id, 0x150eu);
    EXPECT_EQ(info.location_id, 0x0300u);
    EXPECT_EQ(info.family_id, 145u);
    EXPECT_EQ(info.fw_version.ui32.uCode, 42u);
    EXPECT_EQ(info.sdma_fw_version.uCodeSDMA, 17u);
    EXPECT_EQ(info.max_engine_clk_fcompute, 2900u);

    auto uuid_low = uint64_t{0};
    std::memcpy(&uuid_low, info.uuid.bytes, sizeof(uuid_low));
    EXPECT_EQ(uuid_low, 0x0123456789ABCDEFULL);

    // Architectural constants the HSA runtime hardcodes, mirrored from the KFD path.
    EXPECT_EQ(info.workgroup_max_size, 1024u);
    EXPECT_EQ(info.grid_max_size, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
}

TEST(wsl_dxg_topology, incomplete_nodes_are_rejected)
{
    auto info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});

    for(auto zero_field : {&DxgNodeTopology::NumFComputeCores,
                           &DxgNodeTopology::NumSIMDPerCU,
                           &DxgNodeTopology::NumShaderBanks,
                           &DxgNodeTopology::NumArrays,
                           &DxgNodeTopology::WaveFrontSize})
    {
        auto node        = make_gfx1150_node();
        node.*zero_field = 0;
        EXPECT_FALSE(rocprofiler::platform::wsl::apply_node_topology(node, info));
    }
}

TEST(wsl_dxg_topology, unreported_xcc_count_defaults_to_one)
{
    auto node   = make_gfx1150_node();
    node.NumXcc = 0;

    auto info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});
    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.num_xcc, 1u);
}

TEST(wsl_dxg_topology, gfx_name_precedence)
{
    auto node = make_gfx1150_node();

    {
        const auto _env = ForceGfxEnv{nullptr};
        EXPECT_EQ(rocprofiler::platform::wsl::resolve_gfx_name(node), "gfx1150");
    }

    // HSA_OVERRIDE_GFX_VERSION reaches us as OverrideEngineId with EngineId's
    // version fields left zero, exactly as the HSA runtime sees it.
    {
        const auto _env                     = ForceGfxEnv{nullptr};
        auto       overridden               = node;
        overridden.EngineIdMajor            = 0;
        overridden.EngineIdMinor            = 0;
        overridden.EngineIdStepping         = 0;
        overridden.OverrideEngineIdMajor    = 11;
        overridden.OverrideEngineIdMinor    = 0;
        overridden.OverrideEngineIdStepping = 0;
        EXPECT_EQ(rocprofiler::platform::wsl::resolve_gfx_name(overridden), "gfx1100");
    }

    {
        const auto _env = ForceGfxEnv{"gfx942"};
        EXPECT_EQ(rocprofiler::platform::wsl::resolve_gfx_name(node), "gfx942");
    }

    // A malformed override is ignored rather than fed into gfx_target_version.
    {
        const auto _env = ForceGfxEnv{"not-a-target"};
        EXPECT_EQ(rocprofiler::platform::wsl::resolve_gfx_name(node), "gfx1150");
    }

    // A node that reports no engine id yields no name, which makes the
    // enumerator omit the adapter instead of inventing a target.
    {
        const auto _env       = ForceGfxEnv{nullptr};
        auto       unknown    = node;
        unknown.EngineIdMajor = 0;
        EXPECT_TRUE(rocprofiler::platform::wsl::resolve_gfx_name(unknown).empty());
    }
}

TEST(wsl_dxg_topology, adapter_matching_prefers_luid)
{
    auto first         = make_gfx1150_node();
    first.NodeId       = 1;
    first.LuidLowPart  = 1;
    first.LuidHighPart = 0;
    first.DeviceId     = 0x7480;

    auto second         = make_gfx1150_node();
    second.NodeId       = 2;
    second.LuidLowPart  = 2;
    second.LuidHighPart = 0;
    second.DeviceId     = 0x7480;

    const auto nodes = std::vector<DxgNodeTopology>{first, second};
    const auto none  = std::set<uint32_t>{};

    // Identical device ids: only the LUID disambiguates the two adapters.
    EXPECT_EQ(match(nodes, none, 2, 0, 0x7480).node, &nodes[1]);
    EXPECT_EQ(match(nodes, none, 1, 0, 0x7480).node, &nodes[0]);

    // A LUID neither node reports is not silently downgraded to a device-id
    // match, and it is not ambiguity either - the LUIDs say these are other
    // GPUs.
    const auto unmatched = match(nodes, none, 3, 0, 0x7480);
    EXPECT_EQ(unmatched.node, nullptr);
    EXPECT_FALSE(unmatched.ambiguous);
}

TEST(wsl_dxg_topology, adapter_matching_falls_back_to_device_id)
{
    auto node         = make_gfx1150_node();
    node.LuidLowPart  = 0;
    node.LuidHighPart = 0;
    node.DeviceId     = 0x150e;

    const auto nodes = std::vector<DxgNodeTopology>{node};
    const auto none  = std::set<uint32_t>{};

    EXPECT_EQ(match(nodes, none, 7, 0, 0x150e).node, &nodes[0]);
    EXPECT_EQ(match(nodes, none, 0, 0, 0x7480).node, nullptr);
    EXPECT_EQ(match(nodes, none, 0, 0, 0).node, nullptr);
}

TEST(wsl_dxg_topology, identical_device_ids_with_distinct_luids_do_not_alias)
{
    auto first         = make_gfx1150_node();
    first.NodeId       = 1;
    first.LuidLowPart  = 0x1111;
    first.LuidHighPart = 0;
    first.DeviceId     = 0x7480;

    auto second         = make_gfx1150_node();
    second.NodeId       = 2;
    second.LuidLowPart  = 0x2222;
    second.LuidHighPart = 0;
    second.DeviceId     = 0x7480;

    const auto nodes    = std::vector<DxgNodeTopology>{first, second};
    auto       consumed = std::set<uint32_t>{};

    const auto a = match(nodes, consumed, 0x2222, 0, 0x7480);
    ASSERT_NE(a.node, nullptr);
    EXPECT_EQ(a.node->NodeId, 2u);
    consumed.emplace(a.node->NodeId);

    const auto b = match(nodes, consumed, 0x1111, 0, 0x7480);
    ASSERT_NE(b.node, nullptr);
    EXPECT_EQ(b.node->NodeId, 1u);
}

TEST(wsl_dxg_topology, identical_device_ids_without_luids_are_ambiguous)
{
    auto first        = make_gfx1150_node();
    first.NodeId      = 1;
    first.LuidLowPart = 0;
    first.DeviceId    = 0x7480;

    auto second        = make_gfx1150_node();
    second.NodeId      = 2;
    second.LuidLowPart = 0;
    second.DeviceId    = 0x7480;

    const auto nodes = std::vector<DxgNodeTopology>{first, second};

    // Two indistinguishable candidates: report it rather than picking one and
    // attributing this adapter's counters to the wrong GPU.
    const auto ambiguous = match(nodes, std::set<uint32_t>{}, 0, 0, 0x7480);
    EXPECT_EQ(ambiguous.node, nullptr);
    EXPECT_TRUE(ambiguous.ambiguous);

    // Once the first is claimed, only one candidate is left and it resolves.
    const auto resolved = match(nodes, std::set<uint32_t>{1}, 0, 0, 0x7480);
    EXPECT_FALSE(resolved.ambiguous);
    ASSERT_NE(resolved.node, nullptr);
    EXPECT_EQ(resolved.node->NodeId, 2u);
}

TEST(wsl_dxg_topology, a_consumed_node_is_never_matched_twice)
{
    auto node         = make_gfx1150_node();
    node.NodeId       = 4;
    node.LuidLowPart  = 0x99;
    node.LuidHighPart = 0;

    const auto nodes = std::vector<DxgNodeTopology>{node};

    EXPECT_EQ(match(nodes, std::set<uint32_t>{}, 0x99, 0, node.DeviceId).node, &nodes[0]);

    const auto second = match(nodes, std::set<uint32_t>{4}, 0x99, 0, node.DeviceId);
    EXPECT_EQ(second.node, nullptr);
    EXPECT_FALSE(second.ambiguous);
}

TEST(wsl_dxg_topology, cu_per_simd_array_is_derived_across_shader_engines)
{
    // 128 CU on 4 shader engines with 2 arrays each is 16 CU per array. A
    // producer that divided by the per-engine array count alone would say 64.
    auto node             = make_gfx1150_node();
    node.NumShaderBanks   = 4;
    node.NumArrays        = 2;
    node.NumSIMDPerCU     = 2;
    node.NumFComputeCores = 256;
    node.NumCUPerArray    = 64;

    auto info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});
    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.cu_count, 128u);
    EXPECT_EQ(info.array_count, 8u);
    EXPECT_EQ(info.cu_per_simd_array, 16u);
}

TEST(wsl_dxg_topology, a_consistent_producer_is_reproduced_exactly)
{
    auto node             = make_gfx1150_node();
    node.NumShaderBanks   = 4;
    node.NumArrays        = 2;
    node.NumSIMDPerCU     = 2;
    node.NumFComputeCores = 256;
    node.NumCUPerArray    = 16;  // what a corrected producer reports

    auto info = rocprofiler::common::init_public_api_struct(rocprofiler_agent_t{});
    ASSERT_TRUE(rocprofiler::platform::wsl::apply_node_topology(node, info));

    EXPECT_EQ(info.cu_per_simd_array, node.NumCUPerArray);
}

TEST(wsl_dxg_topology, kmt_node_id_survives_the_transform)
{
    auto node   = make_gfx1150_node();
    node.NodeId = 7;

    const auto nodes = std::vector<DxgNodeTopology>{node};
    const auto found = match(nodes, std::set<uint32_t>{}, node.LuidLowPart, 0x11, node.DeviceId);

    ASSERT_NE(found.node, nullptr);
    EXPECT_EQ(found.node->NodeId, 7u)
        << "the enumerator publishes this as rocprofiler_agent_t::node_id, so it must not be "
           "replaced by a compacted ordinal";
}
