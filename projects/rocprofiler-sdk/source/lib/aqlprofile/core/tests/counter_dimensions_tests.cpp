// Copyright © 2026 Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/aqlprofile/core/counter_dimensions.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string_view>
#include <vector>

namespace
{
struct Coordinate
{
    std::string_view name       = {};
    int              extent     = 0;
    int              coordinate = 0;
};

using coordinate_vec_t = std::vector<Coordinate>;

aqlprofile_agent_handle_t
register_test_agent(const char* gfxip)
{
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = gfxip;
    info.xcc_num              = 1;
    info.se_num               = 4;
    info.cu_num               = 64;
    info.shader_arrays_per_se = 2;
    return aql_profile::RegisterAgent(&info);
}

// Pm4Factory caches by name and cu_num, so gfxip must be a key unique to the
// bitmap.
aqlprofile_agent_handle_t
register_partial_wgp_agent(const char* gfxip)
{
    // Four active WGPs per SA at physical slots {0, 1, 2, 4}: active count 4,
    // physical span 5.
    constexpr uint32_t    wgp_mask = 0x33Fu;
    std::vector<uint32_t> bitmap(2 * 2, wgp_mask);

    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = gfxip;
    info.xcc_num              = 1;
    info.se_num               = 2;
    info.cu_num               = 32;
    info.shader_arrays_per_se = 2;
    info.cu_bitmap_num_xcc    = 1;
    info.cu_bitmap_num_se     = 2;
    info.cu_bitmap_num_sa     = 2;
    info.cu_bitmap            = bitmap.data();

    return aql_profile::RegisterAgent(&info);
}

hsa_status_t
collect_coordinate(int, int, int extent, int coordinate, const char* name, void* userdata)
{
    static_cast<coordinate_vec_t*>(userdata)->push_back({name, extent, coordinate});
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
iterate_coordinates(aqlprofile_agent_handle_t           agent,
                    hsa_ven_amd_aqlprofile_block_name_t block_name,
                    uint64_t                            counter_id,
                    coordinate_vec_t&                   coordinates)
{
    aqlprofile_pmc_event_t event{};
    event.block_name = block_name;
    return aqlprofile_iterate_event_coord(
        agent, event, counter_id, collect_coordinate, &coordinates);
}

std::vector<std::string_view>
get_names(const coordinate_vec_t& coordinates)
{
    std::vector<std::string_view> names = {};
    for(const auto& coordinate : coordinates)
        names.emplace_back(coordinate.name);
    return names;
}

std::vector<int>
get_extents(const coordinate_vec_t& coordinates)
{
    std::vector<int> extents = {};
    for(const auto& coordinate : coordinates)
        extents.emplace_back(coordinate.extent);
    return extents;
}

std::vector<int>
get_values(const coordinate_vec_t& coordinates)
{
    std::vector<int> values = {};
    for(const auto& coordinate : coordinates)
        values.emplace_back(coordinate.coordinate);
    return values;
}
}  // namespace

TEST(CounterDimensionsTest, Gfx11V2WgpRetainsSingletonInstanceAndChangesFastest)
{
    const auto agent = register_test_agent("gfx1100");

    constexpr uint64_t counter_id  = (((2 * 2) + 1) * 4) + 1;
    coordinate_vec_t   coordinates = {};
    ASSERT_EQ(
        iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, counter_id, coordinates),
        HSA_STATUS_SUCCESS);

    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("INSTANCE", "SE", "SA", "WGP"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(1, 4, 2, 4));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(0, 2, 1, 1));

    coordinates.clear();
    ASSERT_EQ(iterate_coordinates(
                  agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, counter_id + 1, coordinates),
              HSA_STATUS_SUCCESS);
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(0, 2, 1, 2));
}

TEST(CounterDimensionsTest, HarvestedGfx11UsesPhysicalWgpSpan)
{
    const auto agent = register_partial_wgp_agent("gfx1151-harvested-coordinate-test");

    const auto* factory = aql_profile::Pm4Factory::Create(agent);
    ASSERT_EQ(factory->GetNumWGPs(), 5);

    coordinate_vec_t coordinates = {};
    ASSERT_EQ(iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 4, coordinates),
              HSA_STATUS_SUCCESS);
    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("INSTANCE", "SE", "SA", "WGP"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(1, 2, 2, 5));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(0, 0, 0, 4));

    coordinates.clear();
    ASSERT_EQ(iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 5, coordinates),
              HSA_STATUS_SUCCESS);
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(0, 0, 1, 0));
}

TEST(CounterDimensionsTest, Gfx12UnavailableWgpUsesPhysicalWgpSpan)
{
    const auto agent = register_partial_wgp_agent("gfx1201-unavailable-wgp-coordinate-test");

    const auto* factory = aql_profile::Pm4Factory::Create(agent);
    ASSERT_EQ(factory->GetNumWGPs(), 5);

    coordinate_vec_t coordinates = {};
    ASSERT_EQ(iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 4, coordinates),
              HSA_STATUS_SUCCESS);
    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("SE", "SA", "WGP"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(2, 2, 5));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(0, 0, 4));
}

TEST(CounterDimensionsTest, Gfx12WgpIncludesExplicitInstance)
{
    const auto agent = register_test_agent("gfx1200");

    constexpr uint64_t counter_id  = ((((1 * 4) + 2) * 2 + 1) * 4) + 1;
    coordinate_vec_t   coordinates = {};
    ASSERT_EQ(
        iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, counter_id, coordinates),
        HSA_STATUS_SUCCESS);

    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("INSTANCE", "SE", "SA", "WGP"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(2, 4, 2, 4));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(1, 2, 1, 1));
}

TEST(CounterDimensionsTest, Gfx12WgpOmitsSingletonInstance)
{
    const auto agent = register_test_agent("gfx1200");

    constexpr uint64_t counter_id  = (((2 * 2) + 1) * 4) + 1;
    coordinate_vec_t   coordinates = {};
    ASSERT_EQ(
        iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, counter_id, coordinates),
        HSA_STATUS_SUCCESS);

    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("SE", "SA", "WGP"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(4, 2, 4));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(2, 1, 1));
}

TEST(CounterDimensionsTest, NonWgpOrderIsUnchanged)
{
    const auto agent = register_test_agent("gfx1100");

    constexpr uint64_t counter_id  = ((2 * 4 + 3) * 2) + 1;
    coordinate_vec_t   coordinates = {};
    ASSERT_EQ(
        iterate_coordinates(agent, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, counter_id, coordinates),
        HSA_STATUS_SUCCESS);

    EXPECT_THAT(get_names(coordinates), testing::ElementsAre("INSTANCE", "SE", "SA"));
    EXPECT_THAT(get_extents(coordinates), testing::ElementsAre(4, 4, 2));
    EXPECT_THAT(get_values(coordinates), testing::ElementsAre(2, 3, 1));
}
