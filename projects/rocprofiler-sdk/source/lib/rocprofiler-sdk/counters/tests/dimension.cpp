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

// Allow testing of deprecated calls
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

using namespace rocprofiler::counters::test_constants;

namespace
{
void
check_dim_pos(rocprofiler_counter_instance_id_t                                 test_id,
              rocprofiler::counters::rocprofiler_profile_counter_instance_types dim,
              size_t                                                            expected)
{
    EXPECT_EQ(rec_to_dim_pos(test_id, dim), expected);
    size_t pos = 0;
    rocprofiler_query_record_dimension_position(
        test_id, static_cast<rocprofiler_counter_dimension_id_t>(dim), &pos);
    EXPECT_EQ(pos, expected);
}

void
check_counter_id(rocprofiler_counter_instance_id_t id, uint64_t expected_base_metric)
{
    using namespace rocprofiler::counters;

    rocprofiler_counter_id_t api_id = {.handle = 0};
    rocprofiler_query_record_counter_id(id, &api_id);

    auto reconstructed_id = rec_to_counter_id(id);

    EXPECT_EQ(reconstructed_id.handle, expected_base_metric);
    EXPECT_EQ(api_id.handle, expected_base_metric);

    // Both methods should return the same counter ID
    EXPECT_EQ(reconstructed_id.handle, api_id.handle);
}
}  // namespace

TEST(dimension, set_get)
{
    using namespace rocprofiler::counters;
    int64_t                           max_counter_val = (std::numeric_limits<uint64_t>::max() >>
                               (64 - (DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST)));
    rocprofiler_counter_instance_id_t test_id         = 0;
    rocprofiler_counter_id_t          test_counter{.handle = 123};

    set_counter_in_rec(test_id, test_counter);
    // 0x007B000000000000 = decimal counter id 123 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x007B000000000000);

    test_counter.handle = 321;
    set_counter_in_rec(test_id, test_counter);
    // 0x0141000000000000 = decimal counter id 321 << DIM_BIT_LENGTH
    EXPECT_EQ(test_id, 0x0141000000000000);
    check_counter_id(test_id, 321);

    // Test multiples of i, setting/getting those values across all
    // dimensions
    for(size_t multi_factor = 1; multi_factor < 7; multi_factor++)
    {
        for(size_t i = 1; i < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); i++)
        {
            auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
            set_dim_in_rec(test_id, dim, i);
            check_dim_pos(test_id, dim, i);
            set_dim_in_rec(test_id, dim, i * multi_factor);
            for(size_t j = 1; j < static_cast<size_t>(ROCPROFILER_DIMENSION_LAST); j++)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                check_dim_pos(test_id,
                              static_cast<rocprofiler_profile_counter_instance_types>(j),
                              max_counter_val);
                check_dim_pos(test_id, dim, i * multi_factor);
            }

            for(size_t j = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); j > 0; j--)
            {
                if(i == j) continue;
                set_dim_in_rec(test_id,
                               static_cast<rocprofiler_profile_counter_instance_types>(j),
                               max_counter_val);
                check_dim_pos(
                    test_id, (rocprofiler_profile_counter_instance_types) j, max_counter_val);
                check_dim_pos(test_id, dim, i * multi_factor);
            }

            // Check that name exists
            EXPECT_TRUE(rocprofiler::common::get_val(
                rocprofiler::counters::dimension_map(),
                static_cast<rocprofiler_profile_counter_instance_types>(i)));
        }
    }

    for(size_t i = static_cast<size_t>(ROCPROFILER_DIMENSION_LAST - 1); i > 0; i--)
    {
        auto dim = static_cast<rocprofiler_profile_counter_instance_types>(i);
        set_dim_in_rec(test_id, dim, i * 5);
        check_dim_pos(test_id, dim, i * 5);
        set_dim_in_rec(test_id, dim, i * 3);
        check_dim_pos(test_id, dim, i * 3);
        for(size_t j = 1; j < 64; j++)
        {
            test_id = 0;
            set_dim_in_rec(test_id, dim, j);
            check_dim_pos(test_id, dim, j);
        }
    }

    test_counter.handle = 123;
    set_counter_in_rec(test_id, test_counter);
    check_counter_id(test_id, 123);

    // Test that all bits can be set/fetched for dims, 0xFAFBFCFDFEFF is a random
    // collection of 48 bits.
    set_dim_in_rec(test_id, ROCPROFILER_DIMENSION_NONE, 0xFAFBFCFDFEFF);
    check_dim_pos(test_id, ROCPROFILER_DIMENSION_NONE, 0xFAFBFCFDFEFF);
    check_counter_id(test_id, 123);
}

using namespace rocprofiler;

namespace
{
auto
findDeviceMetrics(const hsa::AgentCache& agent, const std::unordered_set<std::string>& metrics)
{
    std::vector<counters::Metric> ret;
    auto                          mets         = counters::loadMetrics();
    const auto&                   all_counters = mets->arch_to_metric;

    ROCP_INFO << "Looking up counters for " << std::string(agent.name());
    const auto* gfx_metrics = common::get_val(all_counters, std::string(agent.name()));
    if(!gfx_metrics)
    {
        ROCP_ERROR << "No counters found for " << std::string(agent.name());
        return ret;
    }

    for(const auto& counter : *gfx_metrics)
    {
        if(metrics.count(counter.name()) > 0 || metrics.empty())
        {
            ret.push_back(counter);
        }
    }
    return ret;
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

}  // namespace

TEST(dimension, block_dim_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics = findDeviceMetrics(agent, {});
        ASSERT_FALSE(metrics.empty());
        ASSERT_TRUE(agent.get_rocp_agent());
        // aql::AQLPacketConstruct pkt(agent, metrics);
        // auto                    test_pkt = pkt.construct_packet(get_ext_table());
        for(const auto& metric : metrics)
        {
            /**
             * Calculate expected dimensions from AQL Profiler
             */
            std::unordered_map<counters::rocprofiler_profile_counter_instance_types, uint64_t>
                rocp_dims;
            ROCP_INFO << metric.name() << " " << metric.constant();
            if(!metric.constant().empty())
            {
                rocp_dims[counters::rocprofiler_profile_counter_instance_types::
                              ROCPROFILER_DIMENSION_INSTANCE] = 1;
            }
            else if(!metric.expression().empty())
            {
                continue;
            }
            else
            {
                aql::CounterPacketConstruct pkt_gen(agent.get_rocp_agent()->id, {metric});
                const auto&                 events = pkt_gen.get_counter_events(metric);
                for(const auto& event : events)
                {
                    std::map<int, uint64_t> dims;
                    auto status = aql::get_dim_info(agent.get_rocp_agent()->id, event, 0, dims);
                    CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS)
                        << rocprofiler_get_status_string(status);
                    for(const auto& [id, extent] : dims)
                    {
                        if(const auto* inst_type = rocprofiler::common::get_val(
                               counters::aqlprofile_id_to_rocprof_instance(), id))
                        {
                            rocp_dims.emplace(*inst_type, 0).first->second = extent;
                        }
                    }
                }
            }

            /**
             * Compare with actual
             */
            auto dims = getBlockDimensions(agent.get_rocp_agent()->id, metric);
            EXPECT_FALSE(dims.empty());
            EXPECT_EQ(dims.size(), rocp_dims.size());
            for(const auto& dim : dims)
            {
                const auto* ptr = rocprofiler::common::get_val(rocp_dims, dim.type());
                ASSERT_TRUE(ptr) << fmt::format("{}", dim);
                EXPECT_EQ(*ptr, dim.size()) << fmt::format("{}", dim);
                EXPECT_EQ(std::string(counters::dimension_map().at(dim.type())), dim.name())
                    << fmt::format("{}", dim);
            }

            /**
             * Check this value exists in the dimension cache
             */
            auto        dim_ptr   = counters::get_dimension_cache(agent.get_rocp_agent()->id);
            const auto* dim_cache = rocprofiler::common::get_val(dim_ptr->id_to_dim, metric.id());
            ASSERT_TRUE(dim_cache);
            EXPECT_EQ(fmt::format("{}", fmt::join(dims, "|")),
                      fmt::format("{}", fmt::join(*dim_cache, "|")));

            /**
             * Check counter instance count public API
             */
            size_t instance_count            = 0;
            size_t calculated_instance_count = 0;
            rocprofiler_query_counter_instance_count(
                agent.get_rocp_agent()->id, {.handle = metric.id()}, &instance_count);
            for(const auto& dim : dims)
            {
                if(calculated_instance_count == 0)
                    calculated_instance_count = dim.size();
                else if(dim.size() > 0)
                    calculated_instance_count = dim.size() * calculated_instance_count;
            }
            EXPECT_EQ(instance_count, calculated_instance_count);

            /**
             * Check the public API returns this value.
             */
            rocprofiler_iterate_counter_dimensions(
                {.handle = metric.id()},
                [](rocprofiler_counter_id_t,
                   const rocprofiler_counter_record_dimension_info_t* dim_info,
                   size_t                                             num_dims,
                   void* user_data) -> rocprofiler_status_t {
                    auto expected_dims = *static_cast<
                        std::unordered_map<counters::rocprofiler_profile_counter_instance_types,
                                           uint64_t>*>(user_data);
                    EXPECT_EQ(num_dims, expected_dims.size());
                    for(size_t i = 0; i < num_dims; i++)
                    {
                        const auto* lookup_ptr = rocprofiler::common::get_val(
                            expected_dims,
                            static_cast<counters::rocprofiler_profile_counter_instance_types>(
                                dim_info[i].id));
                        EXPECT_TRUE(lookup_ptr);
                        if(!lookup_ptr) return ROCPROFILER_STATUS_ERROR;
                        EXPECT_EQ(*lookup_ptr, dim_info[i].instance_size);
                        EXPECT_EQ(
                            counters::dimension_map().at(
                                static_cast<counters::rocprofiler_profile_counter_instance_types>(
                                    dim_info[i].id)),
                            std::string(dim_info[i].name));
                    }
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                static_cast<void*>(&rocp_dims));
        }
    }

    hsa_shut_down();
}
TEST(dimension, cu_bitmap_wgp_extraction)
{
    // Test the popcount/2 logic used for WGP extraction from CU bitmaps.
    // This tests the computation independently of AQLProfile internals.

    auto cu_bitmap_to_wgp = [](uint32_t bitmap) -> uint32_t {
        return __builtin_popcount(bitmap) / 2;
    };

    // Symmetric: 16 CUs = 8 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0xFFFF), 8);

    // Asymmetric: 18 CUs = 9 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0x3FFFF), 9);

    // Post-harvest: 14 CUs = 7 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0x3FFF), 7);

    // Non-contiguous gaps: 10 CUs set = 5 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0b10101010101010101010), 5);

    // Edge: all zeros = 0 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0x0), 0);

    // Full 32-bit = 16 WGPs
    EXPECT_EQ(cu_bitmap_to_wgp(0xFFFFFFFF), 16);

    // Max extent should be max across all SE/SA pairs
    uint32_t cu_bitmap[4][4] = {};
    cu_bitmap[0][0]          = 0x3FFFF;  // 9 WGPs
    cu_bitmap[0][1]          = 0xFFFF;   // 8 WGPs
    cu_bitmap[1][0]          = 0x3FFFF;  // 9 WGPs
    cu_bitmap[1][1]          = 0x3FFF;   // 7 WGPs (post-harvest)

    uint32_t max_wgp = 0;
    for(int se = 0; se < 4; se++)
        for(int sa = 0; sa < 4; sa++)
        {
            uint32_t wgp = cu_bitmap_to_wgp(cu_bitmap[se][sa]);
            if(wgp > max_wgp) max_wgp = wgp;
        }
    EXPECT_EQ(max_wgp, 9);
}

// --- GPU-free AQLProfile agent registration and coordinate tests ---

namespace
{
struct CoordEntry
{
    int         position;
    int         id;
    int         extent;
    int         coordinate;
    std::string name;
};

hsa_status_t
coord_callback(int position, int id, int extent, int coordinate, const char* name, void* userdata)
{
    auto& coords = *static_cast<std::vector<CoordEntry>*>(userdata);
    coords.push_back({position, id, extent, coordinate, std::string(name)});
    return HSA_STATUS_SUCCESS;
}
}  // namespace

TEST(dimension, register_agent_v0)
{
    aqlprofile_agent_handle_t handle{};
    aqlprofile_agent_info_t   info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V0);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

TEST(dimension, register_agent_v1)
{
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v1_t info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0x1234;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V1);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

TEST(dimension, register_agent_v2_with_cu_bitmap)
{
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0x5678;
    info.cu_bitmap[0][0]      = 0x3FFFF;  // 18 CUs = 9 WGPs
    info.cu_bitmap[1][0]      = 0xFFFF;   // 16 CUs = 8 WGPs

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

TEST(dimension, register_agent_null_info_fails)
{
    aqlprofile_agent_handle_t handle{};
    auto status = aqlprofile_register_agent_info(&handle, nullptr, AQLPROFILE_AGENT_VERSION_V0);
    EXPECT_NE(status, HSA_STATUS_SUCCESS);
}

TEST(dimension, register_agent_invalid_version_fails)
{
    aqlprofile_agent_handle_t handle{};
    aqlprofile_agent_info_t   info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_NONE);
    EXPECT_NE(status, HSA_STATUS_SUCCESS);

    status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_LAST);
    EXPECT_NE(status, HSA_STATUS_SUCCESS);
}

TEST(dimension, iterate_event_coord_gfx900)
{
    // Register a gfx900 agent and verify coordinate decomposition for SQ block
    aqlprofile_agent_handle_t handle{};
    aqlprofile_agent_info_t   info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V0);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    // Sample ID 0 should return valid coordinates
    std::vector<CoordEntry> coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    EXPECT_GT(coords.size(), 0u);

    // Verify all coordinates are within bounds
    for(const auto& c : coords)
    {
        EXPECT_GE(c.coordinate, 0);
        EXPECT_LT(c.coordinate, c.extent) << "Coord " << c.name << " out of bounds";
        EXPECT_FALSE(c.name.empty());
    }
}

TEST(dimension, iterate_event_coord_all_samples_valid)
{
    // Register agent, then iterate all sample IDs and verify coordinates are in bounds
    aqlprofile_agent_handle_t handle{};
    aqlprofile_agent_info_t   info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V0);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    // Get dimension extents from sample 0
    std::vector<CoordEntry> first_coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &first_coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    ASSERT_GT(first_coords.size(), 0u);

    // Calculate total samples from extents
    size_t total_samples = 1;
    for(const auto& c : first_coords)
        total_samples *= static_cast<size_t>(c.extent);

    // Verify every sample ID produces valid coordinates
    for(size_t sample_id = 0; sample_id < total_samples; sample_id++)
    {
        std::vector<CoordEntry> coords;
        status = aqlprofile_iterate_event_coord(handle, event, sample_id, coord_callback, &coords);
        ASSERT_EQ(status, HSA_STATUS_SUCCESS) << "Failed at sample_id=" << sample_id;
        ASSERT_EQ(coords.size(), first_coords.size());

        for(size_t i = 0; i < coords.size(); i++)
        {
            EXPECT_GE(coords[i].coordinate, 0);
            EXPECT_LT(coords[i].coordinate, coords[i].extent)
                << "sample_id=" << sample_id << " dim=" << coords[i].name;
            EXPECT_EQ(coords[i].extent, first_coords[i].extent);
            EXPECT_EQ(coords[i].name, first_coords[i].name);
        }
    }
}

TEST(dimension, iterate_event_coord_v2_asymmetric_bitmap)
{
    // Register with V2 + asymmetric cu_bitmap, verify coordinate decomposition
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0xABCD;
    // Asymmetric: SE0 has 9 WGPs, SE1 has 8, SE2 has 9, SE3 has 7
    info.cu_bitmap[0][0] = 0x3FFFF;  // 18 CUs = 9 WGPs
    info.cu_bitmap[0][1] = 0xFFFF;   // 16 CUs = 8 WGPs
    info.cu_bitmap[1][0] = 0x3FFFF;  // 18 CUs = 9 WGPs
    info.cu_bitmap[1][1] = 0x3FFF;   // 14 CUs = 7 WGPs

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    EXPECT_GT(coords.size(), 0u);

    // Verify coordinates are valid
    for(const auto& c : coords)
    {
        EXPECT_GE(c.coordinate, 0);
        EXPECT_LT(c.coordinate, c.extent) << "Coord " << c.name << " out of bounds";
    }
}

TEST(dimension, symmetric_cu_bitmap_backward_compat)
{
    // Register with V2 but symmetric cu_bitmap — should behave like V0
    aqlprofile_agent_handle_t handle_v0{};
    aqlprofile_agent_info_t   info_v0{};
    info_v0.agent_gfxip          = "gfx900";
    info_v0.cu_num               = 64;
    info_v0.se_num               = 4;
    info_v0.xcc_num              = 1;
    info_v0.shader_arrays_per_se = 2;

    auto status = aqlprofile_register_agent_info(&handle_v0, &info_v0, AQLPROFILE_AGENT_VERSION_V0);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_agent_handle_t  handle_v2{};
    aqlprofile_agent_info_v2_t info_v2{};
    info_v2.agent_gfxip          = "gfx900";
    info_v2.cu_num               = 64;
    info_v2.se_num               = 4;
    info_v2.xcc_num              = 1;
    info_v2.shader_arrays_per_se = 2;
    info_v2.domain               = 0;
    info_v2.location_id          = 0;
    // Symmetric: all SEs/SAs have same CU count
    for(int se = 0; se < 4; se++)
        for(int sa = 0; sa < 2; sa++)
            info_v2.cu_bitmap[se][sa] = 0xFF;  // 8 CUs per SA

    status = aqlprofile_register_agent_info(&handle_v2, &info_v2, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords_v0;
    std::vector<CoordEntry> coords_v2;
    aqlprofile_iterate_event_coord(handle_v0, event, 0, coord_callback, &coords_v0);
    aqlprofile_iterate_event_coord(handle_v2, event, 0, coord_callback, &coords_v2);

    // Same number of dimensions
    ASSERT_EQ(coords_v0.size(), coords_v2.size());

    // Same dimension names and extents
    for(size_t i = 0; i < coords_v0.size(); i++)
    {
        EXPECT_EQ(coords_v0[i].name, coords_v2[i].name);
        EXPECT_EQ(coords_v0[i].extent, coords_v2[i].extent)
            << "Mismatch in dim " << coords_v0[i].name;
    }
}

// --- GFX11 tests: WGP dimension is only active on GFX11+ ---
// These tests verify AQLProfile's decoding of asymmetric CU bitmap data
// using GFX11 agents where the WGP dimension is present in coordinate decomposition.

TEST(dimension, gfx11_wgp_dimension_present)
{
    // Register a GFX11 agent and verify SQ block has WGP dimension
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx1100";
    info.cu_num               = 16;
    info.se_num               = 2;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0;

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // GFX11 SQ should have dimensions: WGP, INSTANCE, SE, SA
    ASSERT_EQ(coords.size(), 4u);
    EXPECT_EQ(coords[0].name, "WGP");
    EXPECT_EQ(coords[1].name, "INSTANCE");
    EXPECT_EQ(coords[2].name, "SE");
    EXPECT_EQ(coords[3].name, "SA");

    // Uniform: wgp_num = (16/2 + 4-1) / 4 = 2
    EXPECT_EQ(coords[0].extent, 2);  // WGP
    EXPECT_EQ(coords[1].extent, 1);  // INSTANCE (SQ has instance_count=1)
    EXPECT_EQ(coords[2].extent, 2);  // SE
    EXPECT_EQ(coords[3].extent, 2);  // SA
}

TEST(dimension, gfx11_asymmetric_wgp_extent)
{
    // Register GFX11 agent with asymmetric cu_bitmap and verify WGP extent = max(popcount/2)
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx1100";
    info.cu_num               = 34;  // total: 9+8+9+8 = 34
    info.se_num               = 2;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0;
    // Asymmetric layout:
    info.cu_bitmap[0][0] = 0x3FFFF;  // 18 CUs = 9 WGPs
    info.cu_bitmap[0][1] = 0xFFFF;   // 16 CUs = 8 WGPs
    info.cu_bitmap[1][0] = 0x3FFFF;  // 18 CUs = 9 WGPs
    info.cu_bitmap[1][1] = 0xFFFF;   // 16 CUs = 8 WGPs

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    ASSERT_EQ(coords.size(), 4u);
    EXPECT_EQ(coords[0].name, "WGP");
    // WGP extent should be max(9, 8, 9, 8) = 9
    EXPECT_EQ(coords[0].extent, 9);
    EXPECT_EQ(coords[2].name, "SE");
    EXPECT_EQ(coords[2].extent, 2);
    EXPECT_EQ(coords[3].name, "SA");
    EXPECT_EQ(coords[3].extent, 2);
}

TEST(dimension, gfx11_asymmetric_all_samples_valid)
{
    // Register GFX11 agent with asymmetric cu_bitmap,
    // iterate ALL sample IDs and verify coordinates are in bounds
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx1100";
    info.cu_num               = 33;  // 9+8+9+7 = 33
    info.se_num               = 2;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0;
    // Asymmetric + harvest:
    info.cu_bitmap[0][0] = 0x3FFFF;  // 9 WGPs
    info.cu_bitmap[0][1] = 0xFFFF;   // 8 WGPs
    info.cu_bitmap[1][0] = 0x3FFFF;  // 9 WGPs
    info.cu_bitmap[1][1] = 0x3FFF;   // 7 WGPs (post-harvest)

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    // Get dimension extents from sample 0
    std::vector<CoordEntry> first_coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &first_coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    ASSERT_EQ(first_coords.size(), 4u);
    EXPECT_EQ(first_coords[0].name, "WGP");
    // Max WGP = max(9, 8, 9, 7) = 9
    EXPECT_EQ(first_coords[0].extent, 9);

    // Total samples in the max-extent cartesian space
    size_t total_samples = 1;
    for(const auto& c : first_coords)
        total_samples *= static_cast<size_t>(c.extent);

    // 9 * 1 * 2 * 2 = 36
    EXPECT_EQ(total_samples, 36u);

    // Verify every sample ID produces valid in-bounds coordinates
    for(size_t sample_id = 0; sample_id < total_samples; sample_id++)
    {
        std::vector<CoordEntry> coords;
        status = aqlprofile_iterate_event_coord(handle, event, sample_id, coord_callback, &coords);
        ASSERT_EQ(status, HSA_STATUS_SUCCESS) << "Failed at sample_id=" << sample_id;
        ASSERT_EQ(coords.size(), first_coords.size());

        for(size_t i = 0; i < coords.size(); i++)
        {
            EXPECT_GE(coords[i].coordinate, 0);
            EXPECT_LT(coords[i].coordinate, coords[i].extent)
                << "sample_id=" << sample_id << " dim=" << coords[i].name;
            EXPECT_EQ(coords[i].extent, first_coords[i].extent);
            EXPECT_EQ(coords[i].name, first_coords[i].name);
        }
    }
}

TEST(dimension, gfx11_noncontiguous_harvest_wgp_extent)
{
    // Test non-contiguous harvesting: gaps in the bitmap
    // Verifies WGP count is based on popcount, not bit position
    aqlprofile_agent_handle_t  handle{};
    aqlprofile_agent_info_v2_t info{};
    info.agent_gfxip          = "gfx1100";
    info.cu_num               = 20;
    info.se_num               = 2;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0;
    // Non-contiguous: gaps between active CU pairs
    // 0b110011001100110011 = WGPs at physical indices 0,2,4,6,8 → 5 WGPs (bits set in pairs)
    info.cu_bitmap[0][0] = 0x33333;  // popcount=10 → 5 WGPs
    info.cu_bitmap[0][1] = 0x33333;  // popcount=10 → 5 WGPs
    info.cu_bitmap[1][0] = 0x33333;  // popcount=10 → 5 WGPs
    info.cu_bitmap[1][1] = 0x33333;  // popcount=10 → 5 WGPs

    auto status = aqlprofile_register_agent_info(&handle, &info, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords;
    status = aqlprofile_iterate_event_coord(handle, event, 0, coord_callback, &coords);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    ASSERT_EQ(coords.size(), 4u);
    EXPECT_EQ(coords[0].name, "WGP");
    EXPECT_EQ(coords[0].extent, 5);  // popcount(0x33333)=10, /2=5
}

TEST(dimension, gfx11_symmetric_bitmap_matches_uniform)
{
    // Verify that a symmetric cu_bitmap produces the same WGP extent
    // as the uniform calculation from cu_num
    aqlprofile_agent_handle_t  handle_no_bitmap{};
    aqlprofile_agent_info_v2_t info_no_bitmap{};
    info_no_bitmap.agent_gfxip          = "gfx1100";
    info_no_bitmap.cu_num               = 16;
    info_no_bitmap.se_num               = 2;
    info_no_bitmap.xcc_num              = 1;
    info_no_bitmap.shader_arrays_per_se = 2;
    info_no_bitmap.domain               = 0;
    info_no_bitmap.location_id          = 0;
    // No cu_bitmap — uses uniform calculation

    auto status = aqlprofile_register_agent_info(
        &handle_no_bitmap, &info_no_bitmap, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_agent_handle_t  handle_bitmap{};
    aqlprofile_agent_info_v2_t info_bitmap{};
    info_bitmap.agent_gfxip          = "gfx1100";
    info_bitmap.cu_num               = 16;
    info_bitmap.se_num               = 2;
    info_bitmap.xcc_num              = 1;
    info_bitmap.shader_arrays_per_se = 2;
    info_bitmap.domain               = 0;
    info_bitmap.location_id          = 0;
    // Symmetric bitmap: 4 CUs per SA = 2 WGPs per SA (same as uniform)
    info_bitmap.cu_bitmap[0][0] = 0xF;  // 4 CUs = 2 WGPs
    info_bitmap.cu_bitmap[0][1] = 0xF;
    info_bitmap.cu_bitmap[1][0] = 0xF;
    info_bitmap.cu_bitmap[1][1] = 0xF;

    status =
        aqlprofile_register_agent_info(&handle_bitmap, &info_bitmap, AQLPROFILE_AGENT_VERSION_V2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    aqlprofile_pmc_event_t event{};
    event.block_name  = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    event.block_index = 0;
    event.event_id    = 0;
    event.flags.raw   = 0;

    std::vector<CoordEntry> coords_no_bitmap;
    std::vector<CoordEntry> coords_bitmap;
    aqlprofile_iterate_event_coord(handle_no_bitmap, event, 0, coord_callback, &coords_no_bitmap);
    aqlprofile_iterate_event_coord(handle_bitmap, event, 0, coord_callback, &coords_bitmap);

    ASSERT_EQ(coords_no_bitmap.size(), coords_bitmap.size());
    for(size_t i = 0; i < coords_no_bitmap.size(); i++)
    {
        EXPECT_EQ(coords_no_bitmap[i].name, coords_bitmap[i].name);
        EXPECT_EQ(coords_no_bitmap[i].extent, coords_bitmap[i].extent)
            << "Mismatch in dim " << coords_no_bitmap[i].name;
    }
}

TEST(dimension, logical_to_physical_wgp_mapping)
{
    // Test the logical-to-physical WGP mapping computation directly.
    // This mirrors the logic in GpuPmcBuilder::logical_to_physical_wgp().
    // With non-contiguous harvesting, logical WGP indices must map to
    // the correct physical positions by walking CU pairs in the bitmap.

    auto logical_to_physical_wgp = [](uint32_t bitmap, uint32_t logical_wgp) -> uint32_t {
        if(bitmap == 0) return logical_wgp;
        uint32_t count = 0;
        for(uint32_t phys_wgp = 0; phys_wgp < 16; phys_wgp++)
        {
            uint32_t cu_pair = (bitmap >> (phys_wgp * 2)) & 0x3;
            if(cu_pair != 0)
            {
                if(count == logical_wgp) return phys_wgp;
                count++;
            }
        }
        return logical_wgp;
    };

    // Contiguous: 0xFF = WGPs at physical 0,1,2,3
    EXPECT_EQ(logical_to_physical_wgp(0xFF, 0), 0u);
    EXPECT_EQ(logical_to_physical_wgp(0xFF, 1), 1u);
    EXPECT_EQ(logical_to_physical_wgp(0xFF, 2), 2u);
    EXPECT_EQ(logical_to_physical_wgp(0xFF, 3), 3u);

    // Gap at WGP 1: 0b11000011 = WGPs at physical 0, 3
    EXPECT_EQ(logical_to_physical_wgp(0xC3, 0), 0u);
    EXPECT_EQ(logical_to_physical_wgp(0xC3, 1), 3u);

    // Gap at WGP 0: 0b00001100 = WGP at physical 1
    EXPECT_EQ(logical_to_physical_wgp(0x0C, 0), 1u);

    // Non-contiguous pattern: 0x33333 = pairs at physical 0,2,4,6,8
    EXPECT_EQ(logical_to_physical_wgp(0x33333, 0), 0u);
    EXPECT_EQ(logical_to_physical_wgp(0x33333, 1), 2u);
    EXPECT_EQ(logical_to_physical_wgp(0x33333, 2), 4u);
    EXPECT_EQ(logical_to_physical_wgp(0x33333, 3), 6u);
    EXPECT_EQ(logical_to_physical_wgp(0x33333, 4), 8u);

    // Harvest at end: 0x3FFFF = 9 WGPs at physical 0..8
    for(uint32_t i = 0; i < 9; i++)
        EXPECT_EQ(logical_to_physical_wgp(0x3FFFF, i), i);

    // Single WGP at high position: 0x30000 = physical WGP 8
    EXPECT_EQ(logical_to_physical_wgp(0x30000, 0), 8u);

    // Zero bitmap falls back to identity
    EXPECT_EQ(logical_to_physical_wgp(0, 5), 5u);
}

#pragma GCC diagnostic pop
