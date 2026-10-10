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

#include "lib/rocprofiler-sdk/kernel_replay/blit-copy-kernel.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/blit-copy.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/snapshot-plan.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace
{
namespace abi  = rocprofiler::kernel_replay::blit::kernel_abi;
namespace ms   = rocprofiler::kernel_replay::memory_snapshot;
namespace plan = rocprofiler::kernel_replay::memory_snapshot::planning;

TEST(kernel_replay_snapshot_plan, variadic_fallback_uses_gpu_cpu_then_null)
{
    auto allocator =
        plan::snapshot_allocator{plan::gpu_allocator{10}, plan::pinned_allocator{20}, {}};
    auto requests = std::vector<plan::request_t>{
        {6, 0, 100},
        {7, 1, 200},
        {4, 2, 300},
    };

    const auto result = plan::build(std::move(requests), allocator, 1);
    ASSERT_TRUE(result.complete());
    ASSERT_EQ(result.placements.size(), 3);
    EXPECT_EQ(result.placements[0].tier, ms::storage_kind::gpu_local);
    EXPECT_EQ(result.placements[0].backing_offset, 0);
    EXPECT_EQ(result.placements[1].tier, ms::storage_kind::pinned_host);
    EXPECT_EQ(result.placements[1].backing_offset, 0);
    // Fallback is non-sticky: this later smaller region still consumes the remaining GPU budget.
    EXPECT_EQ(result.placements[2].tier, ms::storage_kind::gpu_local);
    EXPECT_EQ(result.placements[2].backing_offset, 6);
    EXPECT_EQ(result.gpu_bytes, 10);
    EXPECT_EQ(result.host_bytes, 7);
}

TEST(kernel_replay_snapshot_plan, null_allocator_reports_terminal_region)
{
    auto allocator =
        plan::snapshot_allocator{plan::gpu_allocator{4}, plan::pinned_allocator{4}, {}};
    auto requests = std::vector<plan::request_t>{{5, 17, 0}};

    const auto result = plan::build(std::move(requests), allocator, 1);
    EXPECT_FALSE(result.complete());
    EXPECT_EQ(result.error.code, plan::error_code::no_backing_capacity);
    EXPECT_EQ(result.error.requested, 5);
    EXPECT_EQ(result.error.region_index, 17);
}

TEST(kernel_replay_snapshot_plan, planning_is_deterministic_and_accounts_for_alignment)
{
    auto allocator =
        plan::snapshot_allocator{plan::gpu_allocator{64}, plan::pinned_allocator{0}, {}};
    auto requests = std::vector<plan::request_t>{
        {1, 2, 300},
        {17, 0, 100},
        {1, 1, 200},
    };

    const auto result = plan::build(std::move(requests), allocator, 16);
    ASSERT_TRUE(result.complete());
    ASSERT_EQ(result.placements.size(), 3);
    EXPECT_EQ(result.placements[0].region_index, 0);
    EXPECT_EQ(result.placements[0].backing_offset, 0);
    EXPECT_EQ(result.placements[1].region_index, 1);
    EXPECT_EQ(result.placements[1].backing_offset, 32);
    EXPECT_EQ(result.placements[2].region_index, 2);
    EXPECT_EQ(result.placements[2].backing_offset, 48);
    EXPECT_EQ(result.gpu_bytes, 49);
}

TEST(kernel_replay_snapshot_plan, overflow_is_terminal_and_does_not_fall_through)
{
    auto allocator = plan::snapshot_allocator{
        plan::gpu_allocator{std::numeric_limits<size_t>::max()},
        plan::pinned_allocator{1024},
        {},
    };

    ASSERT_TRUE(allocator.allocate(std::numeric_limits<size_t>::max() - 7, 1));
    const auto overflow = allocator.allocate(16, 16);
    EXPECT_FALSE(overflow);
    EXPECT_EQ(overflow.error.code, plan::error_code::overflow);
    EXPECT_EQ(allocator.get<1>().used(), 0);

    const auto zero_alignment = allocator.allocate(1, 0);
    EXPECT_FALSE(zero_alignment);
    EXPECT_EQ(zero_alignment.error.code, plan::error_code::overflow);
    EXPECT_EQ(allocator.get<1>().used(), 0);
}

TEST(kernel_replay_blit_copy, descriptors_use_size_source_dest_order)
{
    EXPECT_EQ(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, size), 0);
    EXPECT_LT(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, size),
              offsetof(rocprofiler::kernel_replay::blit::copy_region_t, source));
    EXPECT_LT(offsetof(rocprofiler::kernel_replay::blit::copy_region_t, source),
              offsetof(rocprofiler::kernel_replay::blit::copy_region_t, dest));

    EXPECT_EQ(offsetof(abi::copy_descriptor_t, size), 0);
    EXPECT_EQ(offsetof(abi::copy_descriptor_t, source_address), sizeof(std::uint64_t));
    EXPECT_EQ(offsetof(abi::copy_descriptor_t, dest_address), 2 * sizeof(std::uint64_t));
}

TEST(kernel_replay_blit_copy, regions_pack_into_one_aligned_segment)
{
    auto regions = std::vector<ms::snapshot_region_t>{
        {1, nullptr, 0, ms::liveness_kind::tracked_allocation},
        {16, nullptr, 0, ms::liveness_kind::tracked_allocation},
        {17, nullptr, 0, ms::liveness_kind::module_variable},
    };

    const auto required = ms::assign_logical_offsets(regions);
    ASSERT_TRUE(required.has_value());
    EXPECT_EQ(*required, 49);
    EXPECT_EQ(regions[0].logical_offset, 0);
    EXPECT_EQ(regions[1].logical_offset, 16);
    EXPECT_EQ(regions[2].logical_offset, 32);

    auto segments = std::vector<ms::storage_segment_t>{};
    segments.emplace_back(ms::storage_segment_t{*required, 0, {}});
    const auto extents = ms::map_regions_to_storage(regions, segments);
    ASSERT_TRUE(extents.has_value());
    ASSERT_EQ(extents->size(), regions.size());
    for(size_t i = 0; i < regions.size(); ++i)
    {
        EXPECT_EQ(extents->at(i).size, regions[i].size);
        EXPECT_EQ(extents->at(i).region_index, i);
        EXPECT_EQ(extents->at(i).region_offset, 0);
        EXPECT_EQ(extents->at(i).segment_index, 0);
        EXPECT_EQ(extents->at(i).segment_offset, regions[i].logical_offset);
    }
}

TEST(kernel_replay_blit_copy, generic_mapping_can_cross_future_segment_boundary)
{
    auto regions = std::vector<ms::snapshot_region_t>{
        {30, nullptr, 0, ms::liveness_kind::tracked_allocation},
    };
    ASSERT_EQ(ms::assign_logical_offsets(regions), std::optional<size_t>{30});

    auto segments = std::vector<ms::storage_segment_t>{};
    segments.emplace_back(ms::storage_segment_t{16, 0, {}});
    segments.emplace_back(ms::storage_segment_t{14, 16, {}});

    const auto extents = ms::map_regions_to_storage(regions, segments);
    ASSERT_TRUE(extents.has_value());
    ASSERT_EQ(extents->size(), 2);
    EXPECT_EQ(extents->at(0).size, 16);
    EXPECT_EQ(extents->at(0).region_offset, 0);
    EXPECT_EQ(extents->at(0).segment_index, 0);
    EXPECT_EQ(extents->at(1).size, 14);
    EXPECT_EQ(extents->at(1).region_offset, 16);
    EXPECT_EQ(extents->at(1).segment_index, 1);
}

TEST(kernel_replay_blit_copy, generic_mapping_rejects_overlaps_and_gaps)
{
    auto regions = std::vector<ms::snapshot_region_t>{
        {30, nullptr, 0, ms::liveness_kind::tracked_allocation},
    };
    ASSERT_EQ(ms::assign_logical_offsets(regions), std::optional<size_t>{30});

    {
        auto overlapping = std::vector<ms::storage_segment_t>{};
        overlapping.emplace_back(ms::storage_segment_t{20, 0, {}});
        overlapping.emplace_back(ms::storage_segment_t{20, 10, {}});
        EXPECT_FALSE(ms::map_regions_to_storage(regions, overlapping).has_value());
    }
    {
        auto gapped = std::vector<ms::storage_segment_t>{};
        gapped.emplace_back(ms::storage_segment_t{10, 0, {}});
        gapped.emplace_back(ms::storage_segment_t{10, 20, {}});
        EXPECT_FALSE(ms::map_regions_to_storage(regions, gapped).has_value());
    }
}

TEST(kernel_replay_blit_copy, compact_tid_assignment_crosses_descriptor_boundaries)
{
    // Logical regions [0, 20), [20, 80), [80, 100), [100, 145), [145, 200)
    // model the discontiguous 20/60/20/45/55-item example. With 100 workers,
    // every thread must own logical item `tid` and `tid + 100`, even when those
    // items belong to different descriptors.
    constexpr auto stride     = std::uint64_t{100};
    constexpr auto item_count = std::array<std::uint64_t, 5>{20, 60, 20, 45, 55};
    constexpr auto total      = std::uint64_t{200};

    auto owner       = std::array<int, total>{};
    auto assignments = std::array<std::vector<std::pair<size_t, std::uint64_t>>, stride>{};
    owner.fill(-1);

    for(auto thread = std::uint64_t{0}; thread < stride; ++thread)
    {
        auto logical_base = std::uint64_t{0};
        for(auto region = size_t{0}; region < item_count.size(); ++region)
        {
            auto local = abi::first_local_item(thread, stride, logical_base);
            while(local < item_count[region])
            {
                const auto logical = logical_base + local;
                ASSERT_LT(logical, total);
                EXPECT_EQ(owner.at(logical), -1)
                    << "logical item " << logical << " assigned more than once";
                owner.at(logical) = static_cast<int>(thread);
                assignments.at(thread).emplace_back(region, local);

                if(item_count[region] - local <= stride) break;
                local += stride;
            }
            logical_base += item_count[region];
        }
    }

    for(auto logical = std::uint64_t{0}; logical < total; ++logical)
        EXPECT_EQ(owner.at(logical), static_cast<int>(logical % stride))
            << "logical item " << logical;

    for(auto thread = std::uint64_t{0}; thread < stride; ++thread)
        EXPECT_EQ(assignments.at(thread).size(), 2) << "thread " << thread;

    EXPECT_EQ(assignments.at(0), (std::vector<std::pair<size_t, std::uint64_t>>{{0, 0}, {3, 0}}));
    EXPECT_EQ(assignments.at(19),
              (std::vector<std::pair<size_t, std::uint64_t>>{{0, 19}, {3, 19}}));
    EXPECT_EQ(assignments.at(20), (std::vector<std::pair<size_t, std::uint64_t>>{{1, 0}, {3, 20}}));
    EXPECT_EQ(assignments.at(44),
              (std::vector<std::pair<size_t, std::uint64_t>>{{1, 24}, {3, 44}}));
    EXPECT_EQ(assignments.at(45), (std::vector<std::pair<size_t, std::uint64_t>>{{1, 25}, {4, 0}}));
    EXPECT_EQ(assignments.at(79),
              (std::vector<std::pair<size_t, std::uint64_t>>{{1, 59}, {4, 34}}));
    EXPECT_EQ(assignments.at(80), (std::vector<std::pair<size_t, std::uint64_t>>{{2, 0}, {4, 35}}));
    EXPECT_EQ(assignments.at(99),
              (std::vector<std::pair<size_t, std::uint64_t>>{{2, 19}, {4, 54}}));
}

TEST(kernel_replay_blit_copy, partial_region_uses_one_tail_item)
{
    EXPECT_EQ(abi::items_for_size(0), 0);
    EXPECT_EQ(abi::items_for_size(1), 1);
    EXPECT_EQ(abi::items_for_size(15), 1);
    EXPECT_EQ(abi::items_for_size(16), 1);
    EXPECT_EQ(abi::items_for_size(17), 2);
    EXPECT_EQ(abi::items_for_size(31), 2);
    EXPECT_EQ(abi::items_for_size(32), 2);
}
}  // namespace
