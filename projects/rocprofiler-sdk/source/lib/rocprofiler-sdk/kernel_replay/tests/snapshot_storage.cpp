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

#include "lib/rocprofiler-sdk/kernel_replay/snapshot-plan.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/snapshot-storage.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
namespace plan  = rocprofiler::kernel_replay::memory_snapshot::planning;
namespace store = rocprofiler::kernel_replay::memory_snapshot::storage;

struct mock_state
{
    bool                                  fail        = false;
    size_t                                allocations = 0;
    size_t                                last_size   = 0;
    std::weak_ptr<std::vector<std::byte>> last        = {};
};

// Stands in for an HSA pool, mmap, hipFile, ... Its handles release their bytes when destroyed.
struct mock_backing
{
    using handle_type = std::shared_ptr<std::vector<std::byte>>;

    std::shared_ptr<mock_state> state = std::make_shared<mock_state>();

    std::optional<handle_type> allocate(size_t size) const
    {
        ++state->allocations;
        state->last_size = size;
        if(state->fail) return std::nullopt;
        auto bytes  = std::make_shared<std::vector<std::byte>>(size);
        state->last = bytes;
        return bytes;
    }
};

using bump_t    = store::bump_suballocator<mock_backing>;
using whole_t   = store::arena<plan::whole_region, bump_t>;
using chunked_t = store::arena<plan::prefix_chunking, bump_t>;

bump_t
bump(size_t capacity, std::shared_ptr<mock_state> state = std::make_shared<mock_state>())
{
    return bump_t{capacity, 4, mock_backing{std::move(state)}};
}

struct range
{
    size_t offset = 0;
    size_t size   = 0;
};

// First fit over fixed free ranges, standing in for a reused arena whose free space is fragmented.
struct first_fit_suballocator
{
    using handle_type = mock_backing::handle_type;

    std::vector<range>         holes     = {};
    size_t                     extent    = 0;
    mock_backing               backing   = {};
    std::vector<range>         remaining = holes;
    std::optional<handle_type> handle    = {};

    size_t available() const
    {
        size_t largest = 0;
        for(const auto& hole : remaining)
            largest = std::max(largest, hole.size);
        return largest;
    }

    size_t reserve(size_t size)
    {
        auto       hole   = std::find_if(remaining.begin(),
                                 remaining.end(),
                                 [size](const range& candidate) { return candidate.size >= size; });
        const auto offset = hole->offset;
        hole->offset += size;
        hole->size -= size;
        return offset;
    }

    bool commit()
    {
        handle = backing.allocate(extent);
        return handle.has_value();
    }

    void reset()
    {
        handle.reset();
        remaining = holes;
    }

    size_t                     backing_size() const { return extent; }
    std::optional<handle_type> release_backing() { return std::exchange(handle, std::nullopt); }
};

// Takes nothing, so every request moves on to the next tier.
struct decline
{
    size_t admit(const plan::request_t&, size_t) const { return 0; }
};

// Discovery would record which regions must stay device-local, such as kernargs.
struct is_device_only
{
    const std::vector<bool>* device_only = nullptr;

    bool operator()(const plan::request_t& request) const
    {
        return device_only->at(request.region_index);
    }
};

// Routes each request to admission A when the predicate holds and to B otherwise: a segregator
// keyed on memory properties instead of size.
template <typename Predicate, typename A, typename B>
struct a_or_b
{
    Predicate predicate = {};
    A         a         = {};
    B         b         = {};

    size_t admit(const plan::request_t& request, size_t available) const
    {
        return predicate(request) ? a.admit(request, available) : b.admit(request, available);
    }
};

// Move-only, with whatever constructor state a real allocator (hipFile, mmap, ...) needs.
struct configured_backing
{
    using handle_type = size_t;

    std::unique_ptr<size_t> multiplier;
    std::string             name;

    std::optional<handle_type> allocate(size_t size) const { return size * *multiplier; }
};

TEST(kernel_replay_snapshot_storage, whole_regions_spill_without_sticking)
{
    auto storage = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump(6)},
        whole_t{{}, bump(16)},
        store::null_tier{},
    };

    const auto placements = plan::plan_and_commit(storage, {{0, 0, 10}, {1, 0, 3}, {2, 0, 2}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 3);
    EXPECT_EQ(placements->at(0).tier, 1);
    EXPECT_EQ(placements->at(1).tier, 0);
    EXPECT_EQ(placements->at(1).backing_offset, 0);
    EXPECT_EQ(placements->at(2).tier, 0);
    EXPECT_EQ(placements->at(2).backing_offset, 4);
}

TEST(kernel_replay_snapshot_storage, prefix_chunking_splits_on_a_granule)
{
    auto storage = store::fallback<chunked_t, whole_t, store::null_tier>{
        chunked_t{{4}, bump(10)},
        whole_t{{}, bump(16)},
        store::null_tier{},
    };

    const auto placements = plan::plan_and_commit(storage, {{0, 0, 16}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 2);
    EXPECT_EQ(placements->at(0).size, 8);
    EXPECT_EQ(placements->at(0).tier, 0);
    EXPECT_EQ(placements->at(1).region_offset, 8);
    EXPECT_EQ(placements->at(1).size, 8);
    EXPECT_EQ(placements->at(1).tier, 1);
}

TEST(kernel_replay_snapshot_storage, split_requests_interleave_tiers_by_priority)
{
    auto storage = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump(8)},
        whole_t{{}, bump(16)},
        store::null_tier{},
    };

    // One region split before planning: hot [0, 4) and [12, 16) at priority 0, cold [4, 12) at 1.
    const auto placements =
        plan::plan_and_commit(storage, {{0, 0, 4, 0}, {0, 4, 8, 1}, {0, 12, 4, 0}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 3);
    EXPECT_EQ(placements->at(0).region_offset, 0);
    EXPECT_EQ(placements->at(0).tier, 0);
    EXPECT_EQ(placements->at(1).region_offset, 12);
    EXPECT_EQ(placements->at(1).tier, 0);
    EXPECT_EQ(placements->at(1).backing_offset, 4);
    EXPECT_EQ(placements->at(2).region_offset, 4);
    EXPECT_EQ(placements->at(2).tier, 1);
}

TEST(kernel_replay_snapshot_storage, fragmented_tier_takes_a_request_in_pieces)
{
    using fragmented_t = store::arena<plan::prefix_chunking, first_fit_suballocator>;
    auto state         = std::make_shared<mock_state>();
    auto storage       = store::fallback<fragmented_t, whole_t, store::null_tier>{
        fragmented_t{{1}, {{{0, 4}, {8, 8}}, 16, mock_backing{state}}},
        whole_t{{}, bump(16)},
        store::null_tier{},
    };

    const auto placements = plan::plan_and_commit(storage, {{0, 0, 10}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 2);
    EXPECT_EQ(placements->at(0).size, 8);
    EXPECT_EQ(placements->at(0).backing_offset, 8);
    EXPECT_EQ(placements->at(1).size, 2);
    EXPECT_EQ(placements->at(1).backing_offset, 0);
    EXPECT_EQ(placements->at(1).tier, 0);
    EXPECT_EQ(state->last_size, 16);
}

TEST(kernel_replay_snapshot_storage, memory_properties_select_admission)
{
    using gpu_admission  = a_or_b<is_device_only, plan::whole_region, plan::prefix_chunking>;
    using host_admission = a_or_b<is_device_only, decline, plan::whole_region>;
    using gpu_t          = store::arena<gpu_admission, bump_t>;
    using host_t         = store::arena<host_admission, bump_t>;

    const auto device_only  = std::vector<bool>{false, true};
    const auto make_storage = [&device_only](size_t gpu_capacity) {
        return store::fallback<gpu_t, host_t, store::null_tier>{
            gpu_t{{{&device_only}, {}, {2}}, bump(gpu_capacity)},
            host_t{{{&device_only}, {}, {}}, bump(16)},
            store::null_tier{},
        };
    };

    // The device-only region gets priority 0 so it claims the GPU before ordinary memory fills it.
    auto       storage    = make_storage(8);
    const auto placements = plan::plan_and_commit(storage, {{0, 0, 6, 1}, {1, 0, 4, 0}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 3);
    EXPECT_EQ(placements->at(0).region_index, 1);
    EXPECT_EQ(placements->at(0).size, 4);
    EXPECT_EQ(placements->at(0).tier, 0);
    EXPECT_EQ(placements->at(1).region_index, 0);
    EXPECT_EQ(placements->at(1).tier, 0);
    EXPECT_EQ(placements->at(2).region_offset, 4);
    EXPECT_EQ(placements->at(2).tier, 1);

    auto tight = make_storage(4);
    EXPECT_FALSE(plan::plan_and_commit(tight, {{1, 0, 8}}));
    const auto ordinary = plan::plan_and_commit(tight, {{0, 0, 8}});
    ASSERT_TRUE(ordinary);
    ASSERT_EQ(ordinary->size(), 2);
    EXPECT_EQ(ordinary->at(1).tier, 1);
}

TEST(kernel_replay_snapshot_storage, failed_commit_disables_the_tier_and_replans)
{
    auto refusing  = std::make_shared<mock_state>();
    refusing->fail = true;
    auto storage   = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump(16, refusing)},
        whole_t{{}, bump(16)},
        store::null_tier{},
    };

    const auto placements = plan::plan_and_commit(storage, {{0, 0, 6}, {1, 0, 4}});
    ASSERT_TRUE(placements);
    EXPECT_EQ(refusing->allocations, 1);
    ASSERT_EQ(placements->size(), 2);
    EXPECT_EQ(placements->at(0).tier, 1);
    EXPECT_EQ(placements->at(1).tier, 1);
}

TEST(kernel_replay_snapshot_storage, exhausted_tiers_fail_planning_without_allocating)
{
    auto gpu     = std::make_shared<mock_state>();
    auto host    = std::make_shared<mock_state>();
    auto storage = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump(4, gpu)},
        whole_t{{}, bump(4, host)},
        store::null_tier{},
    };

    EXPECT_FALSE(plan::plan_and_commit(storage, {{0, 0, 5}}));
    EXPECT_EQ(gpu->allocations, 0);
    EXPECT_EQ(host->allocations, 0);
}

TEST(kernel_replay_snapshot_storage, commit_failure_releases_committed_tiers)
{
    auto first   = std::make_shared<mock_state>();
    auto second  = std::make_shared<mock_state>();
    second->fail = true;
    auto storage = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump(8, first)},
        whole_t{{}, bump(8, second)},
        store::null_tier{},
    };

    ASSERT_TRUE(plan::plan(storage, {{0, 0, 8}, {1, 0, 4}}));
    const auto failed = storage.commit();
    ASSERT_TRUE(failed);
    EXPECT_EQ(*failed, 1);
    EXPECT_EQ(first->allocations, 1);
    EXPECT_TRUE(first->last.expired());
}

TEST(kernel_replay_snapshot_storage, released_backings_outlive_the_storage)
{
    auto first   = std::make_shared<mock_state>();
    auto second  = std::make_shared<mock_state>();
    auto handles = std::vector<std::pair<size_t, mock_backing::handle_type>>{};
    {
        auto storage = store::fallback<whole_t, whole_t, store::null_tier>{
            whole_t{{}, bump(10, first)},
            whole_t{{}, bump(10, second)},
            store::null_tier{},
        };
        ASSERT_TRUE(plan::plan_and_commit(storage, {{0, 0, 6}}));
        storage.release_backings([&](size_t tier, size_t size, mock_backing::handle_type handle) {
            EXPECT_EQ(handle->size(), size);
            handles.emplace_back(tier, std::move(handle));
        });
    }

    EXPECT_EQ(second->allocations, 0);
    ASSERT_EQ(handles.size(), 1);
    EXPECT_EQ(handles.front().first, 0);
    EXPECT_FALSE(first->last.expired());
}

TEST(kernel_replay_snapshot_storage, backing_allocators_can_be_move_only_with_arbitrary_state)
{
    using configured_t =
        store::arena<plan::whole_region, store::bump_suballocator<configured_backing>>;
    auto storage = store::fallback<configured_t, store::null_tier>{
        configured_t{{}, {16, 1, configured_backing{std::make_unique<size_t>(3), "mock"}}},
        store::null_tier{},
    };

    ASSERT_TRUE(plan::plan_and_commit(storage, {{0, 0, 8}}));
    const auto& handle = std::get<0>(storage.tiers).suballocator.handle;
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(*handle, 24);
}

TEST(kernel_replay_snapshot_storage, near_max_capacity_never_wraps)
{
    constexpr auto largest = std::numeric_limits<size_t>::max();
    auto           storage = store::fallback<whole_t, whole_t, store::null_tier>{
        whole_t{{}, bump_t{largest, 16, mock_backing{}}},
        whole_t{{}, bump(1024)},
        store::null_tier{},
    };

    // The second request's aligned offset would pass SIZE_MAX, so the first tier declines it.
    const auto placements = plan::plan(storage, {{0, 0, largest - 7}, {1, 0, 16}});
    ASSERT_TRUE(placements);
    ASSERT_EQ(placements->size(), 2);
    EXPECT_EQ(placements->at(0).tier, 0);
    EXPECT_EQ(placements->at(1).tier, 1);
}
}  // namespace
