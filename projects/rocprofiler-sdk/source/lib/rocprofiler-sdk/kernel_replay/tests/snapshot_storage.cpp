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

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace plan  = rocprofiler::kernel_replay::memory_snapshot::planning;
namespace store = rocprofiler::kernel_replay::memory_snapshot::storage;

struct mock_allocator_state
{
    bool   fail_allocate = false;
    size_t allocations   = 0;
    size_t releases      = 0;
    size_t last_size     = 0;
};

template <size_t Idx>
class mock_backing_allocator
{
public:
    using handle_type = std::vector<std::byte>;

    mock_backing_allocator(std::shared_ptr<mock_allocator_state> state, bool fail)
    : m_state{std::move(state)}
    {
        m_state->fail_allocate = fail;
    }

    store::allocation_result<handle_type> allocate(size_t size)
    {
        ++m_state->allocations;
        m_state->last_size = size;
        if(m_state->fail_allocate) return {.error = store::error_code::out_of_resources};
        try
        {
            return {.value = handle_type(size)};
        } catch(const std::bad_alloc&)
        {
            return {.error = store::error_code::out_of_resources};
        }
    }

    void release(handle_type) noexcept { ++m_state->releases; }

private:
    std::shared_ptr<mock_allocator_state> m_state;
};

using first_storage  = store::arena<plan::bounded_policy, mock_backing_allocator<0>>;
using second_storage = store::arena<plan::bounded_policy, mock_backing_allocator<1>>;
using third_storage  = store::arena<plan::bounded_policy, mock_backing_allocator<2>>;
using storage_chain =
    store::fallback_allocator<first_storage, second_storage, store::null_allocator>;

storage_chain
make_storage_chain(const std::shared_ptr<mock_allocator_state>& first_state,
                   const std::shared_ptr<mock_allocator_state>& second_state,
                   size_t                                       first_capacity,
                   size_t                                       second_capacity,
                   bool                                         first_fails  = false,
                   bool                                         second_fails = false)
{
    return storage_chain{
        first_storage{plan::bounded_policy{first_capacity}, first_state, first_fails},
        second_storage{plan::bounded_policy{second_capacity}, second_state, second_fails},
        store::null_allocator{},
    };
}

TEST(kernel_replay_snapshot_storage, variadic_fallback_is_non_sticky)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage      = make_storage_chain(first_state, second_state, 10, 20);
    auto requests     = std::vector<plan::request_t>{{6, 0, 100}, {7, 1, 200}, {4, 2, 300}};

    const auto result = plan::build(std::move(requests), storage, 1);
    ASSERT_TRUE(result.complete());
    ASSERT_EQ(result.placements.size(), 3);
    EXPECT_EQ(result.placements[0].storage_index, 0);
    EXPECT_EQ(result.placements[0].backing_offset, 0);
    EXPECT_EQ(result.placements[1].storage_index, 1);
    EXPECT_EQ(result.placements[1].backing_offset, 0);
    EXPECT_EQ(result.placements[2].storage_index, 0);
    EXPECT_EQ(result.placements[2].backing_offset, 6);
    EXPECT_EQ(storage.get<0>().used(), 10);
    EXPECT_EQ(storage.get<1>().used(), 7);
}

TEST(kernel_replay_snapshot_storage, null_allocator_reports_terminal_region)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage      = make_storage_chain(first_state, second_state, 4, 4);
    auto requests     = std::vector<plan::request_t>{{5, 17, 0}};

    const auto result = plan::build(std::move(requests), storage, 1);
    EXPECT_FALSE(result.complete());
    EXPECT_EQ(result.error.code, plan::error_code::no_backing_capacity);
    EXPECT_EQ(result.error.requested, 5);
    EXPECT_EQ(result.error.region_index, 17);
}

TEST(kernel_replay_snapshot_storage, planning_is_deterministic_and_aligned)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage      = make_storage_chain(first_state, second_state, 64, 0);
    auto requests     = std::vector<plan::request_t>{{1, 2, 300}, {17, 0, 100}, {1, 1, 200}};

    const auto result = plan::build(std::move(requests), storage, 16);
    ASSERT_TRUE(result.complete());
    ASSERT_EQ(result.placements.size(), 3);
    EXPECT_EQ(result.placements[0].region_index, 0);
    EXPECT_EQ(result.placements[0].backing_offset, 0);
    EXPECT_EQ(result.placements[1].region_index, 1);
    EXPECT_EQ(result.placements[1].backing_offset, 32);
    EXPECT_EQ(result.placements[2].region_index, 2);
    EXPECT_EQ(result.placements[2].backing_offset, 48);
    EXPECT_EQ(storage.get<0>().used(), 49);
}

TEST(kernel_replay_snapshot_storage, overflow_is_terminal)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage =
        make_storage_chain(first_state, second_state, std::numeric_limits<size_t>::max(), 1024);

    ASSERT_TRUE(storage.reserve(std::numeric_limits<size_t>::max() - 7, 1));
    const auto overflow = storage.reserve(16, 16);
    EXPECT_FALSE(overflow);
    EXPECT_EQ(overflow.error.code, plan::error_code::overflow);
    EXPECT_EQ(storage.get<1>().used(), 0);

    const auto zero_alignment = storage.reserve(1, 0);
    EXPECT_FALSE(zero_alignment);
    EXPECT_EQ(zero_alignment.error.code, plan::error_code::overflow);
    EXPECT_EQ(storage.get<1>().used(), 0);
}

TEST(kernel_replay_snapshot_storage, commits_only_used_tiers)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage      = make_storage_chain(first_state, second_state, 10, 20);
    auto requests     = std::vector<plan::request_t>{{6, 0, 0}, {4, 1, 1}};

    ASSERT_TRUE(plan::build(std::move(requests), storage, 1).complete());
    ASSERT_TRUE(storage.commit());
    EXPECT_EQ(first_state->allocations, 1);
    EXPECT_EQ(first_state->last_size, 10);
    EXPECT_EQ(second_state->allocations, 0);

    auto backing = storage.get<0>().release_backing();
    ASSERT_TRUE(backing.has_value());
    const auto late_reservation = storage.get<0>().reserve(1, 1);
    EXPECT_FALSE(late_reservation);
    EXPECT_EQ(late_reservation.error.code, plan::error_code::invalid_state);
}

TEST(kernel_replay_snapshot_storage, commit_failure_rolls_back_in_reverse)
{
    auto first_state  = std::make_shared<mock_allocator_state>();
    auto second_state = std::make_shared<mock_allocator_state>();
    auto storage      = make_storage_chain(first_state, second_state, 10, 20, false, true);
    auto requests     = std::vector<plan::request_t>{{6, 0, 0}, {7, 1, 1}};

    ASSERT_TRUE(plan::build(std::move(requests), storage, 1).complete());
    const auto commit = storage.commit();
    EXPECT_FALSE(commit);
    EXPECT_EQ(commit.storage_index, 1);
    EXPECT_EQ(first_state->allocations, 1);
    EXPECT_EQ(first_state->releases, 1);
    EXPECT_EQ(second_state->allocations, 1);
    EXPECT_EQ(storage.get<0>().used(), 0);
    EXPECT_EQ(storage.get<1>().used(), 0);
}

TEST(kernel_replay_snapshot_storage, supports_three_backing_tiers)
{
    auto first_state       = std::make_shared<mock_allocator_state>();
    auto second_state      = std::make_shared<mock_allocator_state>();
    auto third_state       = std::make_shared<mock_allocator_state>();
    using three_tier_chain = store::
        fallback_allocator<first_storage, second_storage, third_storage, store::null_allocator>;
    auto storage = three_tier_chain{
        first_storage{plan::bounded_policy{4}, first_state, false},
        second_storage{plan::bounded_policy{4}, second_state, false},
        third_storage{plan::bounded_policy{8}, third_state, false},
        store::null_allocator{},
    };
    auto requests = std::vector<plan::request_t>{{4, 0, 0}, {4, 1, 1}, {5, 2, 2}};

    const auto result = plan::build(std::move(requests), storage, 1);
    ASSERT_TRUE(result.complete());
    ASSERT_EQ(result.placements.size(), 3);
    EXPECT_EQ(result.placements[0].storage_index, 0);
    EXPECT_EQ(result.placements[1].storage_index, 1);
    EXPECT_EQ(result.placements[2].storage_index, 2);
    ASSERT_TRUE(storage.commit());
    EXPECT_EQ(first_state->last_size, 4);
    EXPECT_EQ(second_state->last_size, 4);
    EXPECT_EQ(third_state->last_size, 5);
}

struct configured_backing_allocator
{
    using handle_type = size_t;

    configured_backing_allocator(std::shared_ptr<mock_allocator_state> state_arg,
                                 std::unique_ptr<size_t>               multiplier_arg,
                                 std::string                           name_arg)
    : state{std::move(state_arg)}
    , multiplier{std::move(multiplier_arg)}
    , name{std::move(name_arg)}
    {}

    configured_backing_allocator(configured_backing_allocator&&) noexcept = default;
    configured_backing_allocator& operator=(configured_backing_allocator&&) = delete;

    store::allocation_result<handle_type> allocate(size_t size)
    {
        ++state->allocations;
        state->last_size = size * *multiplier;
        return {.value = size};
    }

    void release(handle_type) noexcept { ++state->releases; }

    std::shared_ptr<mock_allocator_state> state;
    std::unique_ptr<size_t>               multiplier;
    std::string                           name;
};

TEST(kernel_replay_snapshot_storage, forwards_arbitrary_move_only_backing_arguments)
{
    auto state               = std::make_shared<mock_allocator_state>();
    using configured_storage = store::arena<plan::bounded_policy, configured_backing_allocator>;
    auto storage             = configured_storage{
        plan::bounded_policy{16}, state, std::make_unique<size_t>(3), std::string{"mock"}};

    ASSERT_TRUE(storage.reserve(8, 1));
    ASSERT_TRUE(storage.commit());
    EXPECT_EQ(state->allocations, 1);
    EXPECT_EQ(state->last_size, 24);

    auto moved = std::move(storage);
    moved.reset();
    EXPECT_EQ(state->releases, 1);
}
}  // namespace
