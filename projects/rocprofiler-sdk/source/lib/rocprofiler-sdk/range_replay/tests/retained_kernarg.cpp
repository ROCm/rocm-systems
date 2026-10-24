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

// The kernarg block a range's passes read their arguments from is kept for the agent's next range.
// Getting the bookkeeping wrong fails quietly: a block reused for a range it is too small for has
// its arguments written past its end, a block that is displaced without being freed leaks one
// kernarg allocation per range, and a block handed to two ranges at once lets one overwrite the
// arguments of the other's running pass. The blocks here are fake; the free function records what
// it was asked to free, so no kernarg pool is needed.

#include "lib/rocprofiler-sdk/range_replay/retained_kernarg.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace rr = ::rocprofiler::range_replay;

namespace
{
std::vector<void*>&
freed_blocks()
{
    static auto freed = std::vector<void*>{};
    return freed;
}

hsa_status_t
record_free(void* base)
{
    freed_blocks().emplace_back(base);
    return HSA_STATUS_SUCCESS;
}

rr::kernarg_block_t
make_block(uintptr_t base, size_t capacity)
{
    return rr::kernarg_block_t{reinterpret_cast<void*>(base), capacity, &record_free};
}

// The retained blocks are process-wide, so every test uses agents no other test touches.
rocprofiler_agent_id_t
make_agent(uint64_t handle)
{
    return rocprofiler_agent_id_t{.handle = handle};
}

class range_replay_retained_kernarg : public ::testing::Test
{
protected:
    void SetUp() override { freed_blocks().clear(); }
};
}  // namespace

// A retained block serves any range needing between half and all of its capacity. Below half it is
// freed and the range allocates its own, so one large range does not pin its block for the rest of
// the run.
TEST_F(range_replay_retained_kernarg, a_block_fits_requests_from_half_to_all_of_its_capacity)
{
    EXPECT_TRUE(rr::kernarg_block_fits(1024, 1024));
    EXPECT_TRUE(rr::kernarg_block_fits(1024, 512));

    EXPECT_FALSE(rr::kernarg_block_fits(1024, 1025)) << "too small: the pass would overrun it";
    EXPECT_FALSE(rr::kernarg_block_fits(1024, 511)) << "more than twice the request";
    EXPECT_FALSE(rr::kernarg_block_fits(0, 256)) << "an empty block serves nothing";
}

TEST_F(range_replay_retained_kernarg, nothing_is_retained_before_a_range_hands_a_block_back)
{
    const auto agent = make_agent(101);

    EXPECT_EQ(rr::retained_kernarg_bytes(agent), 0u);

    const auto block = rr::take_retained_kernarg_block(agent);
    EXPECT_EQ(block.base, nullptr);
    EXPECT_EQ(block.capacity, 0u);
}

// Taking a block hands it over: a second range on the same agent must not receive it while the
// first still owns it.
TEST_F(range_replay_retained_kernarg, a_retained_block_is_handed_over_exactly_once)
{
    const auto agent = make_agent(102);
    rr::retain_kernarg_block(agent, make_block(0x1000, 4096));
    EXPECT_EQ(rr::retained_kernarg_bytes(agent), 4096u);

    const auto first = rr::take_retained_kernarg_block(agent);
    EXPECT_EQ(first.base, reinterpret_cast<void*>(uintptr_t{0x1000}));
    EXPECT_EQ(first.capacity, 4096u);
    EXPECT_EQ(rr::retained_kernarg_bytes(agent), 0u);

    EXPECT_EQ(rr::take_retained_kernarg_block(agent).base, nullptr);
    EXPECT_TRUE(freed_blocks().empty()) << "handing a block over must not free it";
}

// An agent keeps one block. Retaining another frees the one it displaces rather than dropping it.
TEST_F(range_replay_retained_kernarg, retaining_a_second_block_frees_the_first)
{
    const auto agent = make_agent(103);
    rr::retain_kernarg_block(agent, make_block(0x2000, 1024));
    rr::retain_kernarg_block(agent, make_block(0x3000, 2048));

    ASSERT_EQ(freed_blocks().size(), 1u);
    EXPECT_EQ(freed_blocks().front(), reinterpret_cast<void*>(uintptr_t{0x2000}));

    EXPECT_EQ(rr::take_retained_kernarg_block(agent).base,
              reinterpret_cast<void*>(uintptr_t{0x3000}));
}

// Kernarg blocks come from each agent's own kernarg pool and are mapped for that agent, so a block
// retained for one agent is never handed to a range on another.
TEST_F(range_replay_retained_kernarg, each_agent_keeps_its_own_block)
{
    const auto lhs = make_agent(104);
    const auto rhs = make_agent(105);
    rr::retain_kernarg_block(lhs, make_block(0x4000, 512));
    rr::retain_kernarg_block(rhs, make_block(0x5000, 768));

    EXPECT_EQ(rr::take_retained_kernarg_block(rhs).base,
              reinterpret_cast<void*>(uintptr_t{0x5000}));
    EXPECT_EQ(rr::retained_kernarg_bytes(lhs), 512u);
    EXPECT_EQ(rr::take_retained_kernarg_block(lhs).base,
              reinterpret_cast<void*>(uintptr_t{0x4000}));
    EXPECT_TRUE(freed_blocks().empty());
}

TEST_F(range_replay_retained_kernarg, freeing_a_block_releases_it_once_and_empties_it)
{
    auto block = make_block(0x6000, 256);
    rr::free_kernarg_block(block);

    ASSERT_EQ(freed_blocks().size(), 1u);
    EXPECT_EQ(freed_blocks().front(), reinterpret_cast<void*>(uintptr_t{0x6000}));
    EXPECT_EQ(block.base, nullptr);
    EXPECT_EQ(block.capacity, 0u);

    rr::free_kernarg_block(block);
    EXPECT_EQ(freed_blocks().size(), 1u) << "an emptied block must not be freed again";
}
