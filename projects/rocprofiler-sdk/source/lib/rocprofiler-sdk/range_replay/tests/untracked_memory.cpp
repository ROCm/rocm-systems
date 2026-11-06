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

#include "lib/rocprofiler-sdk/range_replay/untracked_memory.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace rr = ::rocprofiler::range_replay;
namespace um = ::rocprofiler::range_replay::untracked_memory;

namespace
{
using type_t = um::pointer_facts_t::type_t;

constexpr uint64_t pinned_addr  = 0x7f0000001000;
constexpr uint64_t tracked_addr = 0x7f0000100000;

um::pointer_facts_t
facts_of(type_t type, bool owner_is_cpu = false, bool in_host_mapping = false)
{
    auto f                     = um::pointer_facts_t{};
    f.type                     = type;
    f.owner_is_cpu             = owner_is_cpu;
    f.in_writable_host_mapping = in_host_mapping;
    return f;
}

// Host-pinned at pinned_addr; everything else unknown to HSA.
um::pointer_facts_t
pinned_only(uint64_t address)
{
    return address == pinned_addr ? facts_of(type_t::hsa, true) : facts_of(type_t::unknown);
}

std::vector<uint8_t>
kernarg_with(const std::vector<uint64_t>& words, size_t trailing = 0)
{
    auto out = std::vector<uint8_t>(words.size() * sizeof(uint64_t) + trailing, 0);
    std::memcpy(out.data(), words.data(), words.size() * sizeof(uint64_t));
    return out;
}

rr::range_context_t
recording_context()
{
    auto ctx           = rr::range_context_t{};
    ctx.record         = rr::range_record_t{7};
    ctx.snapshot_taken = true;
    return ctx;
}

void
record_dispatch(rr::range_context_t& ctx, const std::vector<uint64_t>& words)
{
    auto d    = rr::recorded_dispatch_t{};
    d.kernarg = kernarg_with(words);
    ASSERT_TRUE(ctx.record.add_dispatch(std::move(d)));
}
}  // namespace

TEST(range_replay_untracked_memory, classifies_memory_the_snapshot_does_not_restore)
{
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::hsa, true)), um::kind_t::host_pinned);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::hsa, false)), um::kind_t::device_other);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::reserved)), um::kind_t::managed);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::vmem)), um::kind_t::virtual_mem);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::locked)), um::kind_t::locked_host);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::unknown, false, true)),
              um::kind_t::host_pageable);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::unknown)), std::nullopt)
        << "not memory a kernel can reach";
}

TEST(range_replay_untracked_memory, ignores_scalars_covered_memory_and_hidden_arguments)
{
    const auto covered = std::vector<um::address_span_t>{{tracked_addr, tracked_addr + 4096}};
    const auto facts   = [](uint64_t) { return facts_of(type_t::hsa, true); };

    // A small scalar, then a pointer into the snapshot, then the hidden block with a stale pointer.
    auto kernarg = kernarg_with({42, tracked_addr + 8}, um::implicit_arg_block_bytes);
    std::memcpy(kernarg.data() + 2 * sizeof(uint64_t), &pinned_addr, sizeof(pinned_addr));

    auto seen = std::unordered_set<uint64_t>{};
    EXPECT_FALSE(um::find_untracked(kernarg.data(), kernarg.size(), covered, facts, seen));
}

TEST(range_replay_untracked_memory, a_dispatch_handed_untracked_memory_declines_the_range)
{
    auto ctx = recording_context();
    record_dispatch(ctx, {42});
    rr::decline_untracked(ctx, pinned_only);
    EXPECT_TRUE(ctx.record.eligible());

    record_dispatch(ctx, {7, pinned_addr});
    rr::decline_untracked(ctx, pinned_only);
    EXPECT_FALSE(ctx.record.eligible());
    EXPECT_EQ(ctx.record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
}

TEST(range_replay_untracked_memory, nothing_is_checked_before_the_entry_snapshot)
{
    auto ctx           = recording_context();
    ctx.snapshot_taken = false;
    record_dispatch(ctx, {pinned_addr});
    rr::decline_untracked(ctx, pinned_only);
    EXPECT_TRUE(ctx.record.eligible());
}
