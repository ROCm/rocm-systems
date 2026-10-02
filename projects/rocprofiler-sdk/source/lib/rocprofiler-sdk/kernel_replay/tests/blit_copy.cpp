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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
namespace abi = rocprofiler::kernel_replay::blit::kernel_abi;

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
