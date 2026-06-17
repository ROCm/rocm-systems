// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/process_tree_builder.hpp"
#include "test_support/process_tree_builders.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

using rocprofsys::test_support::make_meta;
using rocprofsys::test_support::make_row;

TEST(small_process_collapse, collapses_small_process_siblings)
{
    auto small_process = [](pid_t pid) {
        rocprofsys::output::process_node n{};
        n.meta = make_meta(pid, 100);
        n.rows.push_back(make_row("tiny", pid, std::optional<std::uintmax_t>{ 4096 }));
        return n;
    };
    rocprofsys::output::process_node parent{};
    parent.meta     = make_meta(100, -1);
    parent.children = { small_process(200), small_process(201), small_process(202) };
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_small_processes(std::move(roots));
    ASSERT_EQ(collapsed.size(), 1u);
    ASSERT_EQ(collapsed.front().children.size(), 1u);
    const auto& rangenode = collapsed.front().children.front();
    ASSERT_TRUE(rangenode.collapsed.has_value());
    EXPECT_EQ(rangenode.collapsed->min_pid, 200);
    EXPECT_EQ(rangenode.collapsed->max_pid, 202);
    EXPECT_EQ(rangenode.collapsed->count, 3u);
}

TEST(small_process_collapse, single_small_process_is_not_collapsed)
{
    rocprofsys::output::process_node parent{};
    parent.meta = make_meta(100, -1);
    rocprofsys::output::process_node child{};
    child.meta = make_meta(200, 100);
    child.rows.push_back(make_row("tiny", 200, std::optional<std::uintmax_t>{ 4096 }));
    parent.children.push_back(child);
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_small_processes(std::move(roots));
    ASSERT_EQ(collapsed.front().children.size(), 1u);
    EXPECT_FALSE(collapsed.front().children.front().collapsed.has_value());
}

TEST(small_process_collapse, gpu_sibling_never_collapsed)
{
    auto small_process = [](pid_t pid) {
        rocprofsys::output::process_node n{};
        n.meta = make_meta(pid, 100);
        n.rows.push_back(make_row("tiny", pid, std::optional<std::uintmax_t>{ 4096 }));
        return n;
    };
    rocprofsys::output::process_node gpu_node{};
    gpu_node.meta = make_meta(300, 100, "", { 0 });
    gpu_node.rows.push_back(
        make_row("big", 300, std::optional<std::uintmax_t>{ 100ULL * 1024 * 1024 }));
    rocprofsys::output::process_node parent{};
    parent.meta     = make_meta(100, -1);
    parent.children = { small_process(200), small_process(201), gpu_node };
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_small_processes(std::move(roots));
    ASSERT_EQ(collapsed.front().children.size(), 2u);
    // First child: the GPU node (kept as-is, original order preserved
    // for larger processes).
    EXPECT_EQ(collapsed.front().children[0].meta.pid, 300);
    EXPECT_FALSE(collapsed.front().children[0].collapsed.has_value());
    // Second child: the collapsed small-process range.
    ASSERT_TRUE(collapsed.front().children[1].collapsed.has_value());
}
