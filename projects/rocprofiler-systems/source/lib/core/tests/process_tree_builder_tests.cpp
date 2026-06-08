// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/process_tree_builder.hpp"
#include "core/output_file_registry.hpp"
#include "test_support/process_tree_builders.hpp"

#include <sys/types.h>

#include <optional>
#include <string>
#include <vector>

using rocprofsys::test_support::make_meta;
using rocprofsys::test_support::make_row;

TEST(process_tree_builder, single_pid_becomes_single_root)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("a", 100) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 100);
    EXPECT_EQ(built.roots.front().rows.size(), 1u);
    EXPECT_TRUE(built.roots.front().children.empty());
    EXPECT_TRUE(built.diagnostics.missing_metadata_pids.empty());
}

TEST(process_tree_builder, parent_with_two_children_nests_under_parent)
{
    std::vector<rocprofsys::output_file> rows{ make_row("p", 100), make_row("c1", 200),
                                               make_row("c2", 201) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1),
                                                                 make_meta(200, 100),
                                                                 make_meta(201, 100) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 100);
    ASSERT_EQ(built.roots.front().children.size(), 2u);
    EXPECT_EQ(built.roots.front().children[0].meta.pid, 200);
    EXPECT_EQ(built.roots.front().children[1].meta.pid, 201);
}

TEST(process_tree_builder, orphan_with_missing_ppid_attaches_at_root)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("p", 100),
                                               make_row("orphan", 999) };
    std::vector<rocprofsys::output::process_metadata> processes{
        make_meta(100, -1), make_meta(999, 12345 /* unknown ppid */)
    };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 2u);
    EXPECT_EQ(built.roots[0].meta.pid, 100);
    EXPECT_EQ(built.roots[1].meta.pid, 999);
}

TEST(process_tree_builder, missing_metadata_pid_is_diagnosed)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("p", 100),
                                               make_row("ghost", 555) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    EXPECT_EQ(built.diagnostics.missing_metadata_pids, (std::vector<pid_t>{ 555 }));
    // Ghost still renders at root depth — missing metadata degrades
    // gracefully into a stub node rather than dropping the row.
    ASSERT_EQ(built.roots.size(), 2u);
}

TEST(process_tree_builder, deep_parent_chain_does_not_overflow_stack)
{
    // 1000-deep parent chain: pid 1000 → 999 → ... → 1 → orphan.
    // Recursive implementations blow the default 8 MiB stack here;
    // the iterative two-pass walk must succeed.
    constexpr int                                     CHAIN_DEPTH = 1000;
    std::vector<rocprofsys::output::process_metadata> processes;
    processes.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        processes.push_back(make_meta(pid, pid == 1 ? -1 : pid - 1));

    std::vector<rocprofsys::output_file> rows;
    rows.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        rows.push_back(make_row(std::to_string(pid), pid));

    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 1);

    const rocprofsys::output::process_node* cur   = &built.roots.front();
    int                                     depth = 1;
    while(!cur->children.empty())
    {
        ASSERT_EQ(cur->children.size(), 1u);
        cur = &cur->children.front();
        ++depth;
    }
    EXPECT_EQ(depth, CHAIN_DEPTH);
}

TEST(process_tree_builder, rows_sorted_descending_by_size)
{
    std::vector<rocprofsys::output_file> rows{
        make_row("small", 100, std::optional<std::uintmax_t>{ 1024 }),
        make_row("large", 100, std::optional<std::uintmax_t>{ 1024ULL * 1024 }),
        make_row("medium", 100, std::optional<std::uintmax_t>{ 4096 })
    };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    const auto& sorted_rows = built.roots.front().rows;
    ASSERT_EQ(sorted_rows.size(), 3u);
    EXPECT_EQ(sorted_rows[0].path, "large");
    EXPECT_EQ(sorted_rows[1].path, "medium");
    EXPECT_EQ(sorted_rows[2].path, "small");
}

TEST(process_tree_builder, compute_subtree_sizes_rolls_up_own_and_cumulative)
{
    std::vector<rocprofsys::output_file> rows{
        make_row("p", 100, std::optional<std::uintmax_t>{ 1000 }),
        make_row("c1", 200, std::optional<std::uintmax_t>{ 4096 }),
        make_row("c2", 201, std::optional<std::uintmax_t>{ 2048 })
    };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1),
                                                                 make_meta(200, 100),
                                                                 make_meta(201, 100) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    rocprofsys::output::compute_subtree_sizes(built.roots);

    ASSERT_EQ(built.roots.size(), 1u);
    const auto& root = built.roots.front();
    EXPECT_EQ(root.own_size_bytes, 1000u);
    EXPECT_EQ(root.cumulative_size_bytes, 1000u + 4096u + 2048u);
    ASSERT_EQ(root.children.size(), 2u);
    EXPECT_EQ(root.children[0].own_size_bytes, 4096u);
    EXPECT_EQ(root.children[0].cumulative_size_bytes, 4096u);
}

TEST(process_tree_builder, compute_subtree_sizes_ignores_unknown_sizes)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("p", 100) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    rocprofsys::output::compute_subtree_sizes(built.roots);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().own_size_bytes, 0u);
    EXPECT_EQ(built.roots.front().cumulative_size_bytes, 0u);
}

TEST(tree_visitor, for_each_post_visits_children_before_parent)
{
    rocprofsys::output::process_node n4{};
    n4.meta = make_meta(4, 2);
    rocprofsys::output::process_node n5{};
    n5.meta = make_meta(5, 2);
    rocprofsys::output::process_node n2{};
    n2.meta     = make_meta(2, 1);
    n2.children = { n4, n5 };
    rocprofsys::output::process_node n3{};
    n3.meta = make_meta(3, 1);
    rocprofsys::output::process_node n1{};
    n1.meta     = make_meta(1, -1);
    n1.children = { n2, n3 };

    std::vector<pid_t> visited;
    rocprofsys::output::for_each_post(
        n1, [&](rocprofsys::output::process_node& n) { visited.push_back(n.meta.pid); });
    EXPECT_EQ(visited, (std::vector<pid_t>{ 4, 5, 2, 3, 1 }));
}
