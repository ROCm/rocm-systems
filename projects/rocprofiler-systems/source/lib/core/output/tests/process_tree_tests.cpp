// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"
#include "core/output/process_tree.hpp"

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
using rocprofsys::output::artifact;
using rocprofsys::output::output_format;
using rocprofsys::output::process_metadata;
using rocprofsys::output::process_tree;

artifact
make_row(std::string path, pid_t pid, std::uint64_t size_bytes = 0)
{
    artifact a{};
    a.path       = std::move(path);
    a.pid        = pid;
    a.size_bytes = size_bytes;
    a.format     = output_format::text;
    return a;
}

process_metadata
make_meta(pid_t pid, pid_t ppid, std::string command = "")
{
    process_metadata m{};
    m.pid     = pid;
    m.ppid    = ppid;
    m.command = std::move(command);
    return m;
}
}  // namespace

TEST(process_tree, single_pid_becomes_single_root)
{
    std::vector<artifact>         rows{ make_row("a", 100) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    EXPECT_EQ(tree.roots().front().meta.pid, 100);
    EXPECT_EQ(tree.roots().front().rows.size(), 1u);
    EXPECT_TRUE(tree.roots().front().children.empty());
    EXPECT_TRUE(tree.diagnostics().missing_metadata_pids.empty());
    EXPECT_TRUE(tree.diagnostics().cyclic_ppid_pids.empty());
}

TEST(process_tree, parent_with_two_children_nests_under_parent)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("c1", 200),
                                make_row("c2", 201) };
    std::vector<process_metadata> processes{ make_meta(100, -1), make_meta(200, 100),
                                             make_meta(201, 100) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    ASSERT_EQ(tree.roots().front().children.size(), 2u);
    EXPECT_EQ(tree.roots().front().children[0].meta.pid, 200);
    EXPECT_EQ(tree.roots().front().children[1].meta.pid, 201);
}

TEST(process_tree, orphan_with_missing_ppid_attaches_at_root)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("orphan", 999) };
    std::vector<process_metadata> processes{ make_meta(100, -1),
                                             make_meta(999, 12345 /* unknown ppid */) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 2u);
    EXPECT_EQ(tree.roots()[0].meta.pid, 100);
    EXPECT_EQ(tree.roots()[1].meta.pid, 999);
}

TEST(process_tree, missing_metadata_pid_is_diagnosed)
{
    std::vector<artifact>         rows{ make_row("p", 100), make_row("ghost", 555) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    EXPECT_EQ(tree.diagnostics().missing_metadata_pids, (std::vector<pid_t>{ 555 }));
    ASSERT_EQ(tree.roots().size(), 2u);
}

TEST(process_tree, ppid_cycle_excludes_members_and_is_diagnosed)
{
    std::vector<artifact>         rows{ make_row("a", 300), make_row("b", 400) };
    std::vector<process_metadata> processes{ make_meta(300, 400), make_meta(400, 300) };
    process_tree                  tree{ rows, processes };
    EXPECT_TRUE(tree.roots().empty());
    EXPECT_EQ(tree.diagnostics().cyclic_ppid_pids, (std::vector<pid_t>{ 300, 400 }));
}

TEST(process_tree, deep_parent_chain_does_not_overflow_stack)
{
    constexpr int                 CHAIN_DEPTH = 1000;
    std::vector<process_metadata> processes;
    processes.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        processes.push_back(make_meta(pid, pid == 1 ? -1 : pid - 1));

    std::vector<artifact> rows;
    rows.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        rows.push_back(make_row(std::to_string(pid), pid));

    process_tree tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    EXPECT_EQ(tree.roots().front().meta.pid, 1);

    const auto* cur   = &tree.roots().front();
    int         depth = 1;
    while(!cur->children.empty())
    {
        ASSERT_EQ(cur->children.size(), 1u);
        cur = &cur->children.front();
        ++depth;
    }
    EXPECT_EQ(depth, CHAIN_DEPTH);
}

TEST(process_tree, rows_sorted_descending_by_size)
{
    std::vector<artifact>         rows{ make_row("small", 100, 1024),
                                make_row("large", 100, 1024ULL * 1024),
                                make_row("medium", 100, 4096) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    const auto& sorted_rows = tree.roots().front().rows;
    ASSERT_EQ(sorted_rows.size(), 3u);
    EXPECT_EQ(sorted_rows[0].path, "large");
    EXPECT_EQ(sorted_rows[1].path, "medium");
    EXPECT_EQ(sorted_rows[2].path, "small");
}

TEST(process_tree, size_rollup_computed_during_construction)
{
    std::vector<artifact> rows{ make_row("p", 100, 1000), make_row("c1", 200, 4096),
                                make_row("c2", 201, 2048) };
    std::vector<process_metadata> processes{ make_meta(100, -1), make_meta(200, 100),
                                             make_meta(201, 100) };
    process_tree                  tree{ rows, processes };

    ASSERT_EQ(tree.roots().size(), 1u);
    const auto& root = tree.roots().front();
    EXPECT_EQ(root.own_size_bytes, 1000u);
    EXPECT_EQ(root.cumulative_size_bytes, 1000u + 4096u + 2048u);
    ASSERT_EQ(root.children.size(), 2u);
    EXPECT_EQ(root.children[0].own_size_bytes, 4096u);
    EXPECT_EQ(root.children[0].cumulative_size_bytes, 4096u);
}

TEST(process_tree, size_rollup_treats_default_size_as_zero)
{
    std::vector<artifact>         rows{ make_row("p", 100) };
    std::vector<process_metadata> processes{ make_meta(100, -1) };
    process_tree                  tree{ rows, processes };
    ASSERT_EQ(tree.roots().size(), 1u);
    EXPECT_EQ(tree.roots().front().own_size_bytes, 0u);
    EXPECT_EQ(tree.roots().front().cumulative_size_bytes, 0u);
}
