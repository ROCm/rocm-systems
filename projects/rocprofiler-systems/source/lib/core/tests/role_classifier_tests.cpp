// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/process_tree_builder.hpp"
#include "core/output/role_classifier.hpp"
#include "test_support/process_tree_builders.hpp"

#include <vector>

using rocprofsys::test_support::make_meta;

TEST(role_classifier, main_pid_gets_main_role)
{
    rocprofsys::output::process_node node{};
    node.meta = make_meta(100, -1);
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::main);
}

TEST(role_classifier, gpu_agent_gets_gpu_role)
{
    rocprofsys::output::process_node node{};
    node.meta = make_meta(200, 100, "", { 0 });
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::gpu);
}

TEST(role_classifier, parent_of_gpu_child_gets_engine_role)
{
    rocprofsys::output::process_node child{};
    child.meta = make_meta(200, 100, "", { 0 });
    rocprofsys::output::process_node parent{};
    parent.meta = make_meta(100, -1);
    parent.children.push_back(child);
    std::vector<rocprofsys::output::process_node> roots{ parent };
    rocprofsys::output::classify(roots, 999 /* main is elsewhere */);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::engine);
    ASSERT_TRUE(roots.front().children.front().role.has_value());
    EXPECT_EQ(*roots.front().children.front().role, rocprofsys::output::role_hint::gpu);
}

TEST(role_classifier, main_takes_precedence_over_gpu)
{
    // Main PID with GPU agents must still classify as *main*.
    rocprofsys::output::process_node node{};
    node.meta = make_meta(100, -1, "", { 0 });
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::main);
}

TEST(role_classifier, engine_effective_ids_union_of_descendants)
{
    rocprofsys::output::process_node gpu_a{};
    gpu_a.meta = make_meta(200, 100, "", { 0, 1 });
    rocprofsys::output::process_node gpu_b{};
    gpu_b.meta = make_meta(201, 100, "", { 2, 3 });
    rocprofsys::output::process_node engine{};
    engine.meta     = make_meta(100, -1);
    engine.children = { gpu_a, gpu_b };
    std::vector<rocprofsys::output::process_node> roots{ engine };
    rocprofsys::output::classify(roots, 999 /* main is elsewhere */);

    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::engine);
    EXPECT_EQ(roots.front().effective_gpu_ids, (std::vector<int>{ 0, 1, 2, 3 }));
}

TEST(role_classifier, main_with_own_gpus_keeps_main_role)
{
    rocprofsys::output::process_node node{};
    node.meta = make_meta(100, -1, "main_proc", { 0, 1, 2, 3 });
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::main);
    // Own gpu_ids preserved so the renderer can show the main
    // process's own GPU assignment alongside the *main* marker.
    EXPECT_EQ(roots.front().meta.gpu_ids, (std::vector<int>{ 0, 1, 2, 3 }));
}
