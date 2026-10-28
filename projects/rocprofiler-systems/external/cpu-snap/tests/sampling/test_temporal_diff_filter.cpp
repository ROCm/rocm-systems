// Unit tests for temporal_diff_filter, the per-tid divergence detector
// between kernel_drain and frame_trie. Sample `frames` are innermost-first
// (frames[0] is the leaf); the filter walks outermost-first.

#include <cstddef>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <sched.h>
#include <span>
#include <unordered_set>
#include <vector>

#include "cpu_snap/sampling/frame_trie.hpp"
#include "cpu_snap/sampling/temporal_diff_filter.hpp"

using cpu_snap::sampling::internal::frame_trie;
using cpu_snap::sampling::internal::temporal_diff_filter;

namespace
{

constexpr pid_t k_tid_a = 1001;
constexpr pid_t k_tid_b = 1002;

std::vector<frame_trie::node_view>
copy_new_nodes(const temporal_diff_filter::diff_result& r)
{
    return { r.new_nodes.begin(), r.new_nodes.end() };
}

}  // namespace

TEST(TemporalDiffFilter, FirstSampleEmitsFullPathRootToLeaf)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    // frames innermost-first: A=leaf, B=middle, C=outermost.
    const std::array<std::uint64_t, 3> frames = { 0xA, 0xB, 0xC };

    const auto r        = filter.diff(k_tid_a, frames);
    const auto new_list = copy_new_nodes(r);

    ASSERT_EQ(new_list.size(), 3U);

    // Emission order is root-ward first: C (parent=root), then B (parent=C),
    // then A (parent=B). reader rebuilds the trie by forward replay.
    EXPECT_EQ(new_list[0].parent_id, frame_trie::k_root);
    EXPECT_EQ(new_list[0].ip, 0xCU);
    EXPECT_EQ(new_list[1].parent_id, new_list[0].id);
    EXPECT_EQ(new_list[1].ip, 0xBU);
    EXPECT_EQ(new_list[2].parent_id, new_list[1].id);
    EXPECT_EQ(new_list[2].ip, 0xAU);

    EXPECT_EQ(r.leaf_id, new_list[2].id);
    EXPECT_EQ(filter.current_leaf_for_tid(k_tid_a), r.leaf_id);
    EXPECT_EQ(filter.tracked_tids(), 1U);
    EXPECT_EQ(trie.size(), 3U);
}

TEST(TemporalDiffFilter, IdenticalFollowupSampleMakesNoTrieCalls)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 3> frames = { 0xA, 0xB, 0xC };

    const auto first      = filter.diff(k_tid_a, frames);
    const auto first_leaf = first.leaf_id;
    const auto trie_size  = trie.size();

    const auto second = filter.diff(k_tid_a, frames);
    EXPECT_TRUE(second.new_nodes.empty());
    EXPECT_EQ(second.leaf_id, first_leaf);
    EXPECT_EQ(trie.size(), trie_size);  // no new nodes
}

TEST(TemporalDiffFilter, LeafOnlyChangeEmitsSingleNewNode)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 3> frames1 = { 0xA, 0xB, 0xC };
    const std::array<std::uint64_t, 3> frames2 = { 0xA2, 0xB, 0xC };  // new leaf

    const auto first  = filter.diff(k_tid_a, frames1);
    const auto second = filter.diff(k_tid_a, frames2);

    ASSERT_EQ(second.new_nodes.size(), 1U);
    EXPECT_EQ(second.new_nodes[0].ip, 0xA2U);
    EXPECT_EQ(second.new_nodes[0].parent_id, first.new_nodes[1].id);  // child of B
    EXPECT_EQ(second.leaf_id, second.new_nodes[0].id);
    EXPECT_NE(second.leaf_id, first.leaf_id);
    EXPECT_EQ(trie.size(), 4U);  // 3 from first sample + 1 new leaf
}

TEST(TemporalDiffFilter, PushDeeperFrameAtLeafEmitsOneNewNode)
{
    // Prior stack: outermost C, middle B, leaf A. frames=[A,B,C].
    // New stack pushes X as the new leaf on top of A.
    //   Conceptually: C -> B -> A -> X (X is deepest call).
    // Innermost-first new frames: [X, A, B, C].
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    constexpr std::uint64_t k_pushed_leaf_ip = 0xDD;

    const std::array<std::uint64_t, 3> frames1 = { 0xA, 0xB, 0xC };
    const std::array<std::uint64_t, 4> frames2 = { k_pushed_leaf_ip, 0xA, 0xB, 0xC };

    (void) filter.diff(k_tid_a, frames1);
    const auto second = filter.diff(k_tid_a, frames2);

    ASSERT_EQ(second.new_nodes.size(), 1U);
    EXPECT_EQ(second.new_nodes[0].ip, k_pushed_leaf_ip);
    EXPECT_EQ(second.leaf_id, second.new_nodes[0].id);
    EXPECT_EQ(trie.size(), 4U);
}

TEST(TemporalDiffFilter, PopShallowerStackReusesExistingPrefix)
{
    // Prior frames=[A,B,C] populates root->C->B->A.
    // Pop the innermost frame A; new frames=[B,C].
    // Leaf becomes the node for B (already interned), zero new nodes.
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 3> frames1 = { 0xA, 0xB, 0xC };
    const std::array<std::uint64_t, 2> frames2 = { 0xB, 0xC };

    const auto first  = filter.diff(k_tid_a, frames1);
    const auto second = filter.diff(k_tid_a, frames2);

    EXPECT_TRUE(second.new_nodes.empty());
    EXPECT_EQ(second.leaf_id, first.new_nodes[1].id);  // node for B
    EXPECT_NE(second.leaf_id, first.leaf_id);
    EXPECT_EQ(trie.size(), 3U);
}

TEST(TemporalDiffFilter, InterleavedTidsKeepIndependentState)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 3> a_frames = { 0xA1, 0xA2, 0xA3 };
    const std::array<std::uint64_t, 3> b_frames = { 0xB1, 0xB2, 0xB3 };

    const auto a1 = filter.diff(k_tid_a, a_frames);
    const auto b1 = filter.diff(k_tid_b, b_frames);
    EXPECT_EQ(a1.new_nodes.size(), 3U);
    EXPECT_EQ(b1.new_nodes.size(), 3U);
    EXPECT_NE(a1.leaf_id, b1.leaf_id);

    // Repeat each tid's same sample - both should be zero-new-node.
    const auto a2 = filter.diff(k_tid_a, a_frames);
    const auto b2 = filter.diff(k_tid_b, b_frames);
    EXPECT_TRUE(a2.new_nodes.empty());
    EXPECT_TRUE(b2.new_nodes.empty());
    EXPECT_EQ(a2.leaf_id, a1.leaf_id);
    EXPECT_EQ(b2.leaf_id, b1.leaf_id);

    EXPECT_EQ(filter.tracked_tids(), 2U);
    EXPECT_EQ(filter.current_leaf_for_tid(k_tid_a), a1.leaf_id);
    EXPECT_EQ(filter.current_leaf_for_tid(k_tid_b), b1.leaf_id);
}

TEST(TemporalDiffFilter, ForgetTidDropsStateAndNextDiffWalksFullPath)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 2> frames = { 0xAA, 0xBB };

    const auto first = filter.diff(k_tid_a, frames);
    EXPECT_EQ(first.new_nodes.size(), 2U);

    filter.forget_tid(k_tid_a);
    EXPECT_EQ(filter.tracked_tids(), 0U);
    EXPECT_EQ(filter.current_leaf_for_tid(k_tid_a), frame_trie::k_root);

    // Trie already has the nodes interned: walking the full path again
    // must reuse them, so new_nodes is empty but state is repopulated.
    const auto second = filter.diff(k_tid_a, frames);
    EXPECT_TRUE(second.new_nodes.empty());
    EXPECT_EQ(second.leaf_id, first.leaf_id);
    EXPECT_EQ(filter.tracked_tids(), 1U);
}

TEST(TemporalDiffFilter, EmptyFramesReturnsRootWithoutTouchingState)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    const std::array<std::uint64_t, 2> frames = { 0xAA, 0xBB };
    const auto                         first  = filter.diff(k_tid_a, frames);

    const auto empty = filter.diff(k_tid_a, std::span<const std::uint64_t>{});
    EXPECT_EQ(empty.leaf_id, frame_trie::k_root);
    EXPECT_TRUE(empty.new_nodes.empty());

    // Prior state must be intact: a repeat of the original frames stays
    // in steady-state with zero new nodes.
    const auto repeat = filter.diff(k_tid_a, frames);
    EXPECT_TRUE(repeat.new_nodes.empty());
    EXPECT_EQ(repeat.leaf_id, first.leaf_id);
}

TEST(TemporalDiffFilter, SteadyStateRepeatsStopGrowingTrie)
{
    // Microbench-ish: hammer the same 16-frame stack 100k times and
    // assert the trie stops growing after the first call.
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    constexpr std::size_t k_depth      = 16;
    constexpr std::size_t k_iterations = 100'000;

    std::array<std::uint64_t, k_depth> frames{};
    for(std::size_t i = 0; i < k_depth; ++i) {
        frames[i] = 0xF00D'0000ULL + i;
}

    const auto first = filter.diff(k_tid_a, frames);
    EXPECT_EQ(first.new_nodes.size(), k_depth);
    const auto stable_size = trie.size();
    EXPECT_EQ(stable_size, k_depth);

    for(std::size_t i = 0; i < k_iterations; ++i)
    {
        const auto r = filter.diff(k_tid_a, frames);
        ASSERT_TRUE(r.new_nodes.empty());
        ASSERT_EQ(r.leaf_id, first.leaf_id);
    }
    EXPECT_EQ(trie.size(), stable_size);
}

TEST(TemporalDiffFilter, StressManyTidsWithPopPushAroundCommonBase)
{
    frame_trie           trie;
    temporal_diff_filter filter{ trie };

    constexpr std::size_t   k_num_tids        = 100;
    constexpr std::size_t   k_samples_per_tid = 100;
    constexpr std::size_t   k_base_depth      = 8;
    constexpr std::uint64_t k_base_ip_seed    = 0x1000'0000ULL;

    // Shared outermost prefix of length k_base_depth, then a per-tid
    // tail that flips between two leaves to drive pop/push churn.
    std::vector<std::uint64_t> base(k_base_depth);
    for(std::size_t i = 0; i < k_base_depth; ++i) {
        base[i] = k_base_ip_seed + i;
}

    std::mt19937                    rng{ 0xC0FFEE };
    std::uniform_int_distribution<> coin{ 0, 1 };

    std::unordered_set<std::uint64_t> expected_unique_paths;

    for(std::size_t s = 0; s < k_samples_per_tid; ++s)
    {
        for(std::size_t t = 0; t < k_num_tids; ++t)
        {
            const auto          tid = static_cast<pid_t>(2000 + t);
            const std::uint64_t leaf_choice =
                coin(rng) == 0 ? 0xDEAD'0000ULL + t : 0xBEEF'0000ULL + t;
            std::vector<std::uint64_t> frames;
            frames.reserve(k_base_depth + 1);
            frames.push_back(leaf_choice);
            for(std::size_t i = 0; i < k_base_depth; ++i)
            {
                // base is outermost-first in this test's mental model, so
                // append in reverse to keep innermost-first ordering.
                frames.push_back(base[k_base_depth - 1 - i]);
            }
            // Mix the leaf_choice + per-tid into a single key for the
            // unique-path set (one leaf id per (tid, leaf_choice) pair).
            expected_unique_paths.insert((static_cast<std::uint64_t>(tid) << 32) ^
                                         leaf_choice);

            const auto r = filter.diff(tid, frames);
            EXPECT_NE(r.leaf_id, frame_trie::k_root);
        }
    }

    // The trie holds: k_base_depth shared prefix nodes + one leaf node
    // per (tid, leaf_choice) pair that actually got sampled.
    EXPECT_EQ(trie.size(), k_base_depth + expected_unique_paths.size());
    EXPECT_EQ(filter.tracked_tids(), k_num_tids);
}
