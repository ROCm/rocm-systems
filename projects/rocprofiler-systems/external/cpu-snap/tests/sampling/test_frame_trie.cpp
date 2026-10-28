// Unit tests for cpu_snap::sampling::internal::frame_trie. The trie is the
// Flyweight at the heart of the v1 streaming pipeline's L4
// compression layer.

#include <cstddef>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/frame_trie.hpp"

using cpu_snap::sampling::internal::frame_trie;

TEST(FrameTrie, EmptyTrieReportsZeroState)
{
    const frame_trie trie;
    EXPECT_EQ(trie.size(), 0U);
    EXPECT_EQ(trie.last_assigned_id(), frame_trie::k_root);
    EXPECT_EQ(trie.find(frame_trie::k_root, 0x42), frame_trie::k_root);
}

TEST(FrameTrie, FirstInternAllocatesIdOneWithWasNew)
{
    frame_trie trie;

    const auto r1 = trie.intern(frame_trie::k_root, 0xAAA);
    EXPECT_EQ(r1.id, 1U);
    EXPECT_TRUE(r1.was_new);
    EXPECT_EQ(trie.size(), 1U);
    EXPECT_EQ(trie.last_assigned_id(), 1U);

    const auto r2 = trie.intern(frame_trie::k_root, 0xAAA);
    EXPECT_EQ(r2.id, 1U);
    EXPECT_FALSE(r2.was_new);
    EXPECT_EQ(trie.size(), 1U);
}

TEST(FrameTrie, ChainExtendsViaParentChainAndReusesPrefix)
{
    frame_trie trie;

    const auto id1 = trie.intern(frame_trie::k_root, 0xAA).id;
    const auto id2 = trie.intern(id1, 0xBB).id;
    const auto id3 = trie.intern(id2, 0xCC).id;

    EXPECT_EQ(id1, 1U);
    EXPECT_EQ(id2, 2U);
    EXPECT_EQ(id3, 3U);

    const auto v3 = trie.node_at(id3);
    EXPECT_EQ(v3.id, id3);
    EXPECT_EQ(v3.parent_id, id2);
    EXPECT_EQ(v3.ip, 0xCCU);

    // Distinct root-level frame -> distinct id.
    const auto id4 = trie.intern(frame_trie::k_root, 0xDD);
    EXPECT_EQ(id4.id, 4U);
    EXPECT_TRUE(id4.was_new);

    // Reusing the (id1, 0xBB) edge must return id2 with was_new=false.
    const auto id2_again = trie.intern(id1, 0xBB);
    EXPECT_EQ(id2_again.id, id2);
    EXPECT_FALSE(id2_again.was_new);
    EXPECT_EQ(trie.size(), 4U);
}

TEST(FrameTrie, SiblingsUnderSameParentGetDistinctIds)
{
    frame_trie trie;

    const auto parent = trie.intern(frame_trie::k_root, 0x100).id;
    const auto a      = trie.intern(parent, 0x200).id;
    const auto b      = trie.intern(parent, 0x201).id;
    const auto c      = trie.intern(parent, 0x202).id;

    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(b, c);
    EXPECT_EQ(trie.node_at(a).parent_id, parent);
    EXPECT_EQ(trie.node_at(b).parent_id, parent);
    EXPECT_EQ(trie.node_at(c).parent_id, parent);
    EXPECT_EQ(trie.size(), 4U);
}

TEST(FrameTrie, HeavyChainWalksBackToRoot)
{
    constexpr std::size_t k_count = 1'000'000;

    frame_trie trie;

    frame_trie::node_id cur = frame_trie::k_root;
    for(std::size_t i = 0; i < k_count; ++i)
    {
        const auto r = trie.intern(cur, static_cast<std::uint64_t>(i) | 0xCAFE0000ULL);
        ASSERT_TRUE(r.was_new);
        cur = r.id;
    }

    EXPECT_EQ(trie.size(), k_count);
    EXPECT_EQ(trie.last_assigned_id(), static_cast<frame_trie::node_id>(k_count));

    std::size_t walked = 0;
    while(cur != frame_trie::k_root)
    {
        cur = trie.node_at(cur).parent_id;
        ++walked;
    }
    EXPECT_EQ(walked, k_count);
}

TEST(FrameTrie, MaxNodesCapThrowsOnOverflow)
{
    frame_trie trie;
    trie.set_max_nodes(5);

    for(std::uint64_t i = 1; i <= 5; ++i)
    {
        const auto r = trie.intern(frame_trie::k_root, i);
        EXPECT_TRUE(r.was_new);
        EXPECT_EQ(r.id, static_cast<frame_trie::node_id>(i));
    }
    EXPECT_EQ(trie.size(), 5U);

    EXPECT_THROW(
        { [[maybe_unused]] auto r = trie.intern(frame_trie::k_root, 6); },
        cpu_snap::resource_error);
    EXPECT_EQ(trie.size(), 5U);

    // Re-interning an existing edge MUST still succeed at the cap.
    const auto r_existing = trie.intern(frame_trie::k_root, 3);
    EXPECT_EQ(r_existing.id, 3U);
    EXPECT_FALSE(r_existing.was_new);
}

TEST(FrameTrie, FindMatchesInternIdWithoutGrowing)
{
    frame_trie trie;

    const auto id1 = trie.intern(frame_trie::k_root, 0x11).id;
    const auto id2 = trie.intern(id1, 0x22).id;

    const auto before = trie.size();
    EXPECT_EQ(trie.find(frame_trie::k_root, 0x11), id1);
    EXPECT_EQ(trie.find(id1, 0x22), id2);
    EXPECT_EQ(trie.find(id2, 0x33), frame_trie::k_root);  // not present
    EXPECT_EQ(trie.size(), before);

    // After a confirming intern of the absent edge, find sees it.
    const auto id3 = trie.intern(id2, 0x33).id;
    EXPECT_EQ(trie.find(id2, 0x33), id3);
}

TEST(FrameTrie, KeyCollisionSeparatesParentAndIp)
{
    // Distinct (parent, ip) pairs that share a low-bit overlap must
    // still resolve to distinct nodes. Guards against a key function
    // that drops parent bits.
    frame_trie trie;

    const auto a = trie.intern(/*parent*/ 1, /*ip*/ 2).id;
    const auto b = trie.intern(/*parent*/ 2, /*ip*/ 1).id;
    EXPECT_NE(a, b);

    const auto a_view = trie.node_at(a);
    const auto b_view = trie.node_at(b);
    EXPECT_EQ(a_view.parent_id, 1U);
    EXPECT_EQ(a_view.ip, 2U);
    EXPECT_EQ(b_view.parent_id, 2U);
    EXPECT_EQ(b_view.ip, 1U);
}

TEST(FrameTrie, EdgesWithEqualMixedKeyAreDistinctNodes)
{
    constexpr std::uint64_t k_mix = 0x9E3779B97F4A7C15ULL;
    frame_trie              trie;

    const auto first  = trie.intern(1, 0);
    const auto second = trie.intern(0, k_mix);

    EXPECT_TRUE(first.was_new);
    EXPECT_TRUE(second.was_new);
    EXPECT_NE(first.id, second.id);
    EXPECT_EQ(trie.find(1, 0), first.id);
    EXPECT_EQ(trie.find(0, k_mix), second.id);
    EXPECT_EQ(trie.node_at(second.id).parent_id, 0U);
    EXPECT_EQ(trie.node_at(second.id).ip, k_mix);
}
