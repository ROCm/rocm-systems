// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <pybind11/pybind11.h>

#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, m) \
    [[maybe_unused]] static void _roctx_recordfn_test_module_stub_(pybind11::module_& m)

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "../roctx_recordfn.cpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

void reset_state()
{
    if (is_installed())
    {
        uninstall();
    }
    g_stack.clear();
    g_dbg_guards.clear();
    for (auto& shard : g_shards)
    {
        std::lock_guard<std::mutex> guard(shard.mu);
        shard.snapshots.clear();
        shard.lru_order.clear();
        shard.lru_idx.clear();
    }
    g_n_pushes.store(0);
    g_n_pops.store(0);
    g_n_snapshots_saved.store(0);
    g_n_snapshots_consumed.store(0);
    g_n_snapshots_dropped.store(0);
    g_n_callback_errors.store(0);
    g_n_user_scope_pushes.store(0);
    g_n_user_scope_pops.store(0);
    g_n_userscope_inherits.store(0);
}

class RoctxRecordFnTest : public ::testing::Test
{
protected:
    void SetUp() override { reset_state(); }

    void TearDown() override { reset_state(); }
};

}  // namespace

TEST(LeafContext, ForwardTopLevelLeafIsAten)
{
    EXPECT_STREQ(roctx_recordfn::default_leaf_context(false, 42, true), roctx_recordfn::kAtenTopLevelLeaf);
}

TEST(LeafContext, ForwardNestedLeafIsAtenNested)
{
    EXPECT_STREQ(roctx_recordfn::default_leaf_context(false, 42, false), roctx_recordfn::kAtenNestedLeaf);
}

TEST(LeafContext, BackwardWithSeqLeafIsAutogradBwd)
{
    EXPECT_STREQ(roctx_recordfn::default_leaf_context(true, 7, true),
                 roctx_recordfn::kAutogradBackwardLeaf);
}

TEST(LeafContext, BackwardWithoutSeqLeafIsAutogradEngine)
{
    EXPECT_STREQ(roctx_recordfn::default_leaf_context(true, -1, true),
                 roctx_recordfn::kAutogradEngineLeaf);
}

TEST_F(RoctxRecordFnTest, SaveThenConsumeReturnsSavedStack)
{
    const std::vector<StackEntry> stack = {{"A", "a"}, {"B", "b"}};
    save_snapshot(42, stack);

    std::vector<StackEntry> out;
    ASSERT_TRUE(consume_snapshot(42, &out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].marker, "A");
    EXPECT_EQ(out[0].context, "a");
    EXPECT_EQ(out[1].marker, "B");
    EXPECT_EQ(out[1].context, "b");
    EXPECT_EQ(g_n_snapshots_saved.load(), 1u);
    EXPECT_EQ(g_n_snapshots_consumed.load(), 1u);
}

TEST_F(RoctxRecordFnTest, ConsumeUnknownReturnsFalse)
{
    std::vector<StackEntry> out;
    EXPECT_FALSE(consume_snapshot(999, &out));
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(g_n_snapshots_consumed.load(), 0u);
}

TEST_F(RoctxRecordFnTest, ConsumeIsOneShot)
{
    save_snapshot(7, std::vector<StackEntry>{{"X", "x"}});

    std::vector<StackEntry> out;
    ASSERT_TRUE(consume_snapshot(7, &out));
    EXPECT_FALSE(consume_snapshot(7, &out));
    EXPECT_EQ(g_n_snapshots_consumed.load(), 1u);
}

TEST_F(RoctxRecordFnTest, SaveTwiceReturnsLatest)
{
    save_snapshot(1, std::vector<StackEntry>{{"first", "f"}});
    save_snapshot(1, std::vector<StackEntry>{{"second", "s"}});

    std::vector<StackEntry> out;
    ASSERT_TRUE(consume_snapshot(1, &out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].marker, "second");
    EXPECT_EQ(g_n_snapshots_saved.load(), 2u);
}

TEST_F(RoctxRecordFnTest, EvictsOldestPastSoftCap)
{
    // Multiples of NUM_SHARDS keep every seq on shard 0.
    const std::int64_t step = static_cast<std::int64_t>(NUM_SHARDS);
    for (std::size_t i = 0; i < SHARD_SOFT_CAP; ++i)
    {
        save_snapshot(static_cast<std::int64_t>(i) * step, std::vector<StackEntry>{{"k", "v"}});
    }
    ASSERT_EQ(g_n_snapshots_dropped.load(), 0u);

    save_snapshot(static_cast<std::int64_t>(SHARD_SOFT_CAP) * step, std::vector<StackEntry>{{"k", "v"}});
    EXPECT_EQ(g_n_snapshots_dropped.load(), 1u);

    std::vector<StackEntry> out;
    EXPECT_FALSE(consume_snapshot(0, &out));
    EXPECT_TRUE(consume_snapshot(static_cast<std::int64_t>(SHARD_SOFT_CAP) * step, &out));
}

TEST_F(RoctxRecordFnTest, EvictionIsPerShard)
{
    const std::int64_t step = static_cast<std::int64_t>(NUM_SHARDS);
    for (std::size_t i = 0; i < SHARD_SOFT_CAP; ++i)
    {
        save_snapshot(static_cast<std::int64_t>(i) * step, std::vector<StackEntry>{{"k", "v"}});
    }
    save_snapshot(1, std::vector<StackEntry>{{"shard1", "v"}});

    save_snapshot(static_cast<std::int64_t>(SHARD_SOFT_CAP) * step, std::vector<StackEntry>{{"k", "v"}});

    std::vector<StackEntry> out;
    EXPECT_TRUE(consume_snapshot(1, &out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].marker, "shard1");
}

TEST_F(RoctxRecordFnTest, ConcurrentSaveConsumeNoLoss)
{
    constexpr int            n_threads  = 4;
    constexpr int            per_thread = 256;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t)
    {
        threads.emplace_back(
            [t]()
            {
                for (int i = 0; i < per_thread; ++i)
                {
                    const std::int64_t seq = static_cast<std::int64_t>(t) * 100000 + i;
                    save_snapshot(seq, std::vector<StackEntry>{{"k", "v"}});
                    std::vector<StackEntry> out;
                    consume_snapshot(seq, &out);
                }
            });
    }
    for (auto& th : threads)
    {
        th.join();
    }
    const auto expected = static_cast<std::uint64_t>(n_threads) * per_thread;
    EXPECT_EQ(g_n_snapshots_saved.load(), expected);
    EXPECT_EQ(g_n_snapshots_consumed.load(), expected);
}

TEST_F(RoctxRecordFnTest, PushPopAreBalanced)
{
    constexpr int n = 100;
    for (int i = 0; i < n; ++i)
    {
        push_user_scope("m" + std::to_string(i), "c");
    }
    EXPECT_EQ(g_stack.size(), static_cast<std::size_t>(n));
    EXPECT_EQ(g_dbg_guards.size(), static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i)
    {
        pop_user_scope();
    }
    EXPECT_TRUE(g_stack.empty());
    EXPECT_TRUE(g_dbg_guards.empty());
    EXPECT_EQ(g_n_user_scope_pushes.load(), static_cast<std::uint64_t>(n));
    EXPECT_EQ(g_n_user_scope_pops.load(), static_cast<std::uint64_t>(n));
}

TEST_F(RoctxRecordFnTest, PopOnEmptyBumpsCallbackErrors)
{
    ASSERT_TRUE(g_stack.empty());
    pop_user_scope();
    EXPECT_TRUE(g_stack.empty());
    EXPECT_EQ(g_n_user_scope_pops.load(), 0u);
    EXPECT_EQ(g_n_callback_errors.load(), 1u);
}

TEST_F(RoctxRecordFnTest, DeepNestingPreservesOrder)
{
    constexpr int depth = 256;
    for (int i = 0; i < depth; ++i)
    {
        push_user_scope("m" + std::to_string(i), "c" + std::to_string(i));
    }
    ASSERT_EQ(g_stack.size(), static_cast<std::size_t>(depth));

    for (int i = depth - 1; i >= 0; --i)
    {
        ASSERT_EQ(g_stack.back().marker, "m" + std::to_string(i));
        pop_user_scope();
    }
    EXPECT_TRUE(g_stack.empty());
}

TEST_F(RoctxRecordFnTest, InstallReturnsValidHandle)
{
    const auto handle = install();
    EXPECT_NE(handle, static_cast<std::int64_t>(at::INVALID_CALLBACK_HANDLE));
    EXPECT_TRUE(is_installed());
}

TEST_F(RoctxRecordFnTest, InstallIsIdempotent)
{
    const auto first  = install();
    const auto second = install();
    EXPECT_EQ(first, second);
    EXPECT_TRUE(is_installed());
}

TEST_F(RoctxRecordFnTest, UninstallClearsState)
{
    install();
    ASSERT_TRUE(is_installed());

    uninstall();
    EXPECT_FALSE(is_installed());
    EXPECT_EQ(g_handle.load(), at::INVALID_CALLBACK_HANDLE);
}

TEST_F(RoctxRecordFnTest, UninstallWhenNotInstalledIsNoOp)
{
    ASSERT_FALSE(is_installed());
    uninstall();
    EXPECT_FALSE(is_installed());
}

TEST_F(RoctxRecordFnTest, InstallAfterUninstallReinstalls)
{
    install();
    uninstall();

    const auto handle = install();
    EXPECT_NE(handle, static_cast<std::int64_t>(at::INVALID_CALLBACK_HANDLE));
    EXPECT_TRUE(is_installed());
}

TEST_F(RoctxRecordFnTest, EmptyParentChainIsNoOp)
{
    ASSERT_TRUE(g_stack.empty());
    EXPECT_EQ(apply_userscope_overlay(), 0u);
    EXPECT_TRUE(g_stack.empty());
    EXPECT_EQ(g_n_userscope_inherits.load(), 0u);
}

TEST_F(RoctxRecordFnTest, CopiesParentChain)
{
    auto info = std::make_shared<RoctxUserScopeChain>(
        std::vector<StackEntry>{{"P1", "c1"}, {"P2", "c2"}});
    c10::DebugInfoGuard guard(kRoctxDbgKind, info);

    ASSERT_TRUE(g_stack.empty());
    EXPECT_EQ(apply_userscope_overlay(), 2u);
    ASSERT_EQ(g_stack.size(), 2u);
    EXPECT_EQ(g_stack[0].marker, "P1");
    EXPECT_EQ(g_stack[0].context, "c1");
    EXPECT_EQ(g_stack[1].marker, "P2");
    EXPECT_EQ(g_stack[1].context, "c2");
    EXPECT_EQ(g_n_userscope_inherits.load(), 1u);
}

TEST_F(RoctxRecordFnTest, DedupesIdenticalPrefix)
{
    auto info = std::make_shared<RoctxUserScopeChain>(
        std::vector<StackEntry>{{"P1", "c1"}, {"P2", "c2"}});
    c10::DebugInfoGuard guard(kRoctxDbgKind, info);

    g_stack.push_back(StackEntry{"P1", "c1"});
    g_stack.push_back(StackEntry{"P2", "c2"});

    EXPECT_EQ(apply_userscope_overlay(), 0u);
    EXPECT_EQ(g_stack.size(), 2u);
    EXPECT_EQ(g_n_userscope_inherits.load(), 0u);
}
