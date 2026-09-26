// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <pybind11/pybind11.h>

#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, m) \
    [[maybe_unused]] static void _roctx_recordfn_test_module_stub_(pybind11::module_& m)

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "../roctx_recordfn.cpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <gtest/gtest.h>

#include <array>
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

class RoctxRecordFnRealOpsTest : public RoctxRecordFnTest
{
protected:
    void SetUp() override
    {
        RoctxRecordFnTest::SetUp();
        if (!at::hasCUDA())
        {
            GTEST_SKIP() << "ATen built without CUDA support";
        }
    }
};

std::size_t pending_snapshots()
{
    std::size_t pending = 0;
    for (auto& shard : g_shards)
    {
        std::lock_guard<std::mutex> guard(shard.mu);
        pending += shard.snapshots.size();
    }
    return pending;
}

std::size_t count_in_marker_path(const std::string& wire, const std::string& needle)
{
    const auto  colon = wire.find(':');
    const auto  path  = (colon == std::string::npos) ? wire : wire.substr(0, colon);
    std::size_t count = 0;
    std::size_t pos   = 0;
    while ((pos = path.find(needle, pos)) != std::string::npos)
    {
        ++count;
        pos += needle.size();
    }
    return count;
}

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

namespace
{

// Reverses build_marker_string: split the operator path on the '/' separator,
// then decode each segment ('%2F' -> '/', then '%25' -> '%'), matching
// utils_analysis.build_call_trees.
std::vector<std::string> decode_marker_path(const std::string& wire)
{
    const auto        colon = wire.find(':');
    const std::string path  = (colon == std::string::npos) ? wire : wire.substr(0, colon);

    std::vector<std::string> segments;
    std::size_t              start = 0;
    while (true)
    {
        const auto sep = path.find('/', start);
        const std::string raw = path.substr(start,
                                            sep == std::string::npos ? std::string::npos : sep - start);

        std::string decoded;
        for (std::size_t i = 0; i < raw.size();)
        {
            if (raw.compare(i, 3, kEncodedSlash) == 0)
            {
                decoded += '/';
                i += 3;
            }
            else if (raw.compare(i, 3, kEncodedPercent) == 0)
            {
                decoded += '%';
                i += 3;
            }
            else
            {
                decoded += raw[i];
                ++i;
            }
        }
        segments.push_back(decoded);

        if (sep == std::string::npos)
            return segments;
        start = sep + 1;
    }
}

}  // namespace

TEST(MarkerEncoding, EscapesSlashAndPercentWithinNames)
{
    // '/' encodes to %2F and '%' to %25 within a name; the '/' between frames
    // remains the separator.
    const std::vector<StackEntry> stack = {
        {"Torch-Compiled Region: 0/0", "#1@a:1"},
        {"k%2F%name", "#2@b:2"},
    };

    const std::string wire = build_marker_string(stack);

    EXPECT_EQ(wire, "Torch-Compiled Region: 0%2F0/k%252F%25name:#1@a:1/#2@b:2");
}

TEST(MarkerEncoding, RoundTripsThroughBuildCallTreesDecode)
{
    // Encoding then decoding returns the original names.
    const std::vector<std::string> names = {
        "Torch-Compiled Region 0/0",
        "k%2F%name",
        "plain_kernel",
        "literal%2Fnot_a_slash",
        "100%/sec",
    };

    std::vector<StackEntry> stack;
    stack.reserve(names.size());
    for (const auto& name : names)
        stack.push_back(StackEntry{name, "ctx"});

    const std::vector<std::string> decoded = decode_marker_path(build_marker_string(stack));

    EXPECT_EQ(decoded, names);
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
        push_user_scope("m" + std::to_string(i), "c", "gtest");
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
        push_user_scope("m" + std::to_string(i), "c" + std::to_string(i), "gtest");
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

TEST_F(RoctxRecordFnRealOpsTest, FwdBwdCounterSanity)
{
    install();

    auto x = at::randn({8, 8}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
    push_user_scope("test.fwd_bwd", "#1@test:1", "gtest");
    auto y = (x * 2).sum();
    pop_user_scope();
    y.backward();

    EXPECT_GT(g_n_snapshots_saved.load(), 0u);
    EXPECT_GT(g_n_snapshots_consumed.load(), 0u);
    EXPECT_EQ(g_n_callback_errors.load(), 0u);
    EXPECT_EQ(g_n_pushes.load(), g_n_pops.load());
    EXPECT_EQ(g_n_user_scope_pushes.load(), g_n_user_scope_pops.load());
    EXPECT_LE(pending_snapshots(), 4u);
}

TEST_F(RoctxRecordFnRealOpsTest, CaptureLeafLabelsAndUserScope)
{
    install();
    start_capture();

    {
        auto warmup = at::randn({4, 4}, at::TensorOptions().device(at::kCUDA));
        (void)(warmup * 2).sum();
    }

    push_user_scope("test.outer_step", "#1@test:7", "gtest");
    auto x = at::randn({32, 32}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
    auto y = (x.matmul(x)).sum();
    y.backward();
    pop_user_scope();

    const auto captured = stop_capture();
    ASSERT_FALSE(captured.empty());

    bool        saw_aten_top      = false;
    bool        saw_aten_nested   = false;
    bool        saw_bwd_leaf      = false;
    bool        saw_legacy        = false;
    bool        saw_torch_backend = false;
    std::size_t bwd_total         = 0;
    std::size_t bwd_under_scope   = 0;

    const std::string backend_suffix = "|torch";

    for (const auto& m : captured)
    {
        if (m.find("aten:0") != std::string::npos)
            saw_aten_top = true;
        if (m.find("aten.nested:0") != std::string::npos)
            saw_aten_nested = true;
        if (m.find("autograd.bwd:0") != std::string::npos)
            saw_bwd_leaf = true;
        if (m.find("dispatcher:0") != std::string::npos)
            saw_legacy = true;

        const bool is_recordfn_op = m.find("aten:0") != std::string::npos ||
                                    m.find("aten.nested:0") != std::string::npos ||
                                    m.find("autograd.bwd:0") != std::string::npos ||
                                    m.find("autograd.engine:0") != std::string::npos;
        if (is_recordfn_op && m.size() >= backend_suffix.size() &&
            m.compare(m.size() - backend_suffix.size(), backend_suffix.size(), backend_suffix) == 0)
        {
            saw_torch_backend = true;
        }

        if (m.find("autograd.bwd:0") != std::string::npos ||
            m.find("autograd.engine:0") != std::string::npos)
        {
            ++bwd_total;
            if (m.rfind("test.outer_step/", 0) == 0)
            {
                ++bwd_under_scope;
                EXPECT_EQ(count_in_marker_path(m, "test.outer_step"), 1u) << m;
            }
        }
    }

    EXPECT_FALSE(saw_legacy);
    EXPECT_TRUE(saw_aten_top);
    EXPECT_TRUE(saw_aten_nested);
    EXPECT_TRUE(saw_bwd_leaf);
    EXPECT_TRUE(saw_torch_backend);
    ASSERT_GT(bwd_total, 0u);
    EXPECT_GT(bwd_under_scope, 0u);
    EXPECT_GT(g_n_userscope_inherits.load(), 0u);
}

TEST_F(RoctxRecordFnRealOpsTest, ManyStepsCorrelation)
{
    install();
    constexpr int n_steps = 16;
    for (int i = 0; i < n_steps; ++i)
    {
        auto x = at::randn({128, 128}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
        auto y = ((x.matmul(x)) + x).sum();
        y.backward();
    }

    const auto saved    = g_n_snapshots_saved.load();
    const auto consumed = g_n_snapshots_consumed.load();
    EXPECT_GT(saved, 0u);
    EXPECT_GE(consumed, saved / 2);
    EXPECT_EQ(g_n_snapshots_dropped.load(), 0u);
    EXPECT_EQ(g_n_callback_errors.load(), 0u);
}

TEST_F(RoctxRecordFnRealOpsTest, DetachedForwardBounded)
{
    install();
    for (int i = 0; i < 50; ++i)
    {
        auto x = at::randn({32, 32}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
        auto y = (x.matmul(x)).sum().detach();
        (void)y;
    }

    EXPECT_GT(g_n_snapshots_saved.load(), 0u);
    EXPECT_EQ(g_n_callback_errors.load(), 0u);
    // 50 forward-only iterations stay well below a single shard's soft cap.
    EXPECT_LT(pending_snapshots(), SHARD_SOFT_CAP);
    EXPECT_EQ(g_n_snapshots_dropped.load(), 0u);
}

TEST_F(RoctxRecordFnRealOpsTest, ConcurrentThreadsScopedMarkers)
{
    install();
    start_capture();

    constexpr int            n_workers = 4;
    std::vector<std::thread> threads;
    threads.reserve(n_workers);
    for (int wid = 0; wid < n_workers; ++wid)
    {
        threads.emplace_back(
            [wid]()
            {
                const std::string scope = "test.concurrent.worker" + std::to_string(wid);
                push_user_scope(scope, "#1@test_thread:" + std::to_string(wid), "gtest");
                for (int i = 0; i < 4; ++i)
                {
                    auto x = at::randn({64, 64}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
                    (x.matmul(x)).sum().backward();
                }
                pop_user_scope();
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }

    const auto captured = stop_capture();
    ASSERT_FALSE(captured.empty());

    const std::array<std::string, 4> cpp_leaves = {"aten:0",
                                                   "aten.nested:0",
                                                   "autograd.bwd:0",
                                                   "autograd.engine:0"};

    for (int wid = 0; wid < n_workers; ++wid)
    {
        const std::string prefix = "test.concurrent.worker" + std::to_string(wid) + "/";
        bool              saw    = false;
        for (const auto& m : captured)
        {
            if (m.rfind(prefix, 0) != 0)
                continue;
            for (const auto& leaf : cpp_leaves)
            {
                if (m.find(leaf) != std::string::npos)
                {
                    saw = true;
                    break;
                }
            }
            if (saw)
                break;
        }
        EXPECT_TRUE(saw) << "worker " << wid;
    }

    EXPECT_EQ(g_n_callback_errors.load(), 0u);
    EXPECT_EQ(g_n_pushes.load(), g_n_pops.load());
    EXPECT_EQ(g_n_user_scope_pushes.load(), g_n_user_scope_pops.load());
}
