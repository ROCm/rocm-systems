// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "args_capture.h"
#include "leaf_context.h"
#include "marker_stack.h"
#include "process_state.h"
#include "record_function_installation.h"
#include "roctx_range_intercept.h"
#include "snapshot_store.h"
#include "stack_entry.h"
#include "stats.h"
#include "user_scope.h"
#include "wire_format.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>
#include <gtest/gtest.h>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace torch_trace_collector::detail;

namespace
{

// Process-wide state used by these tests.
Stats& stats()
{
    return process_state().stats;
}

SnapshotStore& snapshots()
{
    return process_state().snapshots;
}

nlohmann::json load_marker_args_codec()
{
    std::ifstream in{MARKER_ARGS_CODEC_PATH};
    if (!in)
    {
        ADD_FAILURE() << "cannot open " << MARKER_ARGS_CODEC_PATH;
        return nlohmann::json::object();
    }
    return nlohmann::json::parse(in);
}

void reset_state()
{
    if (is_installed())
    {
        uninstall();
    }
    thread_state().stack.clear();
    thread_state().guards.clear();
    snapshots().clear();
    stats().pushes.store(0);
    stats().pops.store(0);
    stats().snapshots_saved.store(0);
    stats().snapshots_consumed.store(0);
    stats().snapshots_dropped.store(0);
    stats().snapshots_overwritten.store(0);
    stats().callback_errors.store(0);
    stats().user_scope_pushes.store(0);
    stats().user_scope_pops.store(0);
    stats().user_scope_inherits.store(0);
    (void)roctx_range_intercept::stop_recording();
}

class TorchTraceCollectorTest : public ::testing::Test
{
protected:
    void SetUp() override { reset_state(); }

    void TearDown() override { reset_state(); }
};

class TorchTraceCollectorRealOpsTest : public TorchTraceCollectorTest
{
protected:
    void SetUp() override
    {
        TorchTraceCollectorTest::SetUp();
        if (!at::hasCUDA())
        {
            GTEST_SKIP() << "ATen built without CUDA support";
        }
    }
};

// Stand-in thread ids. Real ids come from at::RecordFunction::currentThreadId();
// zero means no forward identity.
constexpr std::uint64_t kThreadA = 1;
constexpr std::uint64_t kThreadB = 2;
constexpr std::uint64_t kThreadC = 3;

// The first `count` sequence numbers that hash to `shard` for `thread_id`.
std::vector<std::int64_t> sequence_numbers_on_shard(std::size_t   shard,
                                                    std::uint64_t thread_id,
                                                    std::size_t   count)
{
    std::vector<std::int64_t> sequence_numbers;
    sequence_numbers.reserve(count);
    for (std::int64_t sequence_number = 0; sequence_numbers.size() < count; ++sequence_number)
    {
        if (SnapshotStore::shard_index(SnapshotKey{sequence_number, thread_id}) == shard)
        {
            sequence_numbers.push_back(sequence_number);
        }
    }
    return sequence_numbers;
}

constexpr const char kArgsSegmentPrefix[] = "|args=";

std::string encoded_args_for_operator(const std::vector<std::string>& recorded,
                                      const std::string&              operator_name)
{
    const std::string top_level_prefix = operator_name + ":";
    const std::string nested_prefix    = "/" + operator_name + ":";
    std::string       encoded_args;
    for (const auto& marker : recorded)
    {
        const bool match = marker.compare(0, top_level_prefix.size(), top_level_prefix) == 0 ||
                           marker.find(nested_prefix) != std::string::npos;
        if (!match)
        {
            continue;
        }
        const auto args_pos = marker.find(kArgsSegmentPrefix);
        if (args_pos == std::string::npos)
        {
            continue;
        }
        const auto        args_begin = args_pos + (sizeof(kArgsSegmentPrefix) - 1);
        const auto        args_end   = marker.find('|', args_begin);
        const std::string encoded    = (args_end == std::string::npos)
                                           ? marker.substr(args_begin)
                                           : marker.substr(args_begin, args_end - args_begin);
        if (!encoded_args.empty() && encoded_args != encoded)
        {
            ADD_FAILURE() << operator_name << " args differ: " << encoded_args << " vs " << encoded;
        }
        encoded_args = encoded;
    }
    if (encoded_args.empty())
    {
        std::string markers;
        for (const auto& marker : recorded)
        {
            markers += marker;
            markers += '\n';
        }
        ADD_FAILURE() << "no args segment for " << operator_name << ":\n" << markers;
    }
    return encoded_args;
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
    EXPECT_STREQ(torch_trace_collector::default_leaf_context(false, 42, true),
                 torch_trace_collector::kAtenTopLevelLeaf);
}

TEST(LeafContext, ForwardNestedLeafIsAtenNested)
{
    EXPECT_STREQ(torch_trace_collector::default_leaf_context(false, 42, false),
                 torch_trace_collector::kAtenNestedLeaf);
}

TEST(LeafContext, BackwardWithSeqLeafIsAutogradBwd)
{
    EXPECT_STREQ(torch_trace_collector::default_leaf_context(true, 7, true),
                 torch_trace_collector::kAutogradBackwardLeaf);
}

TEST(LeafContext, BackwardWithoutSeqLeafIsAutogradEngine)
{
    EXPECT_STREQ(torch_trace_collector::default_leaf_context(true, -1, true),
                 torch_trace_collector::kAutogradEngineLeaf);
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

TEST(RoctxRangeIntercept, RecordsPushA)
{
    roctx_range_intercept::start_recording();
    roctxRangePushA("probe");
    const auto recorded = roctx_range_intercept::stop_recording();
    roctxRangePop();

    ASSERT_EQ(recorded.size(), 1u);
    EXPECT_EQ(recorded.front(), "probe");
}

TEST(ArgsEncoding, MatchesMarkerArgsCodec)
{
    const nlohmann::json codec = load_marker_args_codec();
    ASSERT_FALSE(codec.empty());
    ASSERT_EQ(kMaxArgsLen, codec.at("max_args_len").get<std::size_t>());
    ASSERT_EQ(kMaxArgItems, codec.at("max_arg_items").get<std::size_t>());

    ASSERT_TRUE(codec.contains("cases"));
    ASSERT_FALSE(codec.at("cases").empty());
    for (const auto& test_case : codec.at("cases"))
    {
        const auto name      = test_case.at("name").get<std::string>();
        const auto plaintext = test_case.at("plaintext").get<std::string>();
        const auto encoded   = test_case.at("encoded").get<std::string>();
        EXPECT_EQ(encode_args(plaintext), encoded) << name;
    }
}

TEST(ArgsEncoding, AppendArgsSegmentPlacesEncodedBlobBeforeBackend)
{
    std::string full = "op:#1@x:1";
    append_args_segment(full, "a|b");
    full += "|torch";
    EXPECT_EQ(full, "op:#1@x:1|args=a%7Cb|torch");

    std::string empty_args = "op:#1@x:1";
    append_args_segment(empty_args, "");
    EXPECT_EQ(empty_args, "op:#1@x:1");
}

TEST(ArgsRendering, TensorRendersDtypeAndShape)
{
    const auto t = at::zeros({2, 3}, at::TensorOptions().dtype(at::kLong));
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(t), /*values=*/false), "int64[2x3]");

    const auto scalar = at::zeros({}, at::TensorOptions().dtype(at::kFloat));
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(scalar), /*values=*/false), "float32[]");
}

TEST(ArgsRendering, DtypeSpellingMatchesPythonTier)
{
    EXPECT_EQ(scalar_type_name(c10::ScalarType::Float), "float32");
    EXPECT_EQ(scalar_type_name(c10::ScalarType::BFloat16), "bfloat16");

    // The Python tiers spell these dtypes lowercase via str(tensor.dtype).
    EXPECT_EQ(scalar_type_name(c10::ScalarType::Float8_e5m2), "float8_e5m2");
    EXPECT_EQ(scalar_type_name(c10::ScalarType::Float8_e4m3fn), "float8_e4m3fn");
    EXPECT_EQ(scalar_type_name(c10::ScalarType::QInt8), "qint8");
    EXPECT_EQ(scalar_type_name(c10::ScalarType::ComplexHalf), "complex32");
}

TEST(ArgsRendering, UndefinedTensorRendersNone)
{
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(at::Tensor()), /*values=*/false), "None");
}

TEST(ArgsRendering, TensorListRendersBracketed)
{
    std::vector<at::Tensor> tensors = {
        at::zeros({2}, at::TensorOptions().dtype(at::kFloat)),
        at::zeros({3, 4}, at::TensorOptions().dtype(at::kInt)),
    };
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(tensors), /*values=*/false), "[float32[2], int32[3x4]]");
}

TEST(ArgsRendering, ScalarValueModeRendersValueOtherwiseTypeTag)
{
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(static_cast<int64_t>(7)), /*values=*/true), "7");
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(true), /*values=*/true), "True");
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(false), /*values=*/true), "False");
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(2.5), /*values=*/true), std::to_string(2.5));

    EXPECT_EQ(render_leaf_ivalue(c10::IValue(static_cast<int64_t>(7)), /*values=*/false),
              c10::IValue(static_cast<int64_t>(7)).tagKind());
    EXPECT_EQ(render_leaf_ivalue(c10::IValue(2.5), /*values=*/false), c10::IValue(2.5).tagKind());
}

TEST(ArgsRendering, CapArgsBlobTruncatesPastLimit)
{
    const std::string at_limit(kMaxArgsLen, 'a');
    EXPECT_EQ(cap_args_blob(at_limit), at_limit);

    const std::string over(kMaxArgsLen + 10, 'a');
    const std::string capped = cap_args_blob(over);
    EXPECT_EQ(capped.size(), kMaxArgsLen + 3);
    EXPECT_EQ(capped.compare(kMaxArgsLen, 3, "..."), 0);
    EXPECT_EQ(capped.substr(0, kMaxArgsLen), std::string(kMaxArgsLen, 'a'));
}

TEST_F(TorchTraceCollectorTest, RecordedArgsForMmReluCat)
{
    install(/*capture_args=*/true, /*capture_values=*/false);
    const auto opts = at::TensorOptions().dtype(at::kFloat);
    const auto a    = at::zeros({2, 3}, opts);
    const auto b    = at::zeros({3, 4}, opts);
    const auto t    = at::zeros({2, 3}, opts);

    roctx_range_intercept::start_recording();
    (void)at::mm(a, b);
    (void)at::relu(t);
    (void)at::cat({t, t}, 0);
    const auto recorded = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(recorded.empty());

    EXPECT_EQ(encoded_args_for_operator(recorded, "aten::mm"),
              encode_args("(self=float32[2x3], mat2=float32[3x4])"));
    EXPECT_EQ(encoded_args_for_operator(recorded, "aten::relu"), encode_args("(self=float32[2x3])"));
    EXPECT_EQ(encoded_args_for_operator(recorded, "aten::cat"),
              encode_args("(tensors=[float32[2x3], float32[2x3]], dim=Int)"));
}

TEST_F(TorchTraceCollectorTest, RecordedCatArgsIncludeDimValue)
{
    install(/*capture_args=*/true, /*capture_values=*/true);
    const auto t = at::zeros({2, 3}, at::TensorOptions().dtype(at::kFloat));

    roctx_range_intercept::start_recording();
    (void)at::cat({t, t}, 1);
    const auto recorded = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(recorded.empty());

    EXPECT_EQ(encoded_args_for_operator(recorded, "aten::cat"),
              encode_args("(tensors=[float32[2x3], float32[2x3]], dim=1)"));
}

TEST_F(TorchTraceCollectorTest, PushUserScopeEmitsArgsSegmentBeforeBackend)
{
    roctx_range_intercept::start_recording();
    push_user_scope("op", "#1@x:1", "torch", "(f32[2x2])");
    pop_user_scope();
    const auto captured = roctx_range_intercept::stop_recording();

    ASSERT_EQ(captured.size(), 1u);
    EXPECT_EQ(captured[0], "op:#1@x:1|args=(f32[2x2])|torch");
}

TEST_F(TorchTraceCollectorTest, InstallConfiguresArgsCaptureFlags)
{
    install(/*capture_args=*/false, /*capture_values=*/false);
    EXPECT_FALSE(args_capture_enabled());
    EXPECT_FALSE(args_values_enabled());
    uninstall();

    install(/*capture_args=*/true, /*capture_values=*/true);
    EXPECT_TRUE(args_capture_enabled());
    EXPECT_TRUE(args_values_enabled());
    uninstall();

    // Values require args capture to also be enabled.
    install(/*capture_args=*/false, /*capture_values=*/true);
    EXPECT_FALSE(args_capture_enabled());
    EXPECT_FALSE(args_values_enabled());
}

TEST_F(TorchTraceCollectorTest, SaveThenConsumeReturnsSavedStack)
{
    const std::vector<StackEntry> stack = {{"A", "a"}, {"B", "b"}};
    snapshots().save(42, kThreadA, stack);

    std::vector<StackEntry> out;
    ASSERT_TRUE(snapshots().consume(42, kThreadA, &out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].marker, "A");
    EXPECT_EQ(out[0].context, "a");
    EXPECT_EQ(out[1].marker, "B");
    EXPECT_EQ(out[1].context, "b");
    EXPECT_EQ(stats().snapshots_saved.load(), 1u);
    EXPECT_EQ(stats().snapshots_consumed.load(), 1u);
}

TEST_F(TorchTraceCollectorTest, ConsumeUnknownReturnsFalse)
{
    std::vector<StackEntry> out;
    EXPECT_FALSE(snapshots().consume(999, kThreadA, &out));
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(stats().snapshots_consumed.load(), 0u);
}

TEST_F(TorchTraceCollectorTest, ConsumeIsOneShot)
{
    snapshots().save(7, kThreadA, std::vector<StackEntry>{{"X", "x"}});

    std::vector<StackEntry> out;
    ASSERT_TRUE(snapshots().consume(7, kThreadA, &out));
    EXPECT_FALSE(snapshots().consume(7, kThreadA, &out));
    EXPECT_EQ(stats().snapshots_consumed.load(), 1u);
}

TEST_F(TorchTraceCollectorTest, SaveTwiceReturnsLatest)
{
    snapshots().save(1, kThreadA, std::vector<StackEntry>{{"first", "f"}});
    snapshots().save(1, kThreadA, std::vector<StackEntry>{{"second", "s"}});

    std::vector<StackEntry> out;
    ASSERT_TRUE(snapshots().consume(1, kThreadA, &out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].marker, "second");
    EXPECT_EQ(stats().snapshots_saved.load(), 2u);
    EXPECT_EQ(stats().snapshots_overwritten.load(), 1u);
}

TEST_F(TorchTraceCollectorTest, SameSeqNrOnDifferentThreadsDoesNotCollide)
{
    // Concurrent forward passes both count from zero, so one thread's snapshot
    // must not displace another's under the same sequence number.
    snapshots().save(0, kThreadA, std::vector<StackEntry>{{"threadA", "a"}});
    snapshots().save(0, kThreadB, std::vector<StackEntry>{{"threadB", "b"}});
    EXPECT_EQ(stats().snapshots_saved.load(), 2u);
    EXPECT_EQ(stats().snapshots_overwritten.load(), 0u);
    EXPECT_EQ(snapshots().pending(), 2u);

    std::vector<StackEntry> out_a;
    ASSERT_TRUE(snapshots().consume(0, kThreadA, &out_a));
    ASSERT_EQ(out_a.size(), 1u);
    EXPECT_EQ(out_a[0].marker, "threadA");

    std::vector<StackEntry> out_b;
    ASSERT_TRUE(snapshots().consume(0, kThreadB, &out_b));
    ASSERT_EQ(out_b.size(), 1u);
    EXPECT_EQ(out_b[0].marker, "threadB");

    std::vector<StackEntry> out_c;
    EXPECT_FALSE(snapshots().consume(0, kThreadC, &out_c));
    EXPECT_EQ(stats().snapshots_consumed.load(), 2u);
    EXPECT_EQ(snapshots().pending(), 0u);
}

TEST_F(TorchTraceCollectorTest, EvictsOldestPastMaxEntriesPerShard)
{
    const auto sequence_numbers =
        sequence_numbers_on_shard(0, kThreadA, SnapshotStore::Shard::kMaxEntriesPerShard + 1);
    for (std::size_t i = 0; i < SnapshotStore::Shard::kMaxEntriesPerShard; ++i)
    {
        snapshots().save(sequence_numbers[i], kThreadA, std::vector<StackEntry>{{"k", "v"}});
    }
    ASSERT_EQ(stats().snapshots_dropped.load(), 0u);

    snapshots().save(sequence_numbers.back(), kThreadA, std::vector<StackEntry>{{"k", "v"}});
    EXPECT_EQ(stats().snapshots_dropped.load(), 1u);

    std::vector<StackEntry> out;
    EXPECT_FALSE(snapshots().consume(sequence_numbers.front(), kThreadA, &out));
    EXPECT_TRUE(snapshots().consume(sequence_numbers.back(), kThreadA, &out));
}

TEST_F(TorchTraceCollectorTest, EvictionIsPerShard)
{
    const auto sequence_numbers =
        sequence_numbers_on_shard(0, kThreadA, SnapshotStore::Shard::kMaxEntriesPerShard + 1);
    const auto other_shard_sequence_numbers = sequence_numbers_on_shard(1, kThreadA, 1);
    for (std::size_t i = 0; i < SnapshotStore::Shard::kMaxEntriesPerShard; ++i)
    {
        snapshots().save(sequence_numbers[i], kThreadA, std::vector<StackEntry>{{"k", "v"}});
    }
    snapshots().save(other_shard_sequence_numbers.front(),
                     kThreadA,
                     std::vector<StackEntry>{{"shard1", "v"}});

    snapshots().save(sequence_numbers.back(), kThreadA, std::vector<StackEntry>{{"k", "v"}});

    std::vector<StackEntry> out;
    EXPECT_TRUE(snapshots().consume(other_shard_sequence_numbers.front(), kThreadA, &out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].marker, "shard1");
}

TEST_F(TorchTraceCollectorTest, ConcurrentOverlappingSeqNrsAreIsolated)
{
    // Every thread walks the same sequence numbers, mirroring a thread-local
    // counter. Each must read back its own snapshot.
    constexpr int            n_threads  = 4;
    constexpr int            per_thread = 256;
    std::atomic<int>         lost{0};
    std::atomic<int>         mismatched{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t)
    {
        threads.emplace_back(
            [t, &lost, &mismatched]()
            {
                const std::uint64_t thread_id = static_cast<std::uint64_t>(t) + 1;
                const std::string   marker    = "t" + std::to_string(t);
                for (int i = 0; i < per_thread; ++i)
                {
                    const std::int64_t seq = i;
                    snapshots().save(seq, thread_id, std::vector<StackEntry>{{marker, "v"}});
                    std::vector<StackEntry> out;
                    if (!snapshots().consume(seq, thread_id, &out))
                    {
                        ++lost;
                    }
                    else if (out.size() != 1 || out[0].marker != marker)
                    {
                        ++mismatched;
                    }
                }
            });
    }
    for (auto& th : threads)
    {
        th.join();
    }
    EXPECT_EQ(lost.load(), 0);
    EXPECT_EQ(mismatched.load(), 0);
    const auto expected = static_cast<std::uint64_t>(n_threads) * per_thread;
    EXPECT_EQ(stats().snapshots_saved.load(), expected);
    EXPECT_EQ(stats().snapshots_consumed.load(), expected);
    EXPECT_EQ(snapshots().pending(), 0u);
}

TEST_F(TorchTraceCollectorTest, PushPopAreBalanced)
{
    constexpr int n = 100;
    for (int i = 0; i < n; ++i)
    {
        push_user_scope("m" + std::to_string(i), "c", "gtest");
    }
    EXPECT_EQ(thread_state().stack.size(), static_cast<std::size_t>(n));
    EXPECT_EQ(thread_state().guards.size(), static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i)
    {
        pop_user_scope();
    }
    EXPECT_TRUE(thread_state().stack.empty());
    EXPECT_TRUE(thread_state().guards.empty());
    EXPECT_EQ(stats().user_scope_pushes.load(), static_cast<std::uint64_t>(n));
    EXPECT_EQ(stats().user_scope_pops.load(), static_cast<std::uint64_t>(n));
}

TEST_F(TorchTraceCollectorTest, PopOnEmptyBumpsCallbackErrors)
{
    ASSERT_TRUE(thread_state().stack.empty());
    pop_user_scope();
    EXPECT_TRUE(thread_state().stack.empty());
    EXPECT_EQ(stats().user_scope_pops.load(), 0u);
    EXPECT_EQ(stats().callback_errors.load(), 1u);
}

TEST_F(TorchTraceCollectorTest, DeepNestingPreservesOrder)
{
    constexpr int depth = 256;
    for (int i = 0; i < depth; ++i)
    {
        push_user_scope("m" + std::to_string(i), "c" + std::to_string(i), "gtest");
    }
    ASSERT_EQ(thread_state().stack.size(), static_cast<std::size_t>(depth));

    for (int i = depth - 1; i >= 0; --i)
    {
        ASSERT_EQ(thread_state().stack.back().marker, "m" + std::to_string(i));
        pop_user_scope();
    }
    EXPECT_TRUE(thread_state().stack.empty());
}

TEST_F(TorchTraceCollectorTest, InstallReturnsValidHandle)
{
    const auto handle = install();
    EXPECT_NE(handle, static_cast<std::int64_t>(at::INVALID_CALLBACK_HANDLE));
    EXPECT_TRUE(is_installed());
}

TEST_F(TorchTraceCollectorTest, InstallIsIdempotent)
{
    const auto first  = install();
    const auto second = install();
    EXPECT_EQ(first, second);
    EXPECT_TRUE(is_installed());
}

TEST_F(TorchTraceCollectorTest, UninstallClearsState)
{
    install();
    ASSERT_TRUE(is_installed());

    uninstall();
    EXPECT_FALSE(is_installed());
    const auto handle = process_state().install.rlock([](const InstallState& state)
                                                      { return state.handle; });
    EXPECT_EQ(handle, at::INVALID_CALLBACK_HANDLE);
}

TEST_F(TorchTraceCollectorTest, UninstallWhenNotInstalledIsNoOp)
{
    ASSERT_FALSE(is_installed());
    uninstall();
    EXPECT_FALSE(is_installed());
}

TEST_F(TorchTraceCollectorTest, InstallAfterUninstallReinstalls)
{
    install();
    uninstall();

    const auto handle = install();
    EXPECT_NE(handle, static_cast<std::int64_t>(at::INVALID_CALLBACK_HANDLE));
    EXPECT_TRUE(is_installed());
}

TEST_F(TorchTraceCollectorTest, EmptyParentChainIsNoOp)
{
    install();
    ASSERT_TRUE(thread_state().stack.empty());

    {
        at::RecordFunction rec(at::RecordScope::FUNCTION);
        rec.before("probe");
    }

    EXPECT_TRUE(thread_state().stack.empty());
    EXPECT_EQ(stats().user_scope_inherits.load(), 0u);
}

TEST_F(TorchTraceCollectorTest, CopiesParentChain)
{
    install();
    push_user_scope("P1", "c1", "gtest");
    push_user_scope("P2", "c2", "gtest");

    const at::ThreadLocalState tls;
    std::uint64_t              inherits = 0;
    std::vector<std::string>   recorded;
    bool                       worker_stack_was_empty = false;

    std::thread worker(
        [&]
        {
            at::ThreadLocalStateGuard guard(tls);
            worker_stack_was_empty = thread_state().stack.empty();
            roctx_range_intercept::start_recording();
            {
                at::RecordFunction rec(at::RecordScope::FUNCTION);
                rec.before("probe");
            }
            recorded = roctx_range_intercept::stop_recording();
            inherits = stats().user_scope_inherits.load();
        });
    worker.join();

    pop_user_scope();
    pop_user_scope();

    EXPECT_TRUE(worker_stack_was_empty);
    EXPECT_EQ(inherits, 1u);
    bool saw_overlay = false;
    for (const auto& message : recorded)
    {
        if (message.rfind("P1/P2/probe", 0) == 0)
        {
            saw_overlay = true;
            break;
        }
    }
    EXPECT_TRUE(saw_overlay);
}

TEST_F(TorchTraceCollectorRealOpsTest, ForwardBackwardCountersStayBalanced)
{
    install();

    auto x = at::randn({8, 8}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
    push_user_scope("test.fwd_bwd", "#1@test:1", "gtest");
    auto y = (x * 2).sum();
    pop_user_scope();
    y.backward();

    EXPECT_GT(stats().snapshots_saved.load(), 0u);
    EXPECT_GT(stats().snapshots_consumed.load(), 0u);
    EXPECT_EQ(stats().callback_errors.load(), 0u);
    EXPECT_EQ(stats().pushes.load(), stats().pops.load());
    EXPECT_EQ(stats().user_scope_pushes.load(), stats().user_scope_pops.load());
    EXPECT_LE(snapshots().pending(), 4u);
}

TEST_F(TorchTraceCollectorRealOpsTest, LeafLabelsAndUserScope)
{
    install();
    roctx_range_intercept::start_recording();

    {
        auto warmup = at::randn({4, 4}, at::TensorOptions().device(at::kCUDA));
        (void)(warmup * 2).sum();
    }

    push_user_scope("test.outer_step", "#1@test:7", "gtest");
    auto x = at::randn({32, 32}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
    auto y = (x.matmul(x)).sum();
    y.backward();
    pop_user_scope();

    const auto recorded = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(recorded.empty());

    bool        saw_aten_top           = false;
    bool        saw_aten_nested        = false;
    bool        saw_bwd_leaf           = false;
    bool        saw_legacy             = false;
    bool        saw_torch_backend      = false;
    bool        saw_labeled_tensor_arg = false;
    std::size_t bwd_total              = 0;
    std::size_t bwd_under_scope        = 0;

    const std::string backend_suffix = "|torch";

    for (const auto& m : recorded)
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

        // Leaf args render as "(name=dtype[dims], ...)".
        if (m.find("|args=(") != std::string::npos && m.find("=float32[") != std::string::npos)
        {
            saw_labeled_tensor_arg = true;
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
    EXPECT_TRUE(saw_labeled_tensor_arg);
    ASSERT_GT(bwd_total, 0u);
    EXPECT_GT(bwd_under_scope, 0u);
    EXPECT_GT(stats().user_scope_inherits.load(), 0u);
}

TEST_F(TorchTraceCollectorRealOpsTest, CaptureDisabledEmitsNoArgsSegment)
{
    install(/*capture_args=*/false, /*capture_values=*/false);
    roctx_range_intercept::start_recording();

    auto x = at::randn({8, 8}, at::TensorOptions().device(at::kCUDA));
    (void)(x.matmul(x)).sum();

    const auto captured = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(captured.empty());

    bool saw_torch_op = false;
    for (const auto& m : captured)
    {
        if (m.find("|torch") != std::string::npos)
        {
            saw_torch_op = true;
        }
        EXPECT_EQ(m.find("|args="), std::string::npos) << m;
    }
    EXPECT_TRUE(saw_torch_op);
    EXPECT_EQ(stats().callback_errors.load(), 0u);
}

TEST_F(TorchTraceCollectorRealOpsTest, ManyStepsCorrelation)
{
    install();
    constexpr int n_steps = 16;
    for (int i = 0; i < n_steps; ++i)
    {
        auto x = at::randn({128, 128}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
        auto y = ((x.matmul(x)) + x).sum();
        y.backward();
    }

    const auto saved    = stats().snapshots_saved.load();
    const auto consumed = stats().snapshots_consumed.load();
    EXPECT_GT(saved, 0u);
    EXPECT_GE(consumed, saved / 2);
    EXPECT_EQ(stats().snapshots_dropped.load(), 0u);
    EXPECT_EQ(stats().callback_errors.load(), 0u);
}

TEST_F(TorchTraceCollectorRealOpsTest, DetachedForwardBounded)
{
    install();
    for (int i = 0; i < 50; ++i)
    {
        auto x = at::randn({32, 32}, at::TensorOptions().device(at::kCUDA)).requires_grad_(true);
        auto y = (x.matmul(x)).sum().detach();
        (void)y;
    }

    EXPECT_GT(stats().snapshots_saved.load(), 0u);
    EXPECT_EQ(stats().callback_errors.load(), 0u);
    EXPECT_LT(snapshots().pending(), SnapshotStore::Shard::kMaxEntriesPerShard);
    EXPECT_EQ(stats().snapshots_dropped.load(), 0u);
}

TEST_F(TorchTraceCollectorRealOpsTest, ConcurrentThreadsScopedMarkers)
{
    install();
    roctx_range_intercept::start_recording();

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

    const auto recorded = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(recorded.empty());

    const std::array<std::string, 4> cpp_leaves = {"aten:0",
                                                   "aten.nested:0",
                                                   "autograd.bwd:0",
                                                   "autograd.engine:0"};

    for (int wid = 0; wid < n_workers; ++wid)
    {
        const std::string prefix = "test.concurrent.worker" + std::to_string(wid) + "/";
        bool              saw    = false;
        for (const auto& m : recorded)
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

    // Each worker's markers must carry only its own scope.
    for (const auto& m : recorded)
    {
        std::size_t workers_named = 0;
        for (int wid = 0; wid < n_workers; ++wid)
        {
            if (m.find("test.concurrent.worker" + std::to_string(wid)) != std::string::npos)
            {
                ++workers_named;
            }
        }
        EXPECT_LE(workers_named, 1u) << m;
    }

    EXPECT_EQ(stats().callback_errors.load(), 0u);
    EXPECT_EQ(stats().pushes.load(), stats().pops.load());
    EXPECT_EQ(stats().user_scope_pushes.load(), stats().user_scope_pops.load());
}
