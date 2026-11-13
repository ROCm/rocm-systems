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

#include "lib/rocprofiler-sdk-collector/join.hpp"

#include "ctf_writer.hpp"
#include "scratch_dir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace collector = rocprofiler::collector;
namespace ctf       = rocprofiler::collector::ctf;
namespace fs        = std::filesystem;
namespace test      = rocprofiler::collector::test;

using test::scratch_dir;

// The ids of the events of the test metadata.
enum event_id : uint64_t
{
    a_enter = 0,
    a_exit  = 1,
    b_enter = 2,
    b_exit  = 3,
    other   = 4,
};

std::string
join_metadata(uint64_t vpid)
{
    const auto enter = std::string{
        "\t\tinteger { size = 8; align = 8; signed = 0; encoding = none; base = 16; } "
        "__ms_Opcode;\n"
        "\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _arg;\n"};
    const auto exit =
        std::string{"\t\tinteger { size = 8; align = 8; signed = 0; encoding = none; base = 16; } "
                    "__ms_Opcode;\n"
                    "\t\tinteger { size = 32; align = 8; signed = 1; encoding = none; base = 10; } "
                    "_result;\n"};
    return test::metadata_prologue(vpid) +
           test::metadata_event("rocm_hip:A_enter;k0;", a_enter, enter) +
           test::metadata_event("rocm_hip:A_exit;k0;", a_exit, exit) +
           test::metadata_event("rocm_hip:B_enter;k0;", b_enter, enter) +
           test::metadata_event("rocm_hip:B_exit;k0;", b_exit, exit) +
           test::metadata_event("rocm_hip:Message", other, "");
}

// An event of the test metadata: enter events carry arg = value, exit events result.
test::test_event
event(uint64_t id, uint64_t timestamp, int32_t vpid, int32_t vtid, uint32_t value = 0)
{
    return test::test_event{id, timestamp, vpid, vtid, [id, value](test::bit_writer* out) {
                                if(id == other) return;
                                test::put(out, (id % 2 == 0) ? 1 : 2, 8);
                                test::put(out, value, 32);
                            }};
}

// One process's trace (one stream file per vector of events).
void
write_process(const fs::path&                                   root,
              int32_t                                           vpid,
              const std::vector<std::vector<test::test_event>>& streams)
{
    auto dir = root / ("app-" + std::to_string(vpid));
    fs::create_directories(dir);
    test::write_file(dir / "metadata", join_metadata(vpid));
    for(size_t i = 0; i < streams.size(); ++i)
    {
        auto packet   = test::test_packet{};
        packet.events = streams[i];
        test::write_stream(dir / ("channel0_" + std::to_string(i)), {packet});
    }
}

struct join_result
{
    bool                              ok    = false;
    std::string                       error = {};
    std::vector<collector::api_call>  calls = {};
    std::vector<collector::event_gap> gaps  = {};
    collector::join_statistics        stats = {};

    const collector::api_call* find(uint64_t pid, uint64_t correlation_id) const
    {
        for(const auto& call : calls)
            if(call.pid == pid && call.correlation_id == correlation_id) return &call;
        return nullptr;
    }
};

join_result
join(const fs::path& root)
{
    auto traces = std::vector<ctf::trace_ptr>{};
    auto result = join_result{};
    for(const auto& dir : ctf::find_traces(root.string()))
    {
        auto source = std::make_shared<ctf::trace>();
        EXPECT_TRUE(ctf::load_trace(dir, source.get(), &result.error)) << result.error;
        traces.push_back(source);
    }
    result.ok = collector::join_api_calls(
        traces,
        [&result](const collector::api_call& call) { result.calls.push_back(call); },
        &result.gaps,
        &result.stats,
        &result.error);
    return result;
}
}  // namespace

TEST(join, parses_api_event_names)
{
    auto provider = std::string{};
    auto api      = std::string{};
    auto is_enter = false;
    ASSERT_TRUE(collector::parse_api_event_name(
        "rocm_hip:hipMalloc_enter;k0;", &provider, &api, &is_enter));
    EXPECT_EQ(provider, "rocm_hip");
    EXPECT_EQ(api, "hipMalloc");
    EXPECT_TRUE(is_enter);

    ASSERT_TRUE(
        collector::parse_api_event_name("rocm_hsa:hsa_init_exit", &provider, &api, &is_enter));
    EXPECT_EQ(provider, "rocm_hsa");
    EXPECT_EQ(api, "hsa_init");
    EXPECT_FALSE(is_enter);

    // The keywords suffix may hold several keywords.
    ASSERT_TRUE(collector::parse_api_event_name("p:x_exit;k0;k3;", &provider, &api, &is_enter));
    EXPECT_EQ(api, "x");

    for(const auto* name :
        {"rocm_hip:Message", "rocm_hip:_enter", "no_provider_enter", ":a_enter", "p:a_enterx;k0;"})
        EXPECT_FALSE(collector::parse_api_event_name(name, &provider, &api, &is_enter)) << name;
}

TEST(join, pairs_nested_calls)
{
    auto dir = scratch_dir{};
    // A calls B; then a second A. The thread moves between streams (CPUs) in between.
    write_process(
        dir.path,
        7,
        {{event(a_enter, 100, 7, 70, 11), event(b_enter, 110, 7, 70, 12), event(other, 115, 7, 70)},
         {event(b_exit, 120, 7, 70, 0),
          event(a_exit, 130, 7, 70, static_cast<uint32_t>(-1)),
          event(a_enter, 140, 7, 70, 13),
          event(a_exit, 150, 7, 70, 0)}});

    auto result = join(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.stats.calls, 3u);
    EXPECT_EQ(result.stats.other_events, 1u);
    EXPECT_EQ(result.stats.lost_exits + result.stats.open_at_stop + result.stats.unmatched_exits,
              0u);
    ASSERT_EQ(result.calls.size(), 3u);

    // Reported at their exits; correlation ids in start order.
    EXPECT_EQ(result.calls[0].name, "B");
    EXPECT_EQ(result.calls[1].name, "A");

    const auto* outer = result.find(7, 1);
    const auto* inner = result.find(7, 2);
    const auto* next  = result.find(7, 3);
    ASSERT_TRUE(outer && inner && next);
    EXPECT_EQ(outer->provider, "rocm_hip");
    EXPECT_EQ(outer->tid, 70u);
    EXPECT_EQ(outer->start, 100u);
    EXPECT_EQ(outer->end, 130u);
    EXPECT_EQ(outer->parent_id, 0u);
    EXPECT_EQ(outer->depth, 0u);
    EXPECT_EQ(inner->name, "B");
    EXPECT_EQ(inner->parent_id, 1u);
    EXPECT_EQ(inner->depth, 1u);
    EXPECT_EQ(inner->start, 110u);
    EXPECT_EQ(inner->end, 120u);
    EXPECT_EQ(next->parent_id, 0u);

    // Transport fields are dropped.
    ASSERT_EQ(outer->arguments.size(), 1u);
    EXPECT_EQ(outer->arguments[0].name, "arg");
    EXPECT_EQ(outer->arguments[0].u64, 11u);
    ASSERT_EQ(outer->results.size(), 1u);
    EXPECT_EQ(outer->results[0].name, "result");
    EXPECT_EQ(outer->results[0].i64, -1);
}

TEST(join, keeps_threads_and_processes_apart)
{
    auto dir = scratch_dir{};
    // Two threads of process 7 interleave A calls; process 8 numbers its own calls.
    write_process(dir.path,
                  7,
                  {{event(a_enter, 100, 7, 70),
                    event(a_enter, 105, 7, 71),
                    event(a_exit, 110, 7, 70),
                    event(a_exit, 120, 7, 71)}});
    write_process(dir.path, 8, {{event(a_enter, 101, 8, 80), event(a_exit, 111, 8, 80)}});

    auto result = join(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.calls.size(), 3u);
    const auto* first         = result.find(7, 1);
    const auto* second        = result.find(7, 2);
    const auto* other_process = result.find(8, 1);
    ASSERT_TRUE(first && second && other_process);
    EXPECT_EQ(first->tid, 70u);
    EXPECT_EQ(first->end, 110u);
    EXPECT_EQ(second->tid, 71u);
    EXPECT_EQ(second->parent_id, 0u);  // not nested in the other thread's call
    EXPECT_EQ(second->end, 120u);
    EXPECT_EQ(other_process->end, 111u);
}

TEST(join, reports_missing_events)
{
    auto dir = scratch_dir{};
    write_process(dir.path,
                  7,
                  {{event(a_exit, 90, 7, 70),       // its enter was before the trace
                    event(a_enter, 100, 7, 70),     //
                    event(b_enter, 110, 7, 70),     // its exit is lost
                    event(a_exit, 130, 7, 70),      //
                    event(b_enter, 140, 7, 71)}});  // still running at the end

    auto result = join(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.stats.unmatched_exits, 1u);
    EXPECT_EQ(result.stats.lost_exits, 1u);
    EXPECT_EQ(result.stats.open_at_stop, 1u);
    ASSERT_EQ(result.calls.size(), 3u);
    const auto* outer   = result.find(7, 1);
    const auto* lost    = result.find(7, 2);
    const auto* running = result.find(7, 3);
    ASSERT_TRUE(outer && lost && running);
    EXPECT_EQ(outer->end, 130u);
    EXPECT_EQ(lost->end, collector::no_end);
    EXPECT_TRUE(lost->results.empty());
    EXPECT_EQ(lost->parent_id, 1u);
    EXPECT_EQ(running->end, collector::no_end);
}

TEST(join, does_not_pair_across_gaps)
{
    auto dir   = scratch_dir{};
    auto trace = dir.path / "app-9";
    fs::create_directories(trace);
    test::write_file(trace / "metadata", join_metadata(9));
    // Events were discarded after the first packet: the A_exit at 350 may end a later A
    // whose enter was lost, so it does not end the A started at 150. Calls on either side
    // of the gap pair as usual.
    auto packets = std::vector<test::test_packet>(3);
    packets[0]   = {
        100,
        200,
        0,
        0,
        {event(a_enter, 110, 9, 90), event(a_exit, 120, 9, 90), event(a_enter, 150, 9, 90)}};
    packets[1] = {300, 400, 4, 0, {event(a_exit, 350, 9, 90)}};
    packets[2] = {500, 600, 4, 0, {event(b_enter, 510, 9, 90), event(b_exit, 520, 9, 90)}};
    test::write_stream(trace / "channel0_0", packets);

    auto result = join(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.stats.read.discarded, 4u);
    ASSERT_EQ(result.gaps.size(), 1u);
    EXPECT_EQ(result.gaps[0].pid, 9u);
    EXPECT_EQ(result.gaps[0].count, 4u);
    EXPECT_EQ(result.gaps[0].begin, 200u);
    EXPECT_EQ(result.gaps[0].end, 400u);

    EXPECT_EQ(result.stats.ambiguous_exits, 1u);
    EXPECT_EQ(result.stats.lost_exits, 1u);
    ASSERT_EQ(result.calls.size(), 3u);
    EXPECT_EQ(result.find(9, 1)->end, 120u);
    EXPECT_EQ(result.find(9, 2)->end, collector::no_end);
    EXPECT_EQ(result.find(9, 3)->end, 520u);
}

TEST(join, requires_thread_context)
{
    auto dir   = scratch_dir{};
    auto trace = dir.path / "app";
    fs::create_directories(trace);
    auto metadata = join_metadata(7);
    // A stream without the vpid/vtid context.
    auto begin = metadata.find("\tevent.context");
    metadata.erase(begin, metadata.find("};", begin) + 3 - begin);
    test::write_file(trace / "metadata", metadata);

    auto packet    = test::test_packet{};
    packet.context = false;
    packet.events  = {event(a_enter, 10, 7, 70, 5)};
    test::write_stream(trace / "channel0_0", {packet});

    auto result = join(dir.path);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("no vpid/vtid context"), std::string::npos) << result.error;
}
