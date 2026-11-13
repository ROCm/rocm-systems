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

#include <rocprofiler-sdk-collector/collector.h>
#include <rocprofiler-sdk-collector/records.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

#include "lib/rocprofiler-sdk-collector/ctf/reader.hpp"
#include "lib/rocprofiler-sdk-collector/join.hpp"

#include "format_args.hpp"
#include "scratch_dir.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace
{
namespace fs = std::filesystem;

constexpr auto test_app = ROCPROFILER_COLLECTOR_TEST_APP;

using rocprofiler::collector::test::scratch_dir;

rocprofiler_collector_config_t
make_config(const char* output_dir, const char* const* providers)
{
    auto config = rocprofiler_collector_config_t{};
    std::memset(&config, 0, sizeof(config));
    config.size       = sizeof(config);
    config.output_dir = output_dir;
    config.providers  = providers;
    return config;
}

const char* const test_providers[] = {"rocprofiler_collector_test", nullptr};

// Whether a session daemon can be found the way the collector looks for one.
bool
have_session_daemon()
{
    if(const auto* prefix = std::getenv("ROCPROFILER_COLLECTOR_LTTNG_PREFIX"))
        return ::access((fs::path{prefix} / "bin" / "lttng-sessiond").c_str(), X_OK) == 0;
    const auto* path = std::getenv("PATH");
    if(!path) return false;
    auto stream = std::string{path};
    auto start  = size_t{0};
    while(start <= stream.size())
    {
        auto end = stream.find(':', start);
        auto dir = stream.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if(!dir.empty() && ::access((fs::path{dir} / "lttng-sessiond").c_str(), X_OK) == 0)
            return true;
        if(end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

#define SKIP_WITHOUT_SESSION_DAEMON()                                                              \
    do                                                                                             \
    {                                                                                              \
        if(!have_session_daemon())                                                                 \
            GTEST_SKIP() << "lttng-sessiond not found (set ROCPROFILER_COLLECTOR_LTTNG_PREFIX)";   \
    } while(0)

#define SKIP_WITHOUT_TEST_APP()                                                                    \
    do                                                                                             \
    {                                                                                              \
        if(std::strlen(test_app) == 0)                                                             \
            GTEST_SKIP() << "collector-test-app not built (no lttng-ust development files)";       \
    } while(0)

// The per-process trace directories (trace/ust/pid/<name>-<pid>-<date>) with a metadata
// file and at least one non-empty data stream.
size_t
count_recorded_processes(const fs::path& trace)
{
    auto count = size_t{0};
    auto root  = trace / "ust" / "pid";
    if(!fs::is_directory(root)) return 0;
    for(const auto& dir : fs::directory_iterator{root})
    {
        if(!dir.is_directory() || !fs::exists(dir.path() / "metadata")) continue;
        auto has_data = false;
        for(const auto& file : fs::directory_iterator{dir.path()})
        {
            if(file.path().filename().string().rfind("rocprofiler_collector", 0) == 0 &&
               fs::file_size(file.path()) > 0)
                has_data = true;
        }
        if(has_data) ++count;
    }
    return count;
}
}  // namespace

TEST(collector, status_names_and_strings)
{
    for(int i = 0; i < ROCPROFILER_COLLECTOR_STATUS_LAST; ++i)
    {
        auto status = static_cast<rocprofiler_collector_status_t>(i);
        ASSERT_NE(rocprofiler_collector_get_status_name(status), nullptr) << i;
        ASSERT_NE(rocprofiler_collector_get_status_string(status), nullptr) << i;
    }
    EXPECT_STREQ(rocprofiler_collector_get_status_name(ROCPROFILER_COLLECTOR_STATUS_SUCCESS),
                 "ROCPROFILER_COLLECTOR_STATUS_SUCCESS");
    EXPECT_EQ(rocprofiler_collector_get_status_name(ROCPROFILER_COLLECTOR_STATUS_LAST), nullptr);
    EXPECT_EQ(rocprofiler_collector_get_status_string(ROCPROFILER_COLLECTOR_STATUS_LAST), nullptr);
}

extern "C" int
rocprofiler_collector_c_api_compiles(void);

TEST(collector, c_api_compiles)
{
    // Built from c_api.c, as C. Runs no daemon: the output directory cannot be created.
    EXPECT_EQ(rocprofiler_collector_c_api_compiles(), 1);
}

TEST(collector, version)
{
    auto major = uint32_t{0};
    auto minor = uint32_t{0};
    auto patch = uint32_t{0};
    EXPECT_EQ(rocprofiler_collector_get_version(&major, &minor, &patch),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(major, ROCPROFILER_COLLECTOR_VERSION_MAJOR);
    EXPECT_EQ(minor, ROCPROFILER_COLLECTOR_VERSION_MINOR);
    EXPECT_EQ(patch, ROCPROFILER_COLLECTOR_VERSION_PATCH);
}

TEST(collector, create_rejects_invalid_configuration)
{
    auto dir       = scratch_dir{};
    auto collector = rocprofiler_collector_t{};

    auto no_output = make_config(nullptr, test_providers);
    EXPECT_EQ(rocprofiler_collector_create(&no_output, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);

    auto too_small = make_config(dir.path.c_str(), test_providers);
    too_small.size = sizeof(uint64_t);
    EXPECT_EQ(rocprofiler_collector_create(&too_small, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI);

    const char* const no_providers[] = {nullptr};
    auto              empty          = make_config(dir.path.c_str(), no_providers);
    EXPECT_EQ(rocprofiler_collector_create(&empty, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);

    auto bad_policy          = make_config(dir.path.c_str(), test_providers);
    bad_policy.buffer_policy = ROCPROFILER_COLLECTOR_BUFFER_POLICY_LAST;
    EXPECT_EQ(rocprofiler_collector_create(&bad_policy, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST(collector, lets_launched_targets_block_only_when_lossless)
{
    SKIP_WITHOUT_SESSION_DAEMON();

    // The target exits with 0 when liblttng-ust would let it wait for buffer space.
    const char* const argv[] = {"/bin/sh", "-c", "test \"$LTTNG_UST_ALLOW_BLOCKING\" = 1", nullptr};
    ::setenv("LTTNG_UST_ALLOW_BLOCKING", "0", 1);  // replaced, not duplicated, in the target

    for(auto policy :
        {ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS, ROCPROFILER_COLLECTOR_BUFFER_POLICY_DISCARD})
    {
        auto dir             = scratch_dir{};
        auto config          = make_config(dir.path.c_str(), test_providers);
        config.buffer_policy = policy;
        auto collector       = rocprofiler_collector_t{};
        ASSERT_EQ(rocprofiler_collector_create(&config, &collector),
                  ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
        ASSERT_EQ(rocprofiler_collector_start(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
        auto pid = 0;
        ASSERT_EQ(rocprofiler_collector_launch(collector, argv, &pid),
                  ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
        auto exit_status = -1;
        ASSERT_EQ(rocprofiler_collector_wait(collector, pid, &exit_status),
                  ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
        EXPECT_EQ(exit_status, (policy == ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS) ? 0 : 1);
        EXPECT_EQ(rocprofiler_collector_destroy(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    }
    ::unsetenv("LTTNG_UST_ALLOW_BLOCKING");
}

TEST(collector, create_rejects_non_empty_output_directory)
{
    auto dir = scratch_dir{};
    std::ofstream{dir.path / "existing"} << "data";

    auto config    = make_config(dir.path.c_str(), test_providers);
    auto collector = rocprofiler_collector_t{};
    EXPECT_EQ(rocprofiler_collector_create(&config, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY);
}

TEST(collector, create_reports_missing_lttng_tools)
{
    auto dir            = scratch_dir{};
    auto config         = make_config((dir.path / "out").c_str(), test_providers);
    config.lttng_prefix = "/nonexistent/lttng-tools";
    auto collector      = rocprofiler_collector_t{};
    EXPECT_EQ(rocprofiler_collector_create(&config, &collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND);
}

TEST(collector, unknown_handle_is_rejected)
{
    auto unknown = rocprofiler_collector_t{12345};
    EXPECT_EQ(rocprofiler_collector_start(unknown),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_collector_stop(unknown),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_collector_destroy(unknown),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST(collector, enforces_call_order_and_one_collector)
{
    SKIP_WITHOUT_SESSION_DAEMON();

    auto dir       = scratch_dir{};
    auto config    = make_config(dir.path.c_str(), test_providers);
    auto collector = rocprofiler_collector_t{};
    ASSERT_EQ(rocprofiler_collector_create(&config, &collector),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    // A second collector cannot exist at the same time.
    auto other_dir    = scratch_dir{};
    auto other_config = make_config(other_dir.path.c_str(), test_providers);
    auto other        = rocprofiler_collector_t{};
    EXPECT_EQ(rocprofiler_collector_create(&other_config, &other),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE);

    // Launch and stop need a started session.
    const char* const argv[] = {"true", nullptr};
    auto              pid    = 0;
    EXPECT_EQ(rocprofiler_collector_launch(collector, argv, &pid),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE);
    EXPECT_EQ(rocprofiler_collector_stop(collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE);

    ASSERT_EQ(rocprofiler_collector_start(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_collector_start(collector),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE);

    // A program that cannot be executed is reported by launch.
    const char* const missing[] = {"/nonexistent/program", nullptr};
    EXPECT_EQ(rocprofiler_collector_launch(collector, missing, &pid),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH);

    // Wait accepts only processes this collector launched.
    auto exit_status = 0;
    EXPECT_EQ(rocprofiler_collector_wait(collector, ::getpid(), &exit_status),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(rocprofiler_collector_destroy(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    // After destroy, a new collector can be created.
    EXPECT_EQ(rocprofiler_collector_create(&other_config, &other),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_collector_destroy(other), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
}

TEST(collector, records_a_launched_process_tree)
{
    SKIP_WITHOUT_SESSION_DAEMON();
    SKIP_WITHOUT_TEST_APP();

    ::setenv("LTTNG_HOME", "/collector-test-previous-home", 1);

    auto dir       = scratch_dir{};
    auto config    = make_config(dir.path.c_str(), test_providers);
    auto collector = rocprofiler_collector_t{};
    ASSERT_EQ(rocprofiler_collector_create(&config, &collector),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    // The collector's private home, while it exists.
    auto home = std::string{std::getenv("LTTNG_HOME")};
    EXPECT_NE(home, "/collector-test-previous-home");
    EXPECT_TRUE(fs::is_directory(home));

    ASSERT_EQ(rocprofiler_collector_start(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    const char* const argv[] = {test_app, "100", "3", "child", nullptr};
    auto              pid    = 0;
    ASSERT_EQ(rocprofiler_collector_launch(collector, argv, &pid),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    auto exit_status = -1;
    ASSERT_EQ(rocprofiler_collector_wait(collector, pid, &exit_status),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(exit_status, 3);

    ASSERT_EQ(rocprofiler_collector_stop(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    const char* trace_path = nullptr;
    ASSERT_EQ(rocprofiler_collector_get_trace_path(collector, &trace_path),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(fs::path{trace_path}, dir.path / "trace");

    // The target and its child (fork and exec) each have their own buffers.
    EXPECT_EQ(count_recorded_processes(trace_path), 2u);

    ASSERT_EQ(rocprofiler_collector_destroy(collector), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    // Destroy restores LTTNG_HOME, removes the private home and keeps the output.
    EXPECT_STREQ(std::getenv("LTTNG_HOME"), "/collector-test-previous-home");
    EXPECT_FALSE(fs::exists(home));
    EXPECT_TRUE(fs::exists(dir.path / "trace"));
    EXPECT_TRUE(fs::exists(dir.path / "lttng-sessiond.log"));
    ::unsetenv("LTTNG_HOME");
}

namespace
{
namespace collector = rocprofiler::collector;
namespace ctf       = rocprofiler::collector::ctf;

// Records the test app with the given arguments and joins its calls.
struct recorded_calls
{
    std::vector<collector::api_call>  calls = {};
    std::vector<collector::event_gap> gaps  = {};
    collector::join_statistics        stats = {};
};

void
record_and_join(std::vector<const char*>              args,
                uint64_t                              subbuffer_size,
                uint64_t                              num_subbuffers,
                rocprofiler_collector_buffer_policy_t policy,
                recorded_calls*                       result)
{
    auto dir              = scratch_dir{};
    auto config           = make_config(dir.path.c_str(), test_providers);
    config.subbuffer_size = subbuffer_size;
    config.num_subbuffers = num_subbuffers;
    config.buffer_policy  = policy;
    auto handle           = rocprofiler_collector_t{};
    ASSERT_EQ(rocprofiler_collector_create(&config, &handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_collector_start(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    args.insert(args.begin(), test_app);
    args.push_back(nullptr);
    auto pid = 0;
    ASSERT_EQ(rocprofiler_collector_launch(handle, args.data(), &pid),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    auto exit_status = -1;
    ASSERT_EQ(rocprofiler_collector_wait(handle, pid, &exit_status),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    EXPECT_EQ(exit_status, 0);
    ASSERT_EQ(rocprofiler_collector_stop(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    const char* trace_path = nullptr;
    ASSERT_EQ(rocprofiler_collector_get_trace_path(handle, &trace_path),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    auto traces = std::vector<ctf::trace_ptr>{};
    auto error  = std::string{};
    for(const auto& path : ctf::find_traces(trace_path))
    {
        auto source = std::make_shared<ctf::trace>();
        ASSERT_TRUE(ctf::load_trace(path, source.get(), &error)) << error;
        traces.push_back(source);
    }
    EXPECT_TRUE(collector::join_api_calls(
        traces,
        [result](const collector::api_call& call) { result->calls.push_back(call); },
        &result->gaps,
        &result->stats,
        &error))
        << error;
    EXPECT_EQ(rocprofiler_collector_destroy(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
}
}  // namespace

TEST(collector, joins_recorded_calls)
{
    SKIP_WITHOUT_SESSION_DAEMON();
    SKIP_WITHOUT_TEST_APP();

    // 4 threads of 50 calls in the target, and a child with one thread of 50 calls; each
    // "outer" call makes an "inner" call. The buffers hold every event.
    auto result = recorded_calls{};
    record_and_join({"50", "0", "child", "threads=4"},
                    1 << 20,
                    4,
                    ROCPROFILER_COLLECTOR_BUFFER_POLICY_DISCARD,
                    &result);
    if(HasFatalFailure()) return;

    EXPECT_EQ(result.stats.read.discarded, 0u);
    EXPECT_TRUE(result.gaps.empty());
    EXPECT_EQ(result.stats.lost_exits + result.stats.open_at_stop + result.stats.unmatched_exits +
                  result.stats.ambiguous_exits,
              0u);
    EXPECT_EQ(result.stats.other_events, 250u);  // ping
    ASSERT_EQ(result.calls.size(), 500u);

    auto by_id   = std::map<std::pair<uint64_t, uint64_t>, const collector::api_call*>{};
    auto threads = std::map<uint64_t, std::set<uint64_t>>{};
    for(const auto& call : result.calls)
    {
        by_id[{call.pid, call.correlation_id}] = &call;
        threads[call.pid].insert(call.tid);
    }
    ASSERT_EQ(threads.size(), 2u);
    auto thread_counts = std::multiset<size_t>{};
    for(const auto& [pid, tids] : threads)
        thread_counts.insert(tids.size());
    EXPECT_EQ(thread_counts, (std::multiset<size_t>{1, 4}));

    for(const auto& call : result.calls)
    {
        EXPECT_EQ(call.provider, "rocprofiler_collector_test");
        ASSERT_NE(call.end, collector::no_end);
        EXPECT_LE(call.start, call.end);
        if(call.name == "outer")
        {
            EXPECT_EQ(call.depth, 0u);
            EXPECT_EQ(call.parent_id, 0u);
            ASSERT_EQ(call.arguments.size(), 6u);
            auto index = call.arguments[0].i64;
            EXPECT_EQ(call.arguments[0].name, "index");
            EXPECT_EQ(call.arguments[1].i64, -index);
            EXPECT_EQ(call.arguments[2].str, "call-" + std::to_string(index));
            EXPECT_EQ(call.arguments[3].str, call.arguments[2].str);  // sequence of text
            EXPECT_DOUBLE_EQ(call.arguments[4].f64, 0.5 * static_cast<double>(index));
            EXPECT_EQ(call.arguments[5].enum_label, (index % 2 == 0) ? "even" : "odd");
            ASSERT_EQ(call.results.size(), 1u);
            EXPECT_EQ(call.results[0].i64, index);
        }
        else
        {
            ASSERT_EQ(call.name, "inner");
            EXPECT_EQ(call.depth, 1u);
            const auto* parent = by_id.at({call.pid, call.parent_id});
            EXPECT_EQ(parent->name, "outer");
            EXPECT_EQ(parent->tid, call.tid);
            EXPECT_LE(parent->start, call.start);
            EXPECT_GE(parent->end, call.end);
            auto index = parent->arguments[0].i64;
            ASSERT_EQ(call.arguments.size(), 1u);
            EXPECT_EQ(call.arguments[0].u64, 0xABCD0000u + static_cast<uint64_t>(index));
            EXPECT_EQ(call.results[0].i64, 2 * index);
        }
    }
}

TEST(collector, joins_calls_when_events_are_discarded)
{
    SKIP_WITHOUT_SESSION_DAEMON();
    SKIP_WITHOUT_TEST_APP();

    // Small buffers, many threads and the discard policy: some events are discarded. Every
    // call is still reported, and missing events are counted, not mispaired.
    auto result = recorded_calls{};
    record_and_join(
        {"20000", "0", "threads=8"}, 4096, 2, ROCPROFILER_COLLECTOR_BUFFER_POLICY_DISCARD, &result);
    if(HasFatalFailure()) return;

    auto gap_events = uint64_t{0};
    for(const auto& gap : result.gaps)
    {
        gap_events += gap.count;
        EXPECT_LE(gap.begin, gap.end);
        EXPECT_NE(gap.pid, 0u);
    }
    EXPECT_EQ(gap_events, result.stats.read.discarded);

    auto complete = uint64_t{0};
    for(const auto& call : result.calls)
    {
        if(call.end == collector::no_end) continue;
        ++complete;
        EXPECT_LE(call.start, call.end);
        ASSERT_EQ(call.results.size(), 1u);
        if(call.name == "outer")
        {
            EXPECT_EQ(call.results[0].i64, call.arguments[0].i64);  // paired with its own exit
        }
    }
    EXPECT_EQ(result.calls.size(), complete + result.stats.lost_exits + result.stats.open_at_stop);
    // Every recorded event is in a call, or counted.
    EXPECT_EQ(result.stats.read.events,
              (2 * complete) + result.stats.lost_exits + result.stats.open_at_stop +
                  result.stats.unmatched_exits + result.stats.ambiguous_exits +
                  result.stats.other_events);
    EXPECT_EQ(result.stats.read.events + result.stats.read.discarded, 8u * 20000u * 5u);
}

TEST(collector, loses_no_calls_when_lossless)
{
    SKIP_WITHOUT_SESSION_DAEMON();
    SKIP_WITHOUT_TEST_APP();

    // Buffers that discard events under the discard policy (8 threads writing 800 000
    // events back to back): with the lossless policy the threads wait for space, and every
    // event is recorded. (Smaller buffers also work, but the threads then mostly wait.)
    auto result = recorded_calls{};
    record_and_join({"20000", "0", "threads=8"},
                    65536,
                    2,
                    ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS,
                    &result);
    if(HasFatalFailure()) return;

    EXPECT_EQ(result.stats.read.discarded, 0u);
    EXPECT_TRUE(result.gaps.empty());
    EXPECT_EQ(result.stats.read.events, 8u * 20000u * 5u);
    EXPECT_EQ(result.calls.size(), 8u * 20000u * 2u);
    EXPECT_EQ(result.stats.lost_exits + result.stats.open_at_stop + result.stats.unmatched_exits +
                  result.stats.ambiguous_exits,
              0u);
}

namespace
{
uint64_t
boottime_now()
{
    auto now = timespec{};
    ::clock_gettime(CLOCK_BOOTTIME, &now);
    return (static_cast<uint64_t>(now.tv_sec) * 1000000000) + static_cast<uint64_t>(now.tv_nsec);
}

struct read_hip_call
{
    rocprofiler_buffer_tracing_hip_api_ext_record_t record = {};
    rocprofiler_collector_call_info_t               info   = {};
    std::vector<collector::test::arg_row>           rows   = {};
};
}  // namespace

TEST(collector, reads_hip_api_records)
{
    SKIP_WITHOUT_SESSION_DAEMON();
    SKIP_WITHOUT_TEST_APP();

    const char* const hip_providers[] = {"rocm_hip", nullptr};
    auto              dir             = scratch_dir{};
    auto              config          = make_config(dir.path.c_str(), hip_providers);
    auto              handle          = rocprofiler_collector_t{};
    ASSERT_EQ(rocprofiler_collector_create(&config, &handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    auto read_config       = rocprofiler_collector_read_config_t{};
    read_config.size       = sizeof(read_config);
    read_config.schema_dir = ROCPROFILER_COLLECTOR_TEST_SCHEMA_DIR;
    EXPECT_EQ(rocprofiler_collector_read(handle, &read_config, nullptr),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE);  // not stopped

    auto before = boottime_now();
    ASSERT_EQ(rocprofiler_collector_start(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    const char* const argv[] = {test_app, "4", "0", "hip", nullptr};
    auto              pid    = 0;
    ASSERT_EQ(rocprofiler_collector_launch(handle, argv, &pid),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    auto exit_status = -1;
    ASSERT_EQ(rocprofiler_collector_wait(handle, pid, &exit_status),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_collector_stop(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    auto after = boottime_now();

    // A schema directory without rocm_hip.json.
    auto no_schema       = read_config;
    no_schema.schema_dir = dir.path.c_str();
    EXPECT_EQ(rocprofiler_collector_read(handle, &no_schema, nullptr),
              ROCPROFILER_COLLECTOR_STATUS_ERROR_PROBE_SCHEMA);

    auto calls                   = std::vector<read_hip_call>{};
    read_config.user_data        = &calls;
    read_config.hip_api_callback = [](const rocprofiler_buffer_tracing_hip_api_ext_record_t* record,
                                      const rocprofiler_collector_call_info_t*               info,
                                      void* user_data) {
        static_cast<std::vector<read_hip_call>*>(user_data)->push_back(
            {*record, *info, collector::test::format_args(*record)});
    };
    auto stats = rocprofiler_collector_read_statistics_t{};
    ASSERT_EQ(rocprofiler_collector_read(handle, &read_config, &stats),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);

    EXPECT_EQ(stats.size, sizeof(stats));
    EXPECT_EQ(stats.events, 10u);
    EXPECT_EQ(stats.calls, 5u);
    EXPECT_EQ(stats.discarded_events + stats.calls_no_end + stats.calls_args_incomplete +
                  stats.unmatched_exits + stats.ambiguous_exits + stats.unknown_calls,
              0u);
    EXPECT_EQ(stats.clock_stable, 1u);
    ASSERT_EQ(calls.size(), 5u);

    const char* const kinds[] = {
        "HostToHost", "HostToDevice", "DeviceToHost", "DeviceToDevice", "Default"};
    for(uint64_t i = 0; i < 4; ++i)
    {
        const auto& call = calls[i];
        EXPECT_EQ(call.record.kind, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT);
        EXPECT_EQ(call.record.correlation_id.internal, i + 1);
        EXPECT_EQ(call.record.correlation_id.ancestor, 0u);
        EXPECT_EQ(call.record.thread_id, static_cast<uint64_t>(pid));  // the main thread
        EXPECT_EQ(call.info.pid, static_cast<uint32_t>(pid));
        EXPECT_EQ(call.info.flags, 0u);
        EXPECT_LE(before, call.record.start_timestamp);  // converted to CLOCK_BOOTTIME
        EXPECT_LE(call.record.start_timestamp, call.record.end_timestamp);
        EXPECT_LE(call.record.end_timestamp, after);
        EXPECT_EQ(call.record.retval.hipError_t_retval, hipSuccess);

        auto dst    = fmt::format("{:#x}", 0x7f0000000000ULL + (i * 0x1000));
        auto stream = (i % 2 == 0) ? std::string{"0"} : std::string{"0x55a44b4332b0"};
        EXPECT_EQ(call.rows,
                  (std::vector<collector::test::arg_row>{
                      {0, "void*", "dst", dst},
                      {1, "void const*", "src", "0x7f1000000000"},
                      {2, "unsigned long", "sizeBytes", std::to_string(4096 * (i + 1))},
                      {3, "hipMemcpyKind", "kind", kinds[i]},
                      {4, "ihipStream_t*", "stream", stream}}))
            << "call " << i;
    }

    const auto& get_function = calls[4];
    EXPECT_EQ(get_function.record.operation, ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleGetFunction);
    EXPECT_EQ(get_function.rows,
              (std::vector<collector::test::arg_row>{
                  {0, "ihipModuleSymbol_t**", "function", "0x7ffd00001000"},
                  {1, "ihipModule_t*", "module", "0x560000002000"},
                  {2, "char const*", "kname", "kernel_name"}}));

    // The strings stay valid until destroy, and a second read gives the same records.
    auto again            = std::vector<read_hip_call>{};
    read_config.user_data = &again;
    ASSERT_EQ(rocprofiler_collector_read(handle, &read_config, nullptr),
              ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
    ASSERT_EQ(again.size(), calls.size());
    EXPECT_EQ(again[4].rows, get_function.rows);
    EXPECT_EQ(again[4].record.args.hipModuleGetFunction.kname,
              get_function.record.args.hipModuleGetFunction.kname);

    EXPECT_EQ(rocprofiler_collector_destroy(handle), ROCPROFILER_COLLECTOR_STATUS_SUCCESS);
}
