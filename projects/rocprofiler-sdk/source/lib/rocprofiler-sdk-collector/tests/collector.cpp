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

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace
{
namespace fs = std::filesystem;

constexpr auto test_app = ROCPROFILER_COLLECTOR_TEST_APP;

// A fresh, empty directory for one test; removed when the test ends.
struct scratch_dir
{
    fs::path path = {};

    scratch_dir()
    {
        auto templ = (fs::temp_directory_path() / "collector-test-XXXXXX").string();
        auto buf   = std::vector<char>{templ.begin(), templ.end()};
        buf.emplace_back('\0');
        EXPECT_NE(::mkdtemp(buf.data()), nullptr);
        path = buf.data();
    }

    ~scratch_dir()
    {
        auto ec = std::error_code{};
        fs::remove_all(path, ec);
    }

    scratch_dir(const scratch_dir&) = delete;
    scratch_dir(scratch_dir&&)      = delete;
    scratch_dir& operator=(const scratch_dir&) = delete;
    scratch_dir& operator=(scratch_dir&&) = delete;
};

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
