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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Unit tests for CounterController::check_power_performance_level_for_path(): the
// power_dpm_force_performance_level sysfs path is passed in directly (rather than
// resolved from a real agent's renderD node), so these run without a GPU.
#include "../controller.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>

namespace rocprofiler
{
namespace counters
{
namespace
{
constexpr uint64_t kFakeNodeId         = 7;
constexpr uint32_t kFakeDrmRenderMinor = 128;

// Owns a throwaway "<root>/renderD<N>/device/power_dpm_force_performance_level" file
// (mimicking the real sysfs layout); rm -rf on destruction.
struct fake_perf_level_file
{
    std::string               root;
    common::filesystem::path path;

    explicit fake_perf_level_file(const std::string& contents)
    {
        char tmpl[] = "/tmp/perf_level_test_XXXXXX";
        root        = mkdtemp(tmpl) ? tmpl : "";
        EXPECT_FALSE(root.empty()) << "mkdtemp failed";

        auto device_dir = common::filesystem::path{root} /
                          fmt::format("renderD{}", kFakeDrmRenderMinor) / "device";
        common::filesystem::create_directories(device_dir);
        path = device_dir / "power_dpm_force_performance_level";

        FILE* f = fopen(path.c_str(), "w");
        EXPECT_NE(f, nullptr) << "fopen " << path.string();
        if(f)
        {
            fwrite(contents.data(), 1, contents.size(), f);
            fclose(f);
        }
    }

    ~fake_perf_level_file()
    {
        if(!root.empty())
        {
            [[maybe_unused]] auto _rc = system(("rm -rf '" + root + "'").c_str());
        }
    }

    fake_perf_level_file(const fake_perf_level_file&)            = delete;
    fake_perf_level_file& operator=(const fake_perf_level_file&) = delete;
    fake_perf_level_file(fake_perf_level_file&&)                 = delete;
    fake_perf_level_file& operator=(fake_perf_level_file&&)      = delete;
};

std::string
check_and_capture_stderr(const common::filesystem::path& perf_path)
{
    testing::internal::CaptureStderr();
    CounterController::check_power_performance_level_for_path(
        perf_path, kFakeNodeId, kFakeDrmRenderMinor);
    return testing::internal::GetCapturedStderr();
}
}  // namespace

TEST(CounterControllerPerfLevel, ProfileStandardDoesNotWarn)
{
    fake_perf_level_file fake("profile_standard\n");

    auto output = check_and_capture_stderr(fake.path);

    EXPECT_FALSE(std::regex_search(output, std::regex("power_dpm_force_performance_level")))
        << "output=" << output;
}

TEST(CounterControllerPerfLevel, NonStandardLevelsWarn)
{
    // Anything other than 'profile_standard' (not just 'auto') must warn.
    const char* non_standard_levels[] = {"auto", "low", "high", "manual", "profile_peak"};

    for(const auto* level : non_standard_levels)
    {
        fake_perf_level_file fake(std::string(level) + "\n");

        auto output = check_and_capture_stderr(fake.path);

        auto expected =
            std::regex(fmt::format("Agent {} \\(renderD{}\\) has "
                                   "power_dpm_force_performance_level='{}'",
                                   kFakeNodeId,
                                   kFakeDrmRenderMinor,
                                   level));
        EXPECT_TRUE(std::regex_search(output, expected))
            << "level=" << level << " output=" << output;
    }
}

TEST(CounterControllerPerfLevel, EmptyFileWarns)
{
    fake_perf_level_file fake("");  // empty contents -> getline() yields an empty perf_level

    auto output = check_and_capture_stderr(fake.path);

    auto expected = std::regex(
        fmt::format("Could not get power_dpm_force_performance_level for Agent {} "
                    "\\(renderD{}\\)",
                    kFakeNodeId,
                    kFakeDrmRenderMinor));
    EXPECT_TRUE(std::regex_search(output, expected)) << "output=" << output;
}

TEST(CounterControllerPerfLevel, MissingFileDoesNotWarn)
{
    auto output = check_and_capture_stderr("/nonexistent/perf/level/path");

    EXPECT_TRUE(output.empty()) << "output=" << output;
}

}  // namespace counters
}  // namespace rocprofiler
