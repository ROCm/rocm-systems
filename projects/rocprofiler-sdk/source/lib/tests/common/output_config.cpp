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

#include "lib/output/output_config.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace
{
struct scoped_env
{
    scoped_env(const char* name, std::optional<std::string_view> value)
    : name_v{name}
    {
        if(auto* current = std::getenv(name_v.c_str())) previous_v = std::string{current};

        if(value)
            setenv(name_v.c_str(), std::string{*value}.c_str(), 1);
        else
            unsetenv(name_v.c_str());
    }

    ~scoped_env()
    {
        if(previous_v)
            setenv(name_v.c_str(), previous_v->c_str(), 1);
        else
            unsetenv(name_v.c_str());
    }

    std::string                name_v     = {};
    std::optional<std::string> previous_v = {};
};

struct scoped_output_config_env
{
    scoped_env output_format{"ROCPROF_OUTPUT_FORMAT", std::nullopt};
    scoped_env perfetto_backend{"ROCPROF_PERFETTO_BACKEND", std::nullopt};
    scoped_env perfetto_buffer_size{"ROCPROF_PERFETTO_BUFFER_SIZE_KB", std::nullopt};
    scoped_env stats_summary_unit{"ROCPROF_STATS_SUMMARY_UNITS", std::nullopt};
};
}  // namespace

TEST(output_config, tmp_directory_follows_output_path_when_unset)
{
    auto clean_env   = scoped_output_config_env{};
    auto output_path = scoped_env{"ROCPROF_OUTPUT_PATH", "/tmp/rocprof-output"};
    auto tmp_dir     = scoped_env{"ROCPROF_TMPDIR", std::nullopt};

    auto cfg = rocprofiler::tool::output_config::load_from_env();

    EXPECT_EQ(cfg.output_path, "/tmp/rocprof-output");
    EXPECT_EQ(cfg.tmp_directory, "/tmp/rocprof-output");
}

TEST(output_config, tmp_directory_env_overrides_output_path)
{
    auto clean_env   = scoped_output_config_env{};
    auto output_path = scoped_env{"ROCPROF_OUTPUT_PATH", "/tmp/rocprof-output"};
    auto tmp_dir     = scoped_env{"ROCPROF_TMPDIR", "/tmp/rocprof-tmp"};

    auto cfg = rocprofiler::tool::output_config::load_from_env();

    EXPECT_EQ(cfg.output_path, "/tmp/rocprof-output");
    EXPECT_EQ(cfg.tmp_directory, "/tmp/rocprof-tmp");
}

TEST(output_config, custom_tmp_directory_is_preserved_when_not_implicit)
{
    auto clean_env   = scoped_output_config_env{};
    auto output_path = scoped_env{"ROCPROF_OUTPUT_PATH", "/tmp/rocprof-output"};
    auto tmp_dir     = scoped_env{"ROCPROF_TMPDIR", std::nullopt};

    auto cfg          = rocprofiler::tool::output_config{};
    cfg.output_path   = "/tmp/original-output";
    cfg.tmp_directory = "/tmp/custom-tmp";
    auto env_resolved = rocprofiler::tool::output_config::load_from_env(std::move(cfg));

    EXPECT_EQ(env_resolved.output_path, "/tmp/rocprof-output");
    EXPECT_EQ(env_resolved.tmp_directory, "/tmp/custom-tmp");
}
