// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocprofsys::common_utils
{

enum class tool_mode : uint8_t
{
    run,
    sample
};

struct tool_config
{
    tool_mode        mode;
    std::string_view tool_name;
    std::string_view version_name;
    std::string_view summary;
    std::string_view workflow;
    std::string_view output_prefix = {};

    std::unordered_map<std::string, std::string> deprecated_flags = {};

    [[nodiscard]] bool is_sample() const noexcept { return mode == tool_mode::sample; }
    [[nodiscard]] bool force_sampling() const noexcept { return is_sample(); }
    [[nodiscard]] bool enable_fork() const noexcept { return !is_sample(); }
    [[nodiscard]] bool enable_launcher() const noexcept { return !is_sample(); }
    [[nodiscard]] bool show_sample_flag() const noexcept { return !is_sample(); }
    [[nodiscard]] bool disable_cputime_on_realtime_only() const noexcept
    {
        return is_sample();
    }
};

tool_config
make_run_config();

tool_config
make_sample_config();

[[nodiscard]] int
run_tool(int argc, char** argv, const tool_config& config);

}  // namespace rocprofsys::common_utils
