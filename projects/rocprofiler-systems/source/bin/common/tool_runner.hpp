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

    bool force_sampling                   = false;
    bool enable_fork                      = false;
    bool enable_launcher                  = false;
    bool show_sample_flag                 = false;
    bool disable_cputime_on_realtime_only = false;

    std::unordered_map<std::string, std::string> deprecated_flags = {};
};

tool_config
make_run_config();

tool_config
make_sample_config();

[[nodiscard]] int
run_tool(int argc, char** argv, const tool_config& config);

}  // namespace rocprofsys::common_utils
