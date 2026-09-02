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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/filesystem.hpp"

#include <rocprofiler-sdk/agent.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace counters
{
bool
requires_fixed_counter_performance_level(std::string_view agent_arch);

common::filesystem::path
get_counter_performance_level_path(
    uint32_t                        drm_render_minor,
    const common::filesystem::path& drm_sysfs_root = "/sys/class/drm");

std::optional<std::string>
read_counter_performance_level(uint32_t                        drm_render_minor,
                               const common::filesystem::path& drm_sysfs_root = "/sys/class/drm");

bool
check_agent_counter_performance_level(
    const rocprofiler_agent_t&      agent,
    const common::filesystem::path& drm_sysfs_root = "/sys/class/drm");

}  // namespace counters
}  // namespace rocprofiler
