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

#include "performance_level.hpp"

#include "lib/common/logging.hpp"

#include <fstream>

namespace rocprofiler
{
namespace counters
{
bool
requires_fixed_counter_performance_level(std::string_view agent_arch)
{
    constexpr auto gfx11_prefix = std::string_view{"gfx11"};
    constexpr auto gfx12_prefix = std::string_view{"gfx12"};

    return agent_arch.compare(0, gfx11_prefix.size(), gfx11_prefix) == 0 ||
           agent_arch.compare(0, gfx12_prefix.size(), gfx12_prefix) == 0;
}

common::filesystem::path
get_counter_performance_level_path(uint32_t                        drm_render_minor,
                                   const common::filesystem::path& drm_sysfs_root)
{
    return drm_sysfs_root / ("renderD" + std::to_string(drm_render_minor)) / "device" /
           "power_dpm_force_performance_level";
}

std::optional<std::string>
read_counter_performance_level(uint32_t                        drm_render_minor,
                               const common::filesystem::path& drm_sysfs_root)
{
    auto path  = get_counter_performance_level_path(drm_render_minor, drm_sysfs_root);
    auto file  = std::ifstream{path.string()};
    auto level = std::string{};

    if(!(file >> level)) return std::nullopt;
    return level;
}

bool
check_agent_counter_performance_level(const rocprofiler_agent_t&      agent,
                                      const common::filesystem::path& drm_sysfs_root)
{
    if(agent.type != ROCPROFILER_AGENT_TYPE_GPU || agent.name == nullptr ||
       !requires_fixed_counter_performance_level(agent.name))
        return true;

    auto level = read_counter_performance_level(agent.drm_render_minor, drm_sysfs_root);
    if(!level)
    {
        ROCP_INFO
            << "Unable to read GPU performance level for agent " << agent.node_id << " ("
            << agent.name << ", renderD" << agent.drm_render_minor << ") from '"
            << get_counter_performance_level_path(agent.drm_render_minor, drm_sysfs_root).string()
            << "'";
        return true;
    }

    if(*level == "auto")
    {
        ROCP_WARNING << "Agent " << agent.node_id << " (" << agent.name << ", renderD"
                     << agent.drm_render_minor
                     << ") is using the AUTO performance level. On gfx11 and gfx12, AUTO gates "
                        "the perfmon clock for some GPU blocks and can make supported counters "
                        "report zero. Set a fixed performance level such as STABLE_STD with "
                        "amd-smi (profile_standard via sysfs) before counter collection, then "
                        "restore AUTO afterward. See 'Setting GPU performance level for PMC "
                        "profiling' in the rocprofv3 documentation.";
        return false;
    }

    return true;
}

}  // namespace counters
}  // namespace rocprofiler
