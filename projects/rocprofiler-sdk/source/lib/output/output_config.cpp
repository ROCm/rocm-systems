// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "output_config.hpp"

namespace rocprofiler
{
namespace tool
{
output_config
output_config::load_from_env()
{
    auto cfg = output_config{};
    cfg.parse_env();
    return cfg;
}

output_config
output_config::load_from_env(output_config&& cfg)
{
    cfg.parse_env();
    return cfg;
}

void
output_config::parse_env()
{
    stats         = common::get_env("ROCPROF_STATS", stats);
    stats_summary = common::get_env("ROCPROF_STATS_SUMMARY", stats_summary);
    stats_summary_per_domain =
        common::get_env("ROCPROF_STATS_SUMMARY_PER_DOMAIN", stats_summary_per_domain);
    stats_summary_unit = common::get_env("ROCPROF_STATS_SUMMARY_UNITS", stats_summary_unit);
    stats_summary_file = common::get_env("ROCPROF_STATS_SUMMARY_OUTPUT", stats_summary_file);

    perfetto_backend = common::get_env("ROCPROF_PERFETTO_BACKEND", perfetto_backend);
    perfetto_buffer_fill_policy =
        common::get_env("ROCPROF_PERFETTO_BUFFER_FILL_POLICY", perfetto_buffer_fill_policy);
    perfetto_shmem_size_hint =
        common::get_env("ROCPROF_PERFETTO_SHMEM_SIZE_HINT_KB", perfetto_shmem_size_hint);
    perfetto_buffer_size = common::get_env("ROCPROF_PERFETTO_BUFFER_SIZE_KB", perfetto_buffer_size);

    output_path    = common::get_env("ROCPROF_OUTPUT_PATH", output_path);
    output_file    = common::get_env("ROCPROF_OUTPUT_FILE_NAME", output_file);
    tmp_directory  = common::get_env("ROCPROF_TMPDIR", tmp_directory);
    kernel_rename  = common::get_env("ROCPROF_KERNEL_RENAME", false);
    group_by_queue = common::get_env("ROCPROF_GROUP_BY_QUEUE", false);
    auto to_upper  = [](std::string val) {
        for(auto& vitr : val)
            vitr = toupper(vitr);
        return val;
    };

    output_format = common::get_env("ROCPROF_OUTPUT_FORMAT", output_format);
    auto entries  = std::set<std::string>{};
    for(const auto& itr : sdk::parse::tokenize(output_format, " \t,;:"))
        entries.emplace(to_upper(itr));

    csv_output     = entries.count("CSV") > 0;
    json_output    = entries.count("JSON") > 0;
    pftrace_output = entries.count("PFTRACE") > 0;
    otf2_output    = entries.count("OTF2") > 0;
    rocpd_output   = entries.count("ROCPD") > 0 || entries.empty();

    const auto supported_formats =
        std::set<std::string_view>{"CSV", "JSON", "PFTRACE", "OTF2", "ROCPD"};
    for(const auto& itr : entries)
    {
        LOG_IF(FATAL, supported_formats.count(itr) == 0)
            << "Unsupported output format type: " << itr;
    }

    std::string agent_index = common::get_env("ROCPROF_AGENT_INDEX", "relative");
    if(agent_index == "type-relative")
        agent_index_value = agent_indexing::logical_node_type;
    else if(agent_index == "absolute")
        agent_index_value = agent_indexing::node;
    else
        agent_index_value = agent_indexing::logical_node;

    const auto supported_perfetto_backends = std::set<std::string_view>{"inprocess", "system"};
    LOG_IF(FATAL, supported_perfetto_backends.count(perfetto_backend) == 0)
        << "Unsupported perfetto backend type: " << perfetto_backend;

    if(stats_summary_unit == "sec")
        stats_summary_unit_value = common::units::sec;
    else if(stats_summary_unit == "msec")
        stats_summary_unit_value = common::units::msec;
    else if(stats_summary_unit == "usec")
        stats_summary_unit_value = common::units::usec;
    else if(stats_summary_unit == "nsec")
        stats_summary_unit_value = common::units::nsec;
    else
    {
        ROCP_FATAL << "Unsupported summary units value: " << stats_summary_unit;
    }

    if(auto _summary_grps = common::get_env("ROCPROF_STATS_SUMMARY_GROUPS", "");
       !_summary_grps.empty())
    {
        stats_summary_groups =
            sdk::parse::tokenize(_summary_grps, std::vector<std::string_view>{"##@@##"});

        // remove any empty strings (just in case these slipped through)
        stats_summary_groups.erase(std::remove_if(stats_summary_groups.begin(),
                                                  stats_summary_groups.end(),
                                                  [](const auto& itr) { return itr.empty(); }),
                                   stats_summary_groups.end());
    }

    // enable summary output if any of these are enabled
    summary_output = (stats_summary || stats_summary_per_domain || !stats_summary_groups.empty());
}
}  // namespace tool
}  // namespace rocprofiler
