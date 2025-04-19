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

#include "generateStats.hpp"
#include "domain_type.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"
#include "timestamps.hpp"

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <fmt/format.h>

#include <unistd.h>
#include <cstdint>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace
{
stats_entry_t
get_stats(const stats_map_t& data_v)
{
    auto _stats = stats_entry_t{};
    for(const auto& [id, value] : data_v)
    {
        _stats.entries.emplace_back(id, value);
        _stats.total += value;
    }

    return _stats.sort();
}
}  // namespace

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                    tool_metadata,
               const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& data)
{
    auto kernel_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto kernel_name = tool_metadata.get_kernel_name(record.dispatch_info.kernel_id,
                                                             record.correlation_id.external.value);

            kernel_stats[kernel_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(kernel_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                            tool_metadata,
               const generator<tool_buffer_tracing_hip_api_ext_record_t>& data)
{
    auto hip_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            hip_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(hip_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                               tool_metadata,
               const generator<rocprofiler_buffer_tracing_hsa_api_record_t>& data)
{
    auto hsa_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            hsa_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(hsa_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                tool_metadata,
               const generator<tool_buffer_tracing_memory_copy_ext_record_t>& data)
{
    auto memory_copy_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            memory_copy_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(memory_copy_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                  tool_metadata,
               const generator<rocprofiler_buffer_tracing_marker_api_record_t>& data)
{
    auto marker_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto _name = std::string_view{};

            if(record.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API &&
               (record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA ||
                record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA ||
                record.operation == ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA))
            {
                _name = tool_metadata.get_marker_message(record.correlation_id.internal);
            }
            else
            {
                _name = tool_metadata.get_operation_name(record.kind, record.operation);
            }

            marker_stats[_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(marker_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                      tool_metadata,
               const generator<tool_buffer_tracing_memory_allocation_ext_record_t>& data)
{
    auto memory_allocation_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            memory_allocation_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(memory_allocation_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata& /*tool_metadata*/,
               const generator<tool_counter_record_t>& /*data*/)
{
    return stats_entry_t{};
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                      tool_metadata,
               const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>& data)
{
    auto scratch_memory_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto op_name = tool_metadata.get_operation_name(record.kind, record.operation);
            scratch_memory_stats[op_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(scratch_memory_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                    tool_metadata,
               const generator<rocprofiler_buffer_tracing_rccl_api_ext_record_t>& data)
{
    auto rccl_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rccl_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(rccl_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata& tool_metadata,
               const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& data)
{
    auto rocdecode_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocdecode_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(rocdecode_stats);
}

stats_entry_t
generate_stats(const output_config& /*cfg*/,
               const metadata&                                                   tool_metadata,
               const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>& data)
{
    auto rocjpeg_stats = stats_map_t{};
    for(auto ditr : data)
    {
        for(auto record : data.get(ditr))
        {
            auto api_name = tool_metadata.get_operation_name(record.kind, record.operation);
            rocjpeg_stats[api_name] += (record.end_timestamp - record.start_timestamp);
        }
    }

    return get_stats(rocjpeg_stats);
}

namespace
{
void
generate_stats(const output_config&      cfg,
               output_stream&            os,
               std::string_view          label,
               const domain_stats_vec_t& data_v,
               std::string_view          indent_v)
{
    auto _data = stats_entry_t{};
    auto _cols = std::unordered_map<std::string_view, domain_type>{};

    auto _get_entry = [&_data, &_cols](domain_type      _domain,
                                       std::string_view _key) -> stats_data_t* {
        for(auto& itr : _data.entries)
        {
            if(itr.first == _key) return &itr.second;
        }

        _cols.emplace(_key, _domain);
        auto& itr = _data.entries.emplace_back(_key, stats_data_t{});
        return &itr.second;
    };

    uint64_t name_width   = 40;
    uint64_t domain_width = 12;
    for(const auto& itr : data_v)
    {
        for(const auto& eitr : itr.second.entries)
        {
            _data.total += eitr.second;
            auto* _entry = _get_entry(itr.first, eitr.first);
            *CHECK_NOTNULL(_entry) += eitr.second;
            name_width = std::max(name_width, eitr.first.length());
        }
        domain_width = std::max(domain_width, get_domain_column_name(itr.first).length());
    }

    if(!_data) return;

    std::sort(_data.entries.begin(), _data.entries.end(), [](const auto& lhs, const auto& rhs) {
        return (lhs.second.get_sum() > rhs.second.get_sum());
    });

    const float_type _total_duration = _data.total.get_sum();

    os << fmt::format("\n{}ROCPROFV3 {}:\n\n", indent_v, label) << std::flush;

    {
        auto _header = fmt::format(
            "| {:^{}} | {:^{}} | {:^15} | {:^15} | {:^15} | {:^13} | {:^15} | {:^15} | {:^15} |",
            "NAME",
            name_width,
            "DOMAIN",
            domain_width,
            "CALLS",
            fmt::format("DURATION ({})", cfg.stats_summary_unit),
            fmt::format("AVERAGE ({})", cfg.stats_summary_unit),
            "PERCENT (INC)",
            fmt::format("MIN ({})", cfg.stats_summary_unit),
            fmt::format("MAX ({})", cfg.stats_summary_unit),
            "STDDEV");
        (*os.stream) << indent_v << _header << "\n" << std::flush;

        auto _div =
            fmt::format("|-{0:-^{1}}-|-{0:-^{2}}-|-{0:-^15}-|-{0:-^15}-|-{0:-^15}-|-{0:-^13}"
                        "-|-{0:-^15}-|-{0:-^15}-|-{0:-^15}-|",
                        "",
                        name_width,
                        domain_width);
        (*os.stream) << indent_v << _div << "\n" << std::flush;
    }

    for(const auto& [type, value] : _data.entries)
    {
        auto name        = type;
        auto duration_ns = value.get_sum();
        auto calls       = value.get_count();
        auto avg_ns      = value.get_mean();
        auto percent_v   = value.get_percent(_total_duration);
        auto percent     = std::to_string(percent_v);

        auto _row = std::string{};

        if(cfg.stats_summary_unit_value > 1)
        {
            auto _unit_div = static_cast<double>(cfg.stats_summary_unit_value);
            _row = fmt::format("{}| {:<{}} | {:<{}} | {:15} | {:15} | {:15.3e} | {:>13} | {:15} | "
                               "{:15} | {:15.3e} |",
                               indent_v,
                               name,
                               name_width,
                               get_domain_column_name(_cols.at(name)),
                               domain_width,
                               calls,
                               duration_ns / _unit_div,
                               avg_ns / _unit_div,
                               percent,
                               value.get_min() / _unit_div,
                               value.get_max() / _unit_div,
                               value.get_stddev() / _unit_div);
        }
        else
        {
            _row = fmt::format("{}| {:<{}} | {:<{}} | {:15} | {:15} | {:15.3e} | {:>13} | {:15} | "
                               "{:15} | {:15.3e} |",
                               indent_v,
                               name,
                               name_width,
                               get_domain_column_name(_cols.at(name)),
                               domain_width,
                               calls,
                               duration_ns,
                               avg_ns,
                               percent,
                               value.get_min(),
                               value.get_max(),
                               value.get_stddev());
        }

        (*os.stream) << _row << "\n" << std::flush;
    }

    (*os.stream) << "\n" << std::flush;
}
}  // namespace

void
generate_stats(const output_config& cfg,
               const metadata& /*tool_metadata*/,
               const domain_stats_vec_t& inp_data)
{
    auto data_v = inp_data;

    std::sort(data_v.begin(), data_v.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    output_stream _os     = get_output_stream(cfg, cfg.stats_summary_file, ".txt");
    auto          _indent = (_os.writes_to_file()) ? std::string_view{} : std::string_view{"    "};

    if(cfg.stats_summary_per_domain)
    {
        for(const auto& itr : data_v)
        {
            if(!itr.second) continue;

            auto _name = fmt::format("{} SUMMARY", get_domain_column_name(itr.first));
            auto _tmp  = domain_stats_vec_t{};
            _tmp.emplace_back(itr.first, itr.second);
            generate_stats(cfg, _os, _name, _tmp, _indent);
        }
    }

    if(!cfg.stats_summary_groups.empty())
    {
        auto domain_groups = std::vector<domain_stats_vec_t>{};
        for(const auto& itr : cfg.stats_summary_groups)
        {
            auto _names = std::vector<std::string>{};
            auto _tmp   = domain_stats_vec_t{};
            for(const auto& ditr : data_v)
            {
                auto _col_name = get_domain_column_name(ditr.first);

                if(std::regex_match(_col_name.data(), std::regex{itr}))
                {
                    if(!ditr.second) continue;
                    _names.emplace_back(_col_name);
                    _tmp.emplace_back(ditr.first, ditr.second);
                }
            }

            ROCP_CI_LOG_IF(WARNING, _names.empty())
                << "summary group regex '" << itr << "' matched with zero domain groups";

            auto _name = fmt::format("{} SUMMARY", fmt::join(_names.begin(), _names.end(), " + "));
            generate_stats(cfg, _os, _name, _tmp, _indent);
        }
    }

    if(cfg.stats_summary) generate_stats(cfg, _os, "SUMMARY", data_v, _indent);
}

stats_entry_t
generate_stats(const output_config& /* cfg*/,
               const metadata& /*tool_metadata*/,
               const generator<rocprofiler_tool_pc_sampling_host_trap_record_t>& /*data*/)
{
    // TODO:
    // 1. Implement serialization for PC sampling stats.
    //    The format differs significantly from tracing stats.
    // 2. Decide what is going to be part of the stats.
    //    Some basic information is already available in the tool_metadata.pc_sampling_stats.
    //    This contains the total number of valid VS invalid samples.
    return stats_entry_t{};
}

stats_entry_t
generate_stats(const output_config& /* cfg*/,
               const metadata& /*tool_metadata*/,
               const generator<rocprofiler_tool_pc_sampling_stochastic_record_t>& /*data*/)
{
    // TODO: sames TODOS from the function above applies here.
    return stats_entry_t{};
}

}  // namespace tool
}  // namespace rocprofiler
