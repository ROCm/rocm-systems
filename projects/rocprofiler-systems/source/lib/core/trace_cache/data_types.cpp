// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/data_types.hpp"

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <utility>

namespace rocprofsys::trace_cache::data
{
namespace
{
template <typename Predicate>
[[nodiscard]] enabled_formats_t
filter_formats(const std::vector<format_t>& src, Predicate&& pred)
{
    std::vector<format_t> kept;
    kept.reserve(src.size());
    for(const auto& fmt : src)
    {
        if(fmt.enabled && pred(fmt)) kept.push_back(fmt);
    }
    return enabled_formats_t{ std::move(kept) };
}

template <typename Predicate>
[[nodiscard]] std::string
join_names(const std::vector<format_t>& formats, Predicate&& pred)
{
    std::string out;
    for(const auto& fmt : formats)
    {
        if(!fmt.enabled || !pred(fmt)) continue;
        if(!out.empty()) out += ", ";
        out += fmt.name;
    }
    return out;
}

constexpr auto any_format      = [](const format_t&) { return true; };
constexpr auto parallel_pred   = [](const format_t& f) { return f.process_parallel; };
constexpr auto sequential_pred = [](const format_t& f) { return !f.process_parallel; };
}  // namespace

enabled_formats_t::enabled_formats_t()
: formats{ { format_kind::rocpd, true, get_use_rocpd(), "rocpd" },
           { format_kind::perfetto, false, get_caching_perfetto(), "perfetto" },
           { format_kind::unified_memory, false, get_use_unified_memory_profiling(),
             "unified_memory" } }
{}

enabled_formats_t::enabled_formats_t(std::vector<format_t> _formats) noexcept
: formats(std::move(_formats))
{}

void
enabled_formats_t::print() const
{
    if(std::none_of(formats.begin(), formats.end(),
                    [](const auto& f) { return f.enabled; }))
        return;

    LOG_INFO("Generating [{}] format(s) with collected data from trace cache. "
             "This may take a while..",
             names().c_str());

    if(has_parallel_formats())
        LOG_INFO("  - Using parallel processing for: {}",
                 join_names(formats, parallel_pred));

    if(has_sequential_formats())
        LOG_INFO("  - Using sequential processing for: {}",
                 join_names(formats, sequential_pred));
}

bool
enabled_formats_t::has_parallel_formats() const
{
    return std::any_of(formats.begin(), formats.end(),
                       [](const auto& f) { return f.enabled && f.process_parallel; });
}

bool
enabled_formats_t::has_sequential_formats() const
{
    return std::any_of(formats.begin(), formats.end(),
                       [](const auto& f) { return f.enabled && !f.process_parallel; });
}

enabled_formats_t
enabled_formats_t::get_parallel_formats() const
{
    return filter_formats(formats, parallel_pred);
}

enabled_formats_t
enabled_formats_t::get_sequential_formats() const
{
    return filter_formats(formats, sequential_pred);
}

bool
enabled_formats_t::is_rocpd_enabled() const
{
    auto it = std::find_if(formats.begin(), formats.end(),
                           [](const auto& f) { return f.kind == format_kind::rocpd; });
    return it != formats.end() && it->enabled;
}

bool
enabled_formats_t::is_perfetto_enabled() const
{
    auto it = std::find_if(formats.begin(), formats.end(),
                           [](const auto& f) { return f.kind == format_kind::perfetto; });
    return it != formats.end() && it->enabled;
}

bool
enabled_formats_t::is_unified_memory_enabled() const
{
    auto it = std::find_if(formats.begin(), formats.end(), [](const auto& f) {
        return f.kind == format_kind::unified_memory;
    });
    return it != formats.end() && it->enabled;
}

std::string
enabled_formats_t::names() const
{
    return join_names(formats, any_format);
}

processor_config_t::processor_config_t(
    pid_t pid, pid_t ppid, std::shared_ptr<metadata_registry> metadata_registry_ptr,
    std::shared_ptr<agent_manager> agent_manager_ptr)
: _pid(pid)
, _ppid(ppid)
, _metadata_registry(std::move(metadata_registry_ptr))
, _agent_manager(std::move(agent_manager_ptr))
{}

}  // namespace rocprofsys::trace_cache::data
