// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output_file_registry.hpp"

#include "logger/debug.hpp"
#include "output/text_layout.hpp"

#include <spdlog/fmt/fmt.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <utility>

namespace rocprofsys
{

output_file
output_file_registry::make_entry(std::string path, output_format format,
                                 const std::string& component_name)
{
    switch(format)
    {
        case output_format::perfetto:
            return { "Perfetto trace", std::move(path),
                     "Open in https://ui.perfetto.dev" };
        case output_format::rocpd:
            return { "RocPD database", std::move(path),
                     "sqlite3, AMD Visualizer (OPTIQ), or rocprofiler-sdk provided rocpd "
                     "Python module for conversion to other formats" };
        case output_format::json:
            return {
                component_name.empty() ? "JSON output"
                                       : fmt::format("JSON ({})", component_name),
                path, fmt::format("jq . {}", output::escape_for_shell_single_quotes(path))
            };
        case output_format::text:
            return {
                component_name.empty() ? "Text profile"
                                       : fmt::format("Profile ({})", component_name),
                path, fmt::format("cat {}", output::escape_for_shell_single_quotes(path))
            };
        case output_format::causal_json:
            return { "Causal profile (JSON)", path,
                     fmt::format("jq . {}",
                                 output::escape_for_shell_single_quotes(path)) };
        case output_format::causal_text:
            return { "Causal profile (text)", path,
                     fmt::format("cat {}",
                                 output::escape_for_shell_single_quotes(path)) };
    }
    return { "Unknown", std::move(path), "" };
}

namespace
{
std::optional<std::uintmax_t>
try_file_size(const std::string& path)
{
    std::error_code ec;
    auto            size = std::filesystem::file_size(path, ec);
    if(ec)
    {
        LOG_WARNING(
            "output_file_registry: failed to read size of '{}' ({}); row will render "
            "with size '?'",
            path, ec.message());
        return std::nullopt;
    }
    return size;
}
}  // namespace

void
output_file_registry::push_entry(output_file&& entry, std::optional<pid_t> pid)
{
    // stat() runs before the lock so filesystem latency never blocks
    // concurrent register_file calls on the registry mutex.
    entry.pid        = pid.value_or(getpid());
    entry.size_bytes = try_file_size(entry.path);

    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back({ m_session_id, std::move(entry) });
}

void
output_file_registry::register_file(std::string path, output_format format,
                                    std::optional<pid_t> pid)
{
    push_entry(make_entry(std::move(path), format), pid);
}

void
output_file_registry::register_file(std::string path, output_format format,
                                    const std::string&   component_name,
                                    std::optional<pid_t> pid)
{
    push_entry(make_entry(std::move(path), format, component_name), pid);
}

std::vector<output_file>
output_file_registry::rows() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto                  current = m_session_id;
    std::vector<output_file>    out;
    out.reserve(m_files.size());
    for(const auto& v : m_files)
    {
        if(v.session_id == current) out.push_back(v.value);
    }
    return out;
}

void
output_file_registry::record_process(output::process_metadata meta)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for(auto& rec : m_processes)
    {
        if(rec.session_id == m_session_id && rec.value.pid == meta.pid)
        {
            rec.value = std::move(meta);
            return;
        }
    }
    m_processes.push_back({ m_session_id, std::move(meta) });
}

std::vector<output::process_metadata>
output_file_registry::processes() const
{
    std::lock_guard<std::mutex>           lock(m_mutex);
    const auto                            current = m_session_id;
    std::vector<output::process_metadata> out;
    out.reserve(m_processes.size());
    for(const auto& v : m_processes)
    {
        if(v.session_id == current) out.push_back(v.value);
    }
    return out;
}

std::uint64_t
output_file_registry::bump_session()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_session_id;
    const auto current           = m_session_id;
    auto       not_current_files = [current](const versioned<output_file>& v) {
        return v.session_id != current;
    };
    auto not_current_procs = [current](const versioned<output::process_metadata>& v) {
        return v.session_id != current;
    };
    m_files.erase(std::remove_if(m_files.begin(), m_files.end(), not_current_files),
                  m_files.end());
    m_processes.erase(
        std::remove_if(m_processes.begin(), m_processes.end(), not_current_procs),
        m_processes.end());
    return m_session_id;
}

output_file_registry&
output_file_registry::instance_for_top_level_attach_finalize()
{
    static output_file_registry s_instance{};
    return s_instance;
}

}  // namespace rocprofsys
