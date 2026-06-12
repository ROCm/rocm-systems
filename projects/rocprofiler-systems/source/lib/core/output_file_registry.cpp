// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output_file_registry.hpp"

#include "logger/debug.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <utility>

namespace rocprofsys
{

output_file
output_file_registry::make_entry(std::string path, output_format format)
{
    output_file entry{};
    entry.path   = std::move(path);
    entry.format = format;
    return entry;
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
            // Merge, do not replace: a later sparser record (e.g. the
            // finalize-time self-registration, which knows pid/ppid/command
            // but not gpu_ids) must not erase richer fields a prior
            // post-processor record already supplied. Non-empty incoming
            // fields win; empty ones preserve the existing value.
            if(meta.ppid != -1) rec.value.ppid = meta.ppid;
            if(!meta.command.empty()) rec.value.command = std::move(meta.command);
            if(!meta.gpu_ids.empty()) rec.value.gpu_ids = std::move(meta.gpu_ids);
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

void
output_file_registry::set_node_gpu_count(std::size_t count)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_node_gpu_count = count;
}

std::optional<std::size_t>
output_file_registry::node_gpu_count() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_node_gpu_count;
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
