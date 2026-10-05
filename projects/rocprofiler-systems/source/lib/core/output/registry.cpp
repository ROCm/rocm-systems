// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/output_summary.hpp"

#include "logger/debug.hpp"

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <system_error>
#include <utility>

namespace rocprofsys::output
{

namespace
{
[[nodiscard]] std::uint64_t
get_file_size(const std::string& path)
{
    std::error_code ec;
    const auto      size = std::filesystem::file_size(path, ec);
    if(ec)
    {
        LOG_WARNING("registry: failed to read size of '{}' ({}); reporting size as 0",
                    path, ec.message());
        return 0;
    }
    return static_cast<std::uint64_t>(size);
}
}  // namespace

registry&
registry::instance()
{
    static registry inst{};
    return inst;
}

void
registry::register_file(std::string path, output_format format, std::optional<pid_t> pid)
{
    artifact entry{};
    entry.pid        = pid.value_or(getpid());
    entry.size_bytes = get_file_size(path);
    entry.path       = std::move(path);
    entry.format     = format;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(std::move(entry));
}

void
registry::record_process(process_metadata meta)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto [it, inserted] = m_processes.try_emplace(meta.pid, meta);
    if(!inserted)
    {
        if(meta.ppid != NO_PID) it->second.ppid = meta.ppid;
        if(!meta.command.empty()) it->second.command = std::move(meta.command);
    }
}

std::vector<artifact>
registry::rows() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_files;
}

std::vector<process_metadata>
registry::processes() const
{
    std::lock_guard<std::mutex>   lock(m_mutex);
    std::vector<process_metadata> result;
    result.reserve(m_processes.size());
    for(const auto& [pid, meta] : m_processes)
        result.push_back(meta);
    return result;
}

void
registry::start_new_session()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
    m_processes.clear();
}

}  // namespace rocprofsys::output
