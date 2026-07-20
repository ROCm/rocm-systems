// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/output/registry.hpp"

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
    m_files.push_back({ m_session_id, std::move(entry) });
}

void
registry::record_process(process_metadata meta)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto                        it = m_processes.find(meta.pid);
    if(it == m_processes.end() || it->second.session_id != m_session_id)
    {
        // Capture the key before moving from meta: relying on the RHS of
        // `operator[]=` being sequenced before the LHS (so meta.pid would
        // still read correctly after the move) is a fragile guarantee that
        // only holds because pid_t is trivially copyable.
        const pid_t pid = meta.pid;
        m_processes[pid] =
            session_entry<process_metadata>{ m_session_id, std::move(meta) };
        return;
    }

    if(meta.ppid != NO_PID) it->second.value.ppid = meta.ppid;
    if(!meta.command.empty()) it->second.value.command = std::move(meta.command);
}

std::vector<artifact>
registry::rows() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<artifact>       result;
    result.reserve(m_files.size());
    for(const auto& entry : m_files)
        if(entry.session_id == m_session_id) result.push_back(entry.value);
    return result;
}

std::vector<process_metadata>
registry::processes() const
{
    std::lock_guard<std::mutex>   lock(m_mutex);
    std::vector<process_metadata> result;
    result.reserve(m_processes.size());
    for(const auto& [pid, entry] : m_processes)
        if(entry.session_id == m_session_id) result.push_back(entry.value);
    return result;
}

std::uint64_t
registry::start_new_session()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto                  ended_session = m_session_id;
    ++m_session_id;

    std::erase_if(m_files, [ended_session](const auto& entry) {
        return entry.session_id < ended_session;
    });
    std::erase_if(m_processes, [ended_session](const auto& kv) {
        return kv.second.session_id < ended_session;
    });

    return m_session_id;
}

}  // namespace rocprofsys::output
