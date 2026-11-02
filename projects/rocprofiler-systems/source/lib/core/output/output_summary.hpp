// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::output
{

struct process_metadata
{
    pid_t       pid{};
    pid_t       ppid{};
    std::string command;
};

/**
 * Collects the output files and process metadata of a run and renders them as a
 * per-process tree at finalization. Thread-safe.
 */
class registry
{
public:
    [[nodiscard]] static registry& instance();

    void register_file(std::string path, pid_t pid);

    void record_process(process_metadata meta);

    void start_new_session();

    [[nodiscard]] std::string format_summary() const;

private:
    struct artifact
    {
        std::string   path;
        pid_t         pid{};
        std::uint64_t size_bytes{};
    };

    registry() = default;

    mutable std::mutex                                   m_mutex;
    std::vector<artifact>                                m_files;
    std::unordered_map<pid_t, process_metadata>          m_processes;
    std::optional<std::chrono::steady_clock::time_point> m_start_time;
};

}  // namespace rocprofsys::output
