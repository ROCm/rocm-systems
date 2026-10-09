// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::output
{

// Sentinel for "no such pid" — an absent parent (root process) or a not-yet
// assigned pid field.
inline constexpr pid_t k_no_pid = -1;

struct process_metadata
{
    pid_t       pid{ k_no_pid };
    pid_t       ppid{ k_no_pid };
    std::string command;
};

struct artifact
{
    std::string   path;
    pid_t         pid{ k_no_pid };
    std::uint64_t size_bytes{ 0 };
};

class registry
{
public:
    [[nodiscard]] static registry& instance();

    void register_file(std::string path);

    void record_process(process_metadata meta);

    void start_new_session();

    [[nodiscard]] std::string format_summary() const;

private:
    registry() = default;

    [[nodiscard]] std::vector<artifact>         rows() const;
    [[nodiscard]] std::vector<process_metadata> processes() const;

    mutable std::mutex                          m_mutex;
    std::vector<artifact>                       m_files;
    std::unordered_map<pid_t, process_metadata> m_processes;
    std::chrono::steady_clock::time_point       m_start_time{
        std::chrono::steady_clock::now()
    };
};

struct run_metadata
{
    std::string              run_label;
    std::chrono::nanoseconds duration{ 0 };

    explicit run_metadata(std::chrono::nanoseconds elapsed);
};

}  // namespace rocprofsys::output
