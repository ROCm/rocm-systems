// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::output
{

// Sentinel for "no such pid" — an absent parent (root process) or a not-yet
// assigned pid field.
inline constexpr pid_t NO_PID = -1;

struct process_metadata
{
    pid_t       pid{ NO_PID };
    pid_t       ppid{ NO_PID };
    std::string command;
};

enum class output_format
{
    perfetto,
    rocpd,
    json,
    text
};

struct artifact
{
    std::string   path;
    pid_t         pid{ NO_PID };
    std::uint64_t size_bytes{ 0 };
    output_format format{ output_format::perfetto };
};

[[nodiscard]] inline std::uint64_t
sum_sizes(std::span<const artifact> items)
{
    return std::accumulate(
        items.begin(), items.end(), std::uint64_t{ 0 },
        [](std::uint64_t acc, const artifact& item) { return acc + item.size_bytes; });
}

struct process_node
{
    process_metadata          meta;
    std::vector<artifact>     rows;
    std::vector<process_node> children;
};

/*
 * Keep the track of the missing or excluded pids while
 * building the process tree for rendering
 *
 * missing_metadata_pids - PIDs seen in registered rows/process records
 * but missing their own metadata record
 *
 * cyclic_ppid_pids - PIDs excluded from every root's tree because their
 * ppid chain forms a cycle (corrupted metadata)
 */
struct process_tree_diagnostics
{
    std::vector<pid_t> missing_metadata_pids;
    std::vector<pid_t> cyclic_ppid_pids;
};

class process_tree
{
public:
    // Builds a tree from `rows`/`processes` grouped by pid/ppid
    process_tree(std::span<const artifact>         rows,
                 std::span<const process_metadata> processes);

    [[nodiscard]] const std::vector<process_node>& roots() const noexcept
    {
        return m_roots;
    }

    [[nodiscard]] const process_tree_diagnostics& diagnostics() const noexcept
    {
        return m_diagnostics;
    }

private:
    std::vector<process_node> m_roots;
    process_tree_diagnostics  m_diagnostics;
};

class registry
{
public:
    [[nodiscard]] static registry& instance();

    // `pid` defaults to the calling process's own pid (via getpid()) when
    // not specified.
    void register_file(std::string path, output_format format,
                       std::optional<pid_t> pid = std::nullopt);

    void record_process(process_metadata meta);

    [[nodiscard]] std::vector<artifact>         rows() const;
    [[nodiscard]] std::vector<process_metadata> processes() const;

    // Attach/detach lifecycle: clears a prior session's registrations
    // before a new one starts.
    void start_new_session();

private:
    registry() = default;

    mutable std::mutex                          m_mutex;
    std::vector<artifact>                       m_files;
    std::unordered_map<pid_t, process_metadata> m_processes;
};

struct run_metadata
{
    std::string              run_label;
    std::chrono::nanoseconds duration{ 0 };
    std::string              output_dir_abs;

    [[nodiscard]] static run_metadata capture(
        std::chrono::steady_clock::time_point load_baseline);
};

[[nodiscard]] std::string
summarize_command(std::string_view command);

// Writes nothing at all if `tree` has no rows — there is no output to summarize.
void
write_summary(std::ostream& os, const process_tree& tree, const run_metadata& meta);

}  // namespace rocprofsys::output
