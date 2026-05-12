// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys
{

namespace output
{
struct process_metadata;
}

struct output_file
{
    std::string                   label;
    std::string                   path;
    std::string                   viewer;
    pid_t                         pid{ -1 };
    std::optional<std::uintmax_t> size_bytes{};
    std::uint64_t                 session_id{ 0 };
};

enum class output_format
{
    perfetto,
    rocpd,
    json,
    text,
    causal_json,
    causal_text
};

// Thread-safe registry of output files generated during profiling.
class output_file_registry
{
public:
    // Resolved to getpid() inside register_file when used.
    static constexpr pid_t DEFAULT_PID = -1;

    void register_file(std::string path, output_format format, pid_t pid = DEFAULT_PID);
    void register_file(std::string path, output_format format, std::string component_name,
                       pid_t pid = DEFAULT_PID);

    // Filtered to the current session. Older rows stay in internal
    // storage so an in-flight prior-session registration can still
    // land safely; they are excluded at read time.
    [[nodiscard]] std::vector<output_file>              rows() const;
    [[nodiscard]] std::vector<output::process_metadata> processes() const;

    // Upsert by pid within the current session.
    void record_process(output::process_metadata meta);

    // Increments the session id and compacts rows + processes from
    // prior sessions. Race-safe against concurrent register_file:
    // both share m_mutex, so an in-flight registration lands either
    // in the prior session (and is then compacted away) or in the
    // new one — never torn.
    [[nodiscard]] std::uint64_t bump_session();

    void print_summary() const;

private:
    static output_file make_entry(std::string path, output_format format,
                                  const std::string& component_name = {});

    void push_entry(output_file&& entry, pid_t pid);

    struct process_record
    {
        std::uint64_t                             session_id;
        std::shared_ptr<output::process_metadata> meta;
    };

    mutable std::mutex          m_mutex;
    std::vector<output_file>    m_files;
    std::vector<process_record> m_processes;
    std::uint64_t               m_session_id{ 1 };
};

// Function-local-static singleton. The rocprofiler-sdk attach path
// crosses a C-API boundary that cannot carry a reference parameter,
// so the registry has to be reachable globally. Reset-at-attach is
// implemented via session-id versioning (see bump_session()), not a
// clear, to be race-safe against in-flight prior-finalize writes.
output_file_registry&
registry();

}  // namespace rocprofsys
