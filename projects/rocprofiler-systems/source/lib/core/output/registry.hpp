// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"

#include <sys/types.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::output
{
template <typename EntryType>
struct session_entry
{
    std::uint64_t session_id{ 0 };
    EntryType     value{};
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

    // Attach/detach lifecycle: isolates a new session's rows from a prior
    // session's stale registrations.
    std::uint64_t start_new_session();

private:
    registry() = default;

    mutable std::mutex                                         m_mutex;
    std::vector<session_entry<artifact>>                       m_files;
    std::unordered_map<pid_t, session_entry<process_metadata>> m_processes;
    std::uint64_t                                              m_session_id{ 1 };
};

}  // namespace rocprofsys::output
