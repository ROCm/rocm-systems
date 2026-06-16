// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_metadata.hpp"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys
{

enum class output_format
{
    perfetto,
    rocpd,
    json,
    text,
    causal_json,
    causal_text
};

struct output_file
{
    std::string                   path;
    pid_t                         pid{ -1 };
    std::optional<std::uintmax_t> size_bytes{};
    output_format                 format{ output_format::perfetto };
};

// One registry per process: one output dir, one Output Summary, one
// finalize. Process-singleton, not SDK-session-scoped, because the
// rocprofiler-sdk attach/detach protocol can leak its client_data
// blob and there is no natural shared owner across the registration
// sites (perfetto, rocpd, library finalize).
//
// Access discipline: only the top-level attach/finalize entry points
// call the singleton accessor; downstream consumers take a reference
// so tests can substitute their own instance. Attach/detach resets go
// through bump_session() — see its declaration for the race argument.
class output_file_registry
{
public:
    // Process-singleton accessor. The ugly name documents the
    // constraint: only call this from a top-level attach or finalize
    // entry point.
    [[nodiscard]] static output_file_registry& instance_for_top_level_attach_finalize();

    output_file_registry() = default;

    void register_file(std::string path, output_format format,
                       std::optional<pid_t> pid = std::nullopt);

    // Filtered to the current session. Older records stay in internal
    // storage so an in-flight prior-session registration can still
    // land safely; they are excluded at read time.
    [[nodiscard]] std::vector<output_file>              rows() const;
    [[nodiscard]] std::vector<output::process_metadata> processes() const;

    // Upsert by pid within the current session.
    void record_process(output::process_metadata meta);

    // Total GPU devices on the node, used by the Output Summary header
    // to qualify the utilized-GPU set as "(all)". Unknown until set.
    void                                     set_node_gpu_count(std::size_t count);
    [[nodiscard]] std::optional<std::size_t> node_gpu_count() const;

    // Increments the session id and compacts records from prior
    // sessions. Race-safe against concurrent register_file: both
    // share m_mutex, so an in-flight registration lands either in
    // the prior session (and is then compacted away) or in the new
    // one — never torn.
    [[nodiscard]] std::uint64_t bump_session();

private:
    static output_file make_entry(std::string path, output_format format);

    void push_entry(output_file&& entry, std::optional<pid_t> pid);

    // Internal versioning wrapper. Keeps session bookkeeping off the
    // public value types so consumers see only what they need.
    template <typename T>
    struct versioned
    {
        std::uint64_t session_id{ 0 };
        T             value{};
    };

    mutable std::mutex                               m_mutex;
    std::vector<versioned<output_file>>              m_files;
    std::vector<versioned<output::process_metadata>> m_processes;
    std::uint64_t                                    m_session_id{ 1 };
    std::optional<std::size_t>                       m_node_gpu_count{};
};

}  // namespace rocprofsys
