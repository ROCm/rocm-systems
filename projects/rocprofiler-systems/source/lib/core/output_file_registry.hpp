// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_metadata.hpp"

#include <sys/types.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys
{

struct output_file
{
    std::string                   label;
    std::string                   path;
    std::string                   viewer;
    pid_t                         pid{ -1 };
    std::optional<std::uintmax_t> size_bytes{};
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

// One registry per process. Cardinality matches the user-facing model:
// one output directory, one Output Summary block, one finalize per
// process. Multiple registries would mean files registered in one are
// invisible to the other's summary.
//
// Registration sites span perfetto.cpp, perfetto_processor.cpp,
// rocpd_processor.cpp, and library.cpp finalize — there is no natural
// shared owner across them. The rocprofiler-sdk client_data blob is
// one candidate, but its lifetime is governed by the SDK's
// attach/detach protocol, which does not invoke tool_fini on
// attach-only runs and leaks the blob accordingly. A per-tool-session
// owner would inherit that leak. Process-singleton storage sidesteps
// the lifecycle question — the registry lives for the process,
// regardless of which SDK sessions come and go inside it.
//
// Attach/detach resets are handled by bump_session() — see its
// comment for the race-safety argument.
//
// Access discipline: call instance() only from top-level
// finalize/attach entry points. Downstream consumers (processors,
// post-processing pipelines) receive the registry by reference; this
// preserves the test seam where tests construct their own
// output_file_registry instances and pass them to the consumer under
// test.
class output_file_registry
{
public:
    // Process-singleton accessor. See class-level comment.
    [[nodiscard]] static output_file_registry& instance();

    output_file_registry() = default;

    void register_file(std::string path, output_format format,
                       std::optional<pid_t> pid = std::nullopt);
    void register_file(std::string path, output_format format, std::string component_name,
                       std::optional<pid_t> pid = std::nullopt);

    // Filtered to the current session. Older records stay in internal
    // storage so an in-flight prior-session registration can still
    // land safely; they are excluded at read time.
    [[nodiscard]] std::vector<output_file>              rows() const;
    [[nodiscard]] std::vector<output::process_metadata> processes() const;

    // Upsert by pid within the current session.
    void record_process(output::process_metadata meta);

    // Increments the session id and compacts records from prior
    // sessions. Race-safe against concurrent register_file: both
    // share m_mutex, so an in-flight registration lands either in
    // the prior session (and is then compacted away) or in the new
    // one — never torn.
    [[nodiscard]] std::uint64_t bump_session();

private:
    static output_file make_entry(std::string path, output_format format,
                                  const std::string& component_name = {});

    void push_entry(output_file&& entry, std::optional<pid_t> pid);

    // Internal versioning wrapper. Keeps session bookkeeping off the
    // public value types so consumers see only what they need.
    template <typename T>
    struct versioned
    {
        std::uint64_t session_id;
        T             value;
    };

    mutable std::mutex                               m_mutex;
    std::vector<versioned<output_file>>              m_files;
    std::vector<versioned<output::process_metadata>> m_processes;
    std::uint64_t                                    m_session_id{ 1 };
};

}  // namespace rocprofsys
