// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>

namespace tim
{
class manager;
}

namespace rocprofsys
{
class output_file_registry;

namespace core
{
// Polymorphic destination for trace bytes drained by the engine. The engine
// invokes on_source_drained() once per emitting source and finalize() once
// after all sources have drained (D1).
class trace_sink
{
public:
    virtual ~trace_sink() = default;

    virtual void on_source_drained(int source_id, std::vector<char> bytes) = 0;
    virtual void finalize()                                                = 0;
};

// Live-mode sink: receives the engine's drained bytes for the current pid,
// runs the existing MPI gather (when ROCPROFSYS_USE_MPI is set), writes the
// concatenated trace to disk, and runs rocprof-sys-merge-output.sh on rank 0.
// Holds borrowed references to the legacy plumbing
// (tim::manager / output_file_registry / perfetto_output_error flag) so the
// finalize body is a near-literal lift of perfetto.cpp:post_process().
class live_fd_sink final : public trace_sink
{
public:
    live_fd_sink(tim::manager*         timemory_manager,
                 bool*                 perfetto_output_error,
                 output_file_registry& registry);

    void on_source_drained(int source_id, std::vector<char> bytes) override;
    void finalize() override;

private:
    tim::manager*         m_manager{ nullptr };
    bool*                 m_output_error{ nullptr };
    output_file_registry* m_registry{ nullptr };
    std::vector<char>     m_bytes{};
    bool                  m_drained{ false };
};

// Cached-mode sink: writes per-pid bytes to one .proto file per pid.
// The parent_pid receives the default filename
// (config::get_perfetto_output_filename()); every other pid receives the
// suffix-stamped variant (get_perfetto_output_filename_with_suffix(pid))
// — same contract perfetto_processor.cpp:524-527 has applied for the
// cached path. Per-pid open failures are logged and the source is
// dropped; other pids continue (RF5).
class per_pid_file_sink final : public trace_sink
{
public:
    per_pid_file_sink(pid_t parent_pid, output_file_registry& registry);

    void on_source_drained(int source_id, std::vector<char> bytes) override;
    void finalize() override;

private:
    pid_t                 m_parent_pid{ 0 };
    output_file_registry* m_registry{ nullptr };
};

// Cached-mode sink: concatenates per-pid bytes into one .proto file.
//
// The cached_interceptor exfiltrates TracePackets from the SDK before the
// tracing service stamps trusted_packet_sequence_id, so every packet the
// engine drains arrives with the SDK's placeholder value (1). Concatenating
// per-pid streams that all claim seq_id=1 would make Perfetto interpret
// every packet as belonging to a single producer and mis-apply incremental
// state (TracePacketDefaults, default tracks) across pid boundaries. This
// sink stands in for the missing service-side stamping: each source_id is
// assigned a sequential trusted_packet_sequence_id on first arrival, and
// every TracePacket from that source is rewritten with that id before
// being appended to the in-memory buffer. finalize() writes the buffer.
class single_file_sink final : public trace_sink
{
public:
    explicit single_file_sink(output_file_registry& registry);

    void on_source_drained(int source_id, std::vector<char> bytes) override;
    void finalize() override;

private:
    output_file_registry*                 m_registry{ nullptr };
    std::vector<char>                     m_buffer{};
    std::unordered_map<int, std::uint32_t> m_source_seq_ids{};
    std::uint32_t                         m_next_seq_id{ 1 };
};

// Test-only sink: captures (source_id, bytes) tuples in arrival order so
// unit tests can assert the engine's drain contract without touching disk.
class recording_sink final : public trace_sink
{
public:
    using record_t = std::pair<int, std::vector<char>>;

    void on_source_drained(int source_id, std::vector<char> bytes) override;
    void finalize() override;

    const std::vector<record_t>& records() const noexcept { return m_records; }
    bool                         finalized() const noexcept { return m_finalized; }

private:
    std::vector<record_t> m_records{};
    bool                  m_finalized{ false };
};
}  // namespace core
}  // namespace rocprofsys
