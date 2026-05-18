// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

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
