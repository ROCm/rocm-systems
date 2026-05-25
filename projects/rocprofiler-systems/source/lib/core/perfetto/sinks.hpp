// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <sys/types.h>

namespace rocprofsys
{
class output_file_registry;

namespace core
{
// Live-mode sink: receives the engine's drained bytes for the current pid,
// runs the existing MPI gather (when ROCPROFSYS_USE_MPI is set), writes the
// concatenated trace to disk, and runs rocprof-sys-merge-output.sh on rank 0.
// Holds borrowed references to output_file_registry and a
// perfetto_output_error flag that the driver consults after finalize.
// Used concretely by live_perfetto_driver — not dispatched through the
// engine's sink boundary, so it is not a variant alternative below.
class live_fd_sink
{
public:
    live_fd_sink(bool* perfetto_output_error, output_file_registry& registry);

    // Defaulted move + deleted copy: the engine binds m_active_sink to the
    // address of a sink alternative inside a trace_sink variant; the variant
    // needs movability for construction, but the bound pointer would dangle
    // if the owner copied the sink after binding.
    live_fd_sink(live_fd_sink&&) noexcept            = default;
    live_fd_sink& operator=(live_fd_sink&&) noexcept = default;
    live_fd_sink(const live_fd_sink&)                = delete;
    live_fd_sink& operator=(const live_fd_sink&)     = delete;
    ~live_fd_sink()                                  = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

private:
    bool*                 m_output_error{ nullptr };
    output_file_registry* m_registry{ nullptr };
    std::vector<char>     m_bytes{};
    bool                  m_drained{ false };
};

// Cached-mode sink: writes per-pid bytes to one .proto file per pid.
// The parent_pid receives the default filename
// (config::get_perfetto_output_filename()); every other pid receives the
// suffix-stamped variant (get_perfetto_output_filename_with_suffix(pid)),
// matching the historical filename convention for cached output. Per-pid
// open failures are logged and the source is dropped; other pids continue.
class per_pid_file_sink
{
public:
    per_pid_file_sink(pid_t parent_pid, output_file_registry& registry);

    per_pid_file_sink(per_pid_file_sink&&) noexcept            = default;
    per_pid_file_sink& operator=(per_pid_file_sink&&) noexcept = default;
    per_pid_file_sink(const per_pid_file_sink&)                = delete;
    per_pid_file_sink& operator=(const per_pid_file_sink&)     = delete;
    ~per_pid_file_sink()                                       = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

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
class single_file_sink
{
public:
    explicit single_file_sink(output_file_registry& registry);

    single_file_sink(single_file_sink&&) noexcept            = default;
    single_file_sink& operator=(single_file_sink&&) noexcept = default;
    single_file_sink(const single_file_sink&)                = delete;
    single_file_sink& operator=(const single_file_sink&)     = delete;
    ~single_file_sink()                                      = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

private:
    output_file_registry*                  m_registry{ nullptr };
    std::vector<char>                      m_buffer{};
    std::unordered_map<int, std::uint32_t> m_source_seq_ids{};
    std::uint32_t                          m_next_seq_id{ 1 };
};

// Test-only sink: captures (source_id, bytes) tuples in arrival order so
// unit tests can assert the engine's drain contract without touching disk.
class recording_sink
{
public:
    using record_t = std::pair<int, std::vector<char>>;

    recording_sink()                                     = default;
    recording_sink(recording_sink&&) noexcept            = default;
    recording_sink& operator=(recording_sink&&) noexcept = default;
    recording_sink(const recording_sink&)                = delete;
    recording_sink& operator=(const recording_sink&)     = delete;
    ~recording_sink()                                    = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

    const std::vector<record_t>& records() const noexcept { return m_records; }
    bool                         finalized() const noexcept { return m_finalized; }

private:
    std::vector<record_t> m_records{};
    bool                  m_finalized{ false };
};

// Non-owning, type-erased view over any object exposing
//   void on_source_drained(int, std::vector<char>);
//   void finalize();
// Stored as a trace_sink variant alternative so test fixtures (e.g.
// exception-injecting doubles) can travel through the engine's sink
// boundary without inheriting from a common base. The view does NOT
// extend the target's lifetime — the caller keeps the target alive for
// as long as the view is reachable.
class polymorphic_sink_view
{
public:
    template <typename T, typename = std::enable_if_t<
                              !std::is_same_v<std::decay_t<T>, polymorphic_sink_view>>>
    explicit polymorphic_sink_view(T& target) noexcept
    : m_target{ &target }
    , m_drain_fn{ +[](void* p, int source_id, std::vector<char> bytes) {
        static_cast<T*>(p)->on_source_drained(source_id, std::move(bytes));
    } }
    , m_finalize_fn{ +[](void* p) { static_cast<T*>(p)->finalize(); } }
    {}

    // Rule of 5: the templated forwarding ctor disables implicit copy/move
    // unless we declare them explicitly. All members are trivially copyable
    // and the view is non-owning, so defaulted copy/move is the correct
    // semantics — defaulting makes the intent visible at the type level
    // rather than relying on compiler heuristics.
    polymorphic_sink_view(const polymorphic_sink_view&)            = default;
    polymorphic_sink_view(polymorphic_sink_view&&)                 = default;
    polymorphic_sink_view& operator=(const polymorphic_sink_view&) = default;
    polymorphic_sink_view& operator=(polymorphic_sink_view&&)      = default;
    ~polymorphic_sink_view()                                       = default;

    void on_source_drained(int source_id, std::vector<char> bytes)
    {
        m_drain_fn(m_target, source_id, std::move(bytes));
    }
    void finalize() { m_finalize_fn(m_target); }

private:
    using drain_fn_t    = void (*)(void*, int, std::vector<char>);
    using finalize_fn_t = void (*)(void*);

    // Non-owning: the caller keeps the target alive for as long as the view
    // is reachable. Stored as void* because the type was erased at ctor.
    void*         m_target{ nullptr };
    drain_fn_t    m_drain_fn{ nullptr };
    finalize_fn_t m_finalize_fn{ nullptr };
};

// Engine's sink boundary. Cached-mode trace bytes are dispatched to one of
// these alternatives via std::visit. live_fd_sink is intentionally excluded:
// the live driver owns its sink concretely and does not route through the
// engine. polymorphic_sink_view lets tests inject arbitrary fixtures.
using trace_sink = std::variant<per_pid_file_sink, single_file_sink, recording_sink,
                                polymorphic_sink_view>;
}  // namespace core
}  // namespace rocprofsys
