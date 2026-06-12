// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
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
// Live-mode sink: writes the engine's drained bytes to the configured
// per-rank output filename. Used by live_perfetto_driver for the per-rank
// file under the per_process_only and full layouts; cross-rank merging
// stays in the driver and routes through single_file_sink. Not a variant
// alternative below — the live path owns its sink concretely and does
// not need polymorphism at that boundary.
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
// Each source_id is assigned a base offset on first arrival, and every
// TracePacket from that source has its trusted_packet_sequence_id shifted
// by that base before being appended to the in-memory buffer. The offset
// (not replace) semantics preserve Perfetto's per-seq_id interned-data
// namespacing: collapsing distinct input seq_ids into one output value
// would merge independent iid namespaces and silently misresolve later
// definitions. The cached interceptor exfiltrates packets before the
// tracing service stamps seq_ids so its inputs all carry the SDK's
// placeholder (1) — the offset still produces disjoint output ranges
// across sources. Live-mode MPI inputs carry pre-stamped seq_ids (one
// per producer thread) and the offset preserves those relative ids
// inside each source's window. finalize() writes the buffer.
class single_file_sink
{
public:
    // output_filename_override empty -> resolve via
    // config::get_perfetto_output_filename() at finalize time. Set to a concrete path
    // when the caller wants to write to a different location than the configured base
    // (e.g. the live "full" layout writes its cross-rank merged trace to
    // `<dir>/merged.proto` alongside per-rank files).
    explicit single_file_sink(output_file_registry& registry,
                              std::string           output_filename_override = {});

    single_file_sink(single_file_sink&&) noexcept            = default;
    single_file_sink& operator=(single_file_sink&&) noexcept = default;
    single_file_sink(const single_file_sink&)                = delete;
    single_file_sink& operator=(const single_file_sink&)     = delete;
    ~single_file_sink()                                      = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

    // Switch the sink into "append with file lock" mode for cross-process
    // aggregation: finalize() opens the target with O_APPEND, takes an
    // exclusive flock for the duration of the write, releases, and closes.
    // `seq_id_base` shifts this process's seq_id namespace so concurrent
    // appenders (one rocprof-sys instance per launcher rank) do not collide
    // on trusted_packet_sequence_id. Typically derived from
    // mpi::rank_from_env() multiplied by a per-rank stride. Append mode is
    // used by the live + cached MPI merge paths so the same mechanism
    // works whether rocprof-sys is linked against MPI or only built with
    // MPI headers (and the workload is launched under mpiexec/srun).
    void set_append_mode(std::uint32_t seq_id_base) noexcept;

    // Test-only inspector: exposes the accumulated rewritten bytes before
    // finalize() so unit tests can assert per-source seq_id offsetting
    // without depending on the config singleton + filesystem path that
    // finalize() would resolve. Mirrors recording_sink::records().
    [[nodiscard]] const std::vector<char>& buffer_for_testing() const noexcept
    {
        return m_buffer;
    }

private:
    // Width of each source's seq_id sub-range inside this sink's output
    // namespace. Source N's effective seq_ids land in
    // [base_N, base_N + PER_SOURCE_SEQ_ID_BASE_STRIDE). 1<<16 comfortably
    // exceeds the few-hundred internal seq_ids any single Perfetto SDK
    // TracingSession produces, while still letting many sources fit
    // inside the 1<<20 per-rank window the MPI merge path uses.
    static constexpr std::uint32_t PER_SOURCE_SEQ_ID_BASE_STRIDE = 1u << 16;

    output_file_registry*                  m_registry{ nullptr };
    std::string                            m_output_filename_override{};
    std::vector<char>                      m_buffer{};
    std::unordered_map<int, std::uint32_t> m_source_seq_id_bases{};
    std::uint32_t                          m_next_source_base{ 1 };
    bool                                   m_append_mode{ false };
};

// Composite sink that fans every drained source out to both wrapped sinks.
// Used by the cached `full` layout where each rank must write per-pid files
// (per_pid_file_sink) AND contribute to the cross-rank merged file
// (single_file_sink in append-with-flock mode). The per-drain bytes are
// copied once so each sub-sink consumes its own copy; cached drains run at
// finalize time (not on the hot path) so the copy is acceptable.
class tee_sink
{
public:
    tee_sink(per_pid_file_sink per_pid, single_file_sink single_file);

    tee_sink(tee_sink&&) noexcept            = default;
    tee_sink& operator=(tee_sink&&) noexcept = default;
    tee_sink(const tee_sink&)                = delete;
    tee_sink& operator=(const tee_sink&)     = delete;
    ~tee_sink()                              = default;

    void on_source_drained(int source_id, std::vector<char> bytes);
    void finalize();

private:
    per_pid_file_sink m_per_pid;
    single_file_sink  m_single_file;
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
class polymorphic_sink_view;

template <typename T>
concept polymorphic_sink_target =
    !std::same_as<std::remove_cvref_t<T>, polymorphic_sink_view> &&
    requires(T& target, int source_id, std::vector<char> bytes) {
        { target.on_source_drained(source_id, std::move(bytes)) } -> std::same_as<void>;
        { target.finalize() } -> std::same_as<void>;
    };

class polymorphic_sink_view
{
public:
    template <polymorphic_sink_target T>
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
                                polymorphic_sink_view, tee_sink>;
}  // namespace core
}  // namespace rocprofsys
