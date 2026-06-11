// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

#include "sinks.hpp"

namespace perfetto
{
class TracingSession;
}

namespace rocprofsys::core
{
struct tracing_session_deleter
{
    void operator()(::perfetto::TracingSession* session) const noexcept;
};

using tracing_session_ptr =
    std::unique_ptr<::perfetto::TracingSession, tracing_session_deleter>;

// POD snapshot of the perfetto-relevant configuration. Built once at engine
// construction by build_engine_config_from_settings(); the engine never reads
// rocprofsys::config::* again after that. Tests construct it with literals to
// exercise the engine in isolation.
struct engine_config
{
    enum class fill_policy_t
    {
        discard,
        ring_buffer,
    };

    enum class backend_t
    {
        inprocess,
        system,
        all,
    };

    std::uint32_t            buffer_size_kb     = 0;
    std::uint32_t            shmem_size_hint_kb = 0;
    std::uint32_t            flush_period_ms    = 0;
    fill_policy_t            fill_policy        = fill_policy_t::discard;
    backend_t                backend            = backend_t::inprocess;
    std::vector<std::string> disabled_categories{};
    bool                     suppress_sdk_log_output = false;
};

// Reads the live config::get_perfetto_* getters once and returns a snapshot.
// Called once per process by whoever constructs the perfetto_engine.
engine_config
build_engine_config_from_settings();

// Owns Perfetto SDK init + the per-pid TracingSession map.
//
// Two modes:
// - live_fd: SDK writes to fd (optional) + internal buffer. stop() flushes
//   and halts; bytes are read and pushed to a sink by the orchestrator
//   (perfetto.cpp shim) because it owns the tmp-file vs ReadTraceBlocking
//   selection. No engine-side sink reference.
// - cached_interceptor: per-thread TLS interceptor copies packet bytes
//   keyed by the thread's emitting_pid; stop() drains those into the
//   bound sink via on_source_drained / finalize. Bytes never touch a fd.
class perfetto_engine
{
public:
    enum class mode
    {
        live_fd,
        cached_interceptor,
    };

    explicit perfetto_engine(engine_config cfg);
    ~perfetto_engine() noexcept;

    // Engine address must be stable for the full lifetime of any cached
    // session: cached_interceptor's ThreadLocalState caches the
    // perfetto_engine* at first emission per thread and dereferences it on
    // every later OnTracePacket. Allowing copy or move would let a caller
    // dangle that cached pointer. Owners hold the engine via unique_ptr.
    perfetto_engine(const perfetto_engine&)            = delete;
    perfetto_engine& operator=(const perfetto_engine&) = delete;
    perfetto_engine(perfetto_engine&&)                 = delete;
    perfetto_engine& operator=(perfetto_engine&&)      = delete;

    // Initialises the Perfetto SDK once per process via std::call_once.
    // Safe to call from every engine instance; only the first wins.
    void init_sdk();

    // Live-mode start. fd >= 0 streams output to the file descriptor in
    // addition to the SDK's internal buffer (legacy ROCPROFSYS_USE_TMP_FILES
    // path); fd == -1 disables on-disk capture. Calling start() while
    // running warns and replaces the existing session.
    void start(mode m, int fd);

    // Cached-mode start. Sink is bound for stop()-time drain via the
    // per-thread interceptor; emitting threads must have
    // set_emitting_pid(pid) called before their first TRACE_EVENT_*.
    void start(mode m, trace_sink& sink);

    // Flushes and stops the active session.
    // - live_fd:           flush + StopBlocking only (orchestrator drains).
    // - cached_interceptor: flush + StopBlocking, then drain per-pid
    //                       collected bytes through the bound sink
    //                       (on_source_drained per pid, then finalize();
    //                       first per-source exception is rethrown after
    //                       finalize per drain contract).
    // No-op when no session is active.
    void stop();

    // Reads the trace bytes from the session for the given pid via
    // ReadTraceBlocking. Returns an empty vector when no session exists.
    // Caller is responsible for disposing the session via destroy_session
    // afterwards if desired.
    [[nodiscard]] std::vector<char> read_trace(pid_t pid);

    // Destroys the per-pid session: equivalent to .reset() on the slot.
    // Use this when the session is genuinely done (e.g. post-stop cleanup).
    // Paired with forget_session() which DETACHES without destroying.
    void destroy_session(pid_t pid);

    // Detaches the engine's ownership of the per-pid session without
    // destroying the underlying TracingSession. Used by fork_gotcha in the
    // forked child to drop the inherited session pointer that the PARENT
    // process still owns; calling reset() in the child would corrupt the
    // parent's state. Paired with destroy_session() which DESTROYS.
    void forget_session(pid_t pid);

    // Whether a session is currently active.
    [[nodiscard]] bool is_running() const noexcept;

    // Pre-creates per-pid byte buffer slots so cached emission stays
    // lock-free on the hot path. MUST be called between start(cached_…)
    // and the first emit from any parser thread; callers know the full
    // pid set up front (it comes from the cache_manager's processor
    // configs). Subsequent collect_packet_bytes calls for unknown pids
    // are dropped with an error log.
    void preregister_pids(const std::vector<int>& source_pids);

    // Called by the cached-mode interceptor TLS to append per-pid bytes
    // during emission. Public so the TU-private interceptor inside the
    // .cpp can reach it without crossing the private-member boundary; not
    // intended for outside callers. noexcept because it runs inside the
    // Perfetto SDK callback frame, which is not exception-safe; internal
    // bad_alloc is caught and the packet is dropped with LOG_ERROR.
    void collect_packet_bytes(int pid, const void* data, std::size_t size) noexcept;

private:
    [[nodiscard]] bool is_system_backend() const noexcept;
    void               start_session(pid_t pid, int fd, mode m);

    engine_config m_cfg{};
    pid_t         m_active_pid{ 0 };
    bool          m_running{ false };
    mode          m_current_mode{ mode::live_fd };
    trace_sink*   m_active_sink{ nullptr };

    std::mutex m_sessions_mutex{};

    std::mutex                                 m_collector_mutex{};
    std::unordered_map<int, std::vector<char>> m_collected_bytes{};
    std::atomic<bool>                          m_collected_bytes_frozen{ false };
    std::atomic<std::size_t>                   m_dropped_packet_count{ 0 };

    static std::once_flag s_sdk_init_flag;

    // The header only needs the opaque TracingSession handle; TraceConfig and
    // SDK setup details stay local to engine.cpp while session ownership remains
    // explicit in the engine state.
    std::unordered_map<pid_t, tracing_session_ptr> m_sessions{};
};

// Thread-local pid tag consumed by the cached-mode interceptor TLS to key
// each thread's emissions to a pid. Tagging threads is the emitter's
// responsibility — call set_emitting_pid(pid) before the first
// TRACE_EVENT_* on the thread. Free functions because they operate on a
// TU-local thread_local; the class scope these previously lived under
// implied non-existent object state.
void
set_emitting_pid(int pid) noexcept;

int
get_emitting_pid() noexcept;
}  // namespace rocprofsys::core
