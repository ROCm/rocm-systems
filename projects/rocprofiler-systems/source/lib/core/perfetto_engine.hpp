// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sys/types.h>

namespace perfetto
{
class TracingSession;
}

namespace rocprofsys
{
namespace core
{
// POD snapshot of the perfetto-relevant configuration. Built once at engine
// construction by build_engine_config_from_settings(); the engine never reads
// rocprofsys::config::* again after that. Tests construct it with literals to
// exercise the engine in isolation (RQ3, D6).
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
};

// Reads the live config::get_perfetto_* getters once and returns a snapshot.
// Composition root calls this exactly once per process.
engine_config
build_engine_config_from_settings();

// Owns Perfetto SDK init + the per-pid TracingSession map for the live path.
// Slice B exposes only mode::live_fd; mode::cached_interceptor lands in
// slice C1.
//
// Live mode posture (slice B): the engine is stop-only — start() opens the
// session, stop() flushes + StopBlocking. The drain (read bytes + push
// through trace_sink) is orchestrated by the perfetto.cpp shim, which knows
// whether the legacy tmp-file path is in use and must read from disk
// instead of session->ReadTraceBlocking. Cached mode (slice C+) will drive
// drain through the sink directly because bytes come from per-thread
// interceptor TLS buffers, not a fd.
class perfetto_engine
{
public:
    enum class mode
    {
        live_fd,
        // cached_interceptor — added in slice C1
    };

    explicit perfetto_engine(engine_config cfg);
    ~perfetto_engine();

    perfetto_engine(const perfetto_engine&)            = delete;
    perfetto_engine& operator=(const perfetto_engine&) = delete;
    perfetto_engine(perfetto_engine&&)                 = delete;
    perfetto_engine& operator=(perfetto_engine&&)      = delete;

    // Initialises the Perfetto SDK once per process via std::call_once (D5).
    // Safe to call from every engine instance; only the first wins.
    void init_sdk();

    // Starts a tracing session in the requested mode. fd >= 0 instructs the
    // SDK to stream output to the file descriptor in addition to its
    // internal buffer (legacy ROCPROFSYS_USE_TMP_FILES path); fd == -1
    // disables on-disk capture. Calling start() while running warns and
    // replaces the existing session.
    void start(mode m, int fd);

    // Flushes and stops the active session. No-op when no session is active
    // (RF6). Bytes are not read or drained here — see the live-mode posture
    // note above.
    void stop();

    // Reads the trace bytes from the session for the given pid via
    // ReadTraceBlocking. Returns an empty vector when no session exists.
    // Caller is responsible for releasing the session via release_session
    // afterwards if desired.
    std::vector<char> read_trace(pid_t pid);

    // Drops the per-pid session slot. After release the slot is empty;
    // session_ref(pid) will return a fresh empty unique_ptr the next time
    // it is asked for that pid.
    void release_session(pid_t pid);

    // Whether a session is currently active.
    bool is_running() const noexcept;

    // Per-pid TracingSession bridge for legacy callers (notably
    // rocprofsys::get_perfetto_session(pid_t) used by fork_gotcha to release
    // the parent's session in the child after fork). Returns the same
    // unique_ptr slot the engine writes during start(); allocates an empty
    // slot on first access for an unknown pid.
    std::unique_ptr<::perfetto::TracingSession>& session_ref(pid_t pid);

    // Thread-local pid tag used by future cached-mode interceptor TLS (D4).
    // Exposed in slice B so tests can exercise the round-trip; live mode
    // does not consume it.
    static void set_emitting_pid(int pid) noexcept;
    static int  get_emitting_pid() noexcept;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};
}  // namespace core
}  // namespace rocprofsys
