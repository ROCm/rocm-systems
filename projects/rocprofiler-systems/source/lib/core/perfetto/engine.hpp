// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "fwd.hpp"
#include "sinks.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

namespace rocprofsys::core
{
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

enum class perfetto_engine_mode
{
    live_fd,
    cached_interceptor,
};

// Reads the live config::get_perfetto_* getters once and returns a snapshot.
// Called once per process by whoever constructs the perfetto_engine.
engine_config
build_engine_config_from_settings();

using cached_engine_collect_fn = void (*)(void*, int, const void*, std::size_t) noexcept;

void*
activate_cached_engine(void* engine, cached_engine_collect_fn collect) noexcept;

bool
clear_active_cached_engine(void* expected, void** observed) noexcept;

// Owns Perfetto SDK init + the per-pid tracing session map.
//
// The Backend policy is the compile-time seam around the concrete Perfetto SDK.
// Production uses perfetto_sdk_backend; tests can instantiate
// basic_perfetto_engine<mock_backend> and verify engine orchestration without a
// real Perfetto session or SDK global state.
template <perfetto_backend Backend>
class basic_perfetto_engine
{
public:
    using mode        = perfetto_engine_mode;
    using backend_t   = Backend;
    using session_ptr = typename Backend::session_ptr;

    explicit basic_perfetto_engine(engine_config cfg);
    ~basic_perfetto_engine() noexcept;

    basic_perfetto_engine(const basic_perfetto_engine&)            = delete;
    basic_perfetto_engine& operator=(const basic_perfetto_engine&) = delete;
    basic_perfetto_engine(basic_perfetto_engine&&)                 = delete;
    basic_perfetto_engine& operator=(basic_perfetto_engine&&)      = delete;

    void init_sdk();
    void start(mode m, int fd);
    void start(mode m, trace_sink& sink);
    void stop();

    [[nodiscard]] std::vector<char> read_trace(pid_t pid);
    void                            destroy_session(pid_t pid);
    void                            forget_session(pid_t pid);
    [[nodiscard]] bool              is_running() const noexcept;

    void preregister_pids(const std::vector<int>& source_pids);
    void collect_packet_bytes(int pid, const void* data, std::size_t size) noexcept;

private:
    static constexpr std::size_t COLLECTED_BYTES_SLAB_SIZE =
        std::size_t{ 8 } * 1024 * 1024;

    [[nodiscard]] bool is_system_backend() const noexcept;
    void               start_session(pid_t pid, int fd, mode m);
    static void        collect_thunk(void* engine, int pid, const void* data,
                                     std::size_t size) noexcept;

    engine_config m_cfg{};
    Backend       m_backend{};
    pid_t         m_active_pid{ 0 };
    bool          m_running{ false };
    mode          m_current_mode{ mode::live_fd };
    trace_sink*   m_active_sink{ nullptr };

    std::mutex m_sessions_mutex{};

    std::mutex                                 m_collector_mutex{};
    std::unordered_map<int, std::vector<char>> m_collected_bytes{};
    std::atomic<bool>                          m_collected_bytes_frozen{ false };
    std::atomic<std::size_t>                   m_dropped_packet_count{ 0 };

    std::unordered_map<pid_t, session_ptr> m_sessions{};
};

extern template class basic_perfetto_engine<perfetto_sdk_backend>;

// Thread-local pid tag consumed by the cached-mode interceptor TLS to key each
// thread's emissions to a pid. Tagging threads is the emitter's responsibility —
// call set_emitting_pid(pid) before the first TRACE_EVENT_* on the thread.
void
set_emitting_pid(int pid) noexcept;

int
get_emitting_pid() noexcept;
}  // namespace rocprofsys::core
