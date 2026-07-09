// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "fwd.hpp"
#include "sinks/trace_sink.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::core
{
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

engine_config
build_engine_config_from_settings();

using cached_engine_collect_fn = void (*)(void*, int, const void*, std::size_t) noexcept;

void*
activate_cached_engine(void* engine, cached_engine_collect_fn collect) noexcept;

bool
clear_active_cached_engine(void* expected, void** observed) noexcept;

template <perfetto_backend Backend>
class basic_cached_perfetto_engine
{
public:
    using backend_t   = Backend;
    using session_ptr = typename Backend::session_ptr;

    explicit basic_cached_perfetto_engine(engine_config cfg);
    ~basic_cached_perfetto_engine() noexcept;

    basic_cached_perfetto_engine(const basic_cached_perfetto_engine&)            = delete;
    basic_cached_perfetto_engine& operator=(const basic_cached_perfetto_engine&) = delete;
    basic_cached_perfetto_engine(basic_cached_perfetto_engine&&)                 = delete;
    basic_cached_perfetto_engine& operator=(basic_cached_perfetto_engine&&)      = delete;

    void init_sdk();
    void start(trace_sink& sink);
    void stop();

    [[nodiscard]] bool is_running() const noexcept;

    void preregister_pids(const std::vector<int>& source_pids);
    void collect_packet_bytes(int pid, const void* data, std::size_t size) noexcept;

private:
    // Initial per-pid capacity hint, not a cap -- collect_packet_bytes grows the
    // vector past this via normal amortized reallocation. Kept small because
    // preregister_pids reserves this for every cached pid up front; a large value
    // multiplies with pid count on high-rank-count MPI workloads.
    static constexpr std::size_t COLLECTED_BYTES_INITIAL_CAPACITY =
        std::size_t{ 256 } * 1024;

    [[nodiscard]] bool is_system_backend() const noexcept;
    static void        collect_thunk(void* engine, int pid, const void* data,
                                     std::size_t size) noexcept;

    engine_config                                     m_cfg{};
    Backend                                           m_backend{};
    bool                                              m_running{ false };
    std::optional<std::reference_wrapper<trace_sink>> m_active_sink{};
    session_ptr                                       m_session{};

    std::mutex                                 m_collector_mutex{};
    std::unordered_map<int, std::vector<char>> m_collected_bytes{};
    std::atomic<bool>                          m_collected_bytes_frozen{ false };
    std::atomic<std::size_t>                   m_dropped_packet_count{ 0 };
};

extern template class basic_cached_perfetto_engine<perfetto_sdk_backend>;

void
set_emitting_pid(int pid) noexcept;

int
get_emitting_pid() noexcept;
}  // namespace rocprofsys::core
