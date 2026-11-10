// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// sampler.cpp
//
// Out-of-line implementations of cpu_snap::sampling::sampler. The
// header keeps the class declaration + trivial accessors; this TU owns
// the heavy bodies and selects the capture backend at construction.

#include "cpu_snap/sampling/sampler.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)
#include <unistd.h>

#include <csignal>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/backend_selection.hpp"
#include "cpu_snap/sampling/fork_registry.hpp"
#include "cpu_snap/sampling/logger.hpp"
#include "cpu_snap/sampling/sampler_capabilities.hpp"
#include "cpu_snap/sampling/sampler_plan.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"
#include "cpu_snap/sampling/thread_tracker.hpp"
#include "cpu_snap/sampling/time.hpp"
#include "cpu_snap/sampling/triggers/perf/backend.hpp"
#include "cpu_snap/sampling/triggers/perf/kernel_drain.hpp"
#include "cpu_snap/sampling/triggers/perf/ring_sizing.hpp"
#include "cpu_snap/sampling/triggers/signal/backend.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace cpu_snap::sampling
{

namespace
{

unwind::capture_kind
required_capture_of(const unwind::unwinder& unw) noexcept
{
    return std::visit(
        [](const auto& u) { return std::decay_t<decltype(u)>::required_capture(); }, unw);
}

// dwarf_unwinder is neither copyable nor movable (owns a non-relocatable
// libdwfl session), so the variant itself has no move/copy assignment;
// emplace() is the only way to (re)select an alternative in place.
void
select_unwinder(unwind_mode resolved, backend_kind bk, unwind::unwinder& unw)
{
    if(resolved == unwind_mode::dwarf)
    {
        unw.emplace<unwind::dwarf_unwinder>();
    }
    else if(bk == backend_kind::signal)
    {
        unw.emplace<unwind::fp_copy_unwinder>();
    }
    else
    {
        unw.emplace<unwind::kernel_callchain_unwinder>();
    }
}

[[nodiscard]] std::string_view
backend_name(backend_kind kind) noexcept
{
    return kind == backend_kind::perf ? "perf" : "signal";
}

[[nodiscard]] std::string_view
unwind_name(unwind_mode mode) noexcept
{
    return mode == unwind_mode::dwarf ? "dwarf" : "frame_pointer";
}

constexpr std::uint32_t k_frame_pointer_hint_freq_hz = 10000;

[[nodiscard]] std::string_view
clock_name(clock_kind clock) noexcept
{
    switch(clock)
    {
        case clock_kind::boottime: return "boottime";
        case clock_kind::wallclock: return "wallclock";
        case clock_kind::monotonic: break;
    }
    return "monotonic";
}

[[nodiscard]] std::string
describe_exception(std::exception_ptr error)
{
    try
    {
        std::rethrow_exception(std::move(error));
    } catch(const std::exception& e)
    {
        return e.what();
    } catch(...)
    {
        return "unknown error";
    }
}

}  // namespace

sampler::sampler(config cfg, std::unique_ptr<triggers::i_trigger> backend)
: m_cfg{}
, m_caps{ probe_sampler_capabilities() }
, m_backend{ std::move(backend) }
{
    auto plan =
        internal::resolve_plan(std::move(cfg), &probe_perf_open, m_backend != nullptr);
    m_cfg                      = std::move(plan.cfg);
    m_choice                   = std::move(plan.choice);
    const bool unwind_was_auto = plan.unwind_was_auto;
    select_unwinder(m_cfg.perf.unwind, m_cfg.backend, m_unwinder);
    m_threads.emplace(internal::proc_thread_source(), m_cfg.logger, m_cfg.clock,
                      m_cfg.threads.include_main_thread, m_cfg.threads.is_excluded);

    if(!m_backend)
    {
        switch(m_cfg.backend)
        {
            case backend_kind::perf:
                m_backend = std::make_unique<triggers::perf::backend>(
                    internal::make_drain_config(m_cfg));
                break;
            case backend_kind::signal:
                m_backend = std::make_unique<triggers::signal::backend>(
                    internal::make_signal_config(m_cfg));
                break;
            case backend_kind::automatic: break;
        }
    }

    // capture_kind handshake, checked once here against the real constructed
    // trigger (resolve_plan already validated this via the
    // context-free capability query, ahead of construction).
    if(!m_backend->can_provide(required_capture_of(m_unwinder)))
    {
        throw config_error(
            "sampler: selected trigger cannot provide the capture kind the "
            "chosen unwinder requires");
    }

    internal::emit(m_cfg.logger, m_cfg.clock,
                   m_choice.reason.empty() ? log_level::info : log_level::warn,
                   describe_backend_choice(m_choice));
    const auto freq = m_cfg.backend == backend_kind::perf ? m_cfg.perf.sample_freq_hz
                                                          : m_cfg.signal.sample_freq_hz;
    internal::emit(m_cfg.logger, m_cfg.clock, log_level::info, [&] {
        return "configuration: backend=" + std::string(backend_name(m_cfg.backend)) +
               " clock=" + std::string(clock_name(m_cfg.clock)) +
               " unwind=" + std::string(unwind_name(m_cfg.perf.unwind)) +
               (unwind_was_auto ? " (auto_best)" : "") + " freq=" + std::to_string(freq) +
               "Hz";
    });
    if(m_cfg.perf.unwind == unwind_mode::dwarf && freq >= k_frame_pointer_hint_freq_hz)
    {
        internal::emit(m_cfg.logger, m_cfg.clock, log_level::info,
                       "hint: at >= 10 kHz unwind=frame_pointer costs roughly 20x less "
                       "sampler CPU if the target binaries keep frame pointers");
    }
}

sampler::~sampler()
{
    // Dtor must not throw; stop() may throw on worker.join() if the
    // worker was already joined out-of-band, or from a host-supplied
    // streaming sink that escapes the framer.
    try
    {
        stop();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch): dtor must not throw
    catch(...)
    {
        // sampler destructor: stop() failure must not propagate
        // (E.16); stack unwinding is already best-effort here.
    }
    internal::fork_registry::remove(this);
}

void
sampler::on_child_fork() noexcept
{
    // Worker thread does not exist in the child (fork copies only
    // the calling thread). Detach to avoid std::terminate at dtor.
    if(m_worker.joinable()) m_worker.detach();
    m_backend->close_all();
    if(m_streaming) m_streaming->abandon();
    m_streaming.reset();
    // The worker may have held the mutex at fork time; it can never unlock it here.
    std::construct_at(&m_state_mutex);
    m_running.store(false, std::memory_order_release);
    m_paused.store(false, std::memory_order_release);
}

void
sampler::start()
{
    if(m_running.load(std::memory_order_acquire)) return;

    try
    {
        m_paused.store(false, std::memory_order_release);
        {
            const auto                        guard = hold_rescan_guard();
            const std::lock_guard<std::mutex> lk(m_state_mutex);
            m_streaming.emplace(build_streaming_config());
            m_final_stream_stats = {};

            m_threads->attach_initial(*m_backend, excluded_tid());
        }

        m_shutdown.store(false, std::memory_order_relaxed);
        m_worker_failed.store(false, std::memory_order_relaxed);
        m_worker = std::thread([this] { worker_loop(); });
        internal::fork_registry::add(this, [](void* self) noexcept {
            static_cast<sampler*>(self)->on_child_fork();
        });
        m_running.store(true, std::memory_order_release);
    } catch(...)
    {
        rollback_start();
        throw;
    }

    internal::emit(m_cfg.logger, m_cfg.clock, log_level::info, [this] {
        const auto freq = m_cfg.backend == backend_kind::perf
                              ? m_cfg.perf.sample_freq_hz
                              : m_cfg.signal.sample_freq_hz;
        return "sampler started: backend=" + std::string(backend_name(m_cfg.backend)) +
               " freq=" + std::to_string(freq) +
               " unwind=" + std::string(unwind_name(m_cfg.perf.unwind));
    });
}

void
sampler::rollback_start() noexcept
{
    m_shutdown.store(true, std::memory_order_release);
    if(m_worker.joinable()) m_worker.join();
    internal::fork_registry::remove(this);

    const std::lock_guard<std::mutex> lk(m_state_mutex);
    m_backend->close_all();
    m_threads->reset();
    if(m_streaming)
    {
        m_streaming->abandon();
        m_streaming.reset();
    }
}

void
sampler::pause() noexcept
{
    if(!m_running.load(std::memory_order_acquire)) return;
    if(m_paused.exchange(true, std::memory_order_acq_rel)) return;
    const std::lock_guard<std::mutex> lk(m_state_mutex);
    m_backend->disable_all();
}

void
sampler::resume() noexcept
{
    if(!m_running.load(std::memory_order_acquire)) return;
    if(!m_paused.exchange(false, std::memory_order_acq_rel)) return;
    const std::lock_guard<std::mutex> lk(m_state_mutex);
    m_backend->enable_all();
}

void
sampler::stop()
{
    if(!m_running.exchange(false, std::memory_order_acq_rel)) return;
    m_shutdown.store(true, std::memory_order_release);
    if(m_worker.joinable()) m_worker.join();

    {
        const std::lock_guard<std::mutex> lk(m_state_mutex);
        m_backend->disable_all();
        drain_once();
        m_peak_tracked.store(m_backend->tracked_tids(), std::memory_order_relaxed);
        m_backend->close_all();
        m_threads->reset();
        if(m_streaming)
        {
            m_streaming->finalize();
            m_final_stream_stats = m_streaming->snapshot();
            m_streaming.reset();
        }
    }

    internal::emit(m_cfg.logger, m_cfg.clock, log_level::info, [this] {
        return "sampler stopped: samples=" + std::to_string(total_samples()) +
               " dropped=" + std::to_string(streaming_dropped_samples()) +
               " internal_errors=" + std::to_string(streaming_internal_errors()) +
               " live_stack_samples=" + std::to_string(unwinder_live_stack_samples());
    });
}

void
sampler::drain_once() noexcept
{
    m_backend->drain_into(*m_streaming, m_unwinder);
}

internal::streaming_pipeline::config
sampler::build_streaming_config()
{
    return internal::streaming_pipeline::config{
        .sink          = m_cfg.streaming.sink,
        .chunk_bytes   = m_cfg.streaming.chunk_bytes,
        .flags         = internal::streaming_flags_for(m_cfg.perf.unwind, m_cfg.clock),
        .pid           = static_cast<std::uint64_t>(::getpid()),
        .start_unix_ns = utils::wall_unix_ns_now(),
        .process_name  = m_cfg.streaming.process_name.empty()
                             ? utils::read_proc_self_comm()
                             : m_cfg.streaming.process_name,
        .maps_poll_interval  = m_cfg.streaming.maps_poll_interval,
        .checkpoint_interval = m_cfg.streaming.checkpoint_interval,
        .trie_max_nodes      = m_cfg.streaming.trie_max_nodes,
        .total_counter       = &m_total,
        .lost_counter        = &m_lost,
        .clock               = m_cfg.clock,
    };
}

pid_t
sampler::excluded_tid() const noexcept
{
    return m_cfg.threads.exclude_sampler_worker
               ? m_worker_tid.load(std::memory_order_acquire)
               : pid_t{ -1 };
}

std::shared_ptr<void>
sampler::hold_rescan_guard() const
{
    return m_cfg.threads.rescan_guard ? m_cfg.threads.rescan_guard() : nullptr;
}

void
sampler::rescan_threads()
{
    const auto guard   = hold_rescan_guard();
    const auto current = m_threads->poll();
    if(!current)
    {
        m_rescan_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const std::lock_guard<std::mutex> lk(m_state_mutex);
    const auto                        result = m_threads->rescan(
        *m_backend, m_streaming ? &*m_streaming : nullptr, excluded_tid(), *current);
    m_rescan_failures.fetch_add(result.failed_attaches, std::memory_order_relaxed);
}

void
sampler::worker_loop() noexcept
{
    m_worker_tid.store(static_cast<pid_t>(::syscall(SYS_gettid)),
                       std::memory_order_release);
    try
    {
        // Level-triggered wait is owned by the backend. Timeout bounds
        // tick latency so shutdown + rescan still fire when no ring has
        // data.
        constexpr int k_wait_timeout_ms = 10;
        auto next_rescan = std::chrono::steady_clock::now() + m_cfg.threads.rescan_period;

        while(!m_shutdown.load(std::memory_order_acquire))
        {
            (void) m_backend->wait(k_wait_timeout_ms);
            drain_once();

            if(m_cfg.threads.watch_new_threads &&
               std::chrono::steady_clock::now() >= next_rescan)
            {
                rescan_threads();
                next_rescan =
                    std::chrono::steady_clock::now() + m_cfg.threads.rescan_period;
            }
        }
    }
    // Worker must not propagate to std::terminate.
    catch(...)
    {
        m_worker_failed.store(true, std::memory_order_release);
        internal::emit(m_cfg.logger, m_cfg.clock, log_level::error, [] {
            return "worker thread terminated by an exception: " +
                   describe_exception(std::current_exception());
        });
    }
}

}  // namespace cpu_snap::sampling
