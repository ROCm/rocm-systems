// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/process_sampler.hpp"

#include "common/clocks.hpp"
#include "core/collection/detail/source_runner.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sample_collector.hpp"
#include "core/collection/source_interface.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/collection_sink.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/sources.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>
#include <timemory/process/threading.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::process_sampler
{
namespace
{
using sink_t = trace_cache::collection_sink<trace_cache::buffer_storage_t,
                                            trace_cache::metadata_registry>;

/** Set by init_poll_thread(): shutdown() and post_process() must not stop or destroy the
 * collector from one of its own poll threads (a source library that calls exit()). */
thread_local bool g_on_poll_thread = false;

void
init_poll_thread(std::string_view source)
{
    g_on_poll_thread = true;
    // thread_info reads timemory's id offset; no project-owned naming facility exists
    threading::offset_this_id(true);
    threading::set_thread_name(fmt::format("omni.{}", source).c_str());
    state::thread::push(state::thread::Internal);
}

/** The one place that picks a source's polling interval. Today: the global one. */
[[nodiscard]] std::chrono::nanoseconds
interval_for(std::string_view /*source*/, std::chrono::nanoseconds global)
{
    return global;
}

/** Members are destroyed in reverse: the collector (joins its threads, releases the
 * providers) before the sink. */
class sampler
{
public:
    sampler(std::chrono::nanoseconds interval, std::uint64_t deadline_ns,
            collection::fork_gate& gate, std::function<bool()> allowed)
    : m_interval(interval)
    , m_sink(trace_cache::get_buffer_storage(), trace_cache::get_metadata_registry())
    , m_collector(m_sink, gate,
                  collection::detail::run_options{ .allowed     = std::move(allowed),
                                                   .thread_init = &init_poll_thread,
                                                   .deadline_ns = deadline_ns })
    {}

    void add(std::unique_ptr<collection::source_interface> source)
    {
        const auto interval = interval_for(source->name(), m_interval);
        m_collector.add_source(std::move(source), interval);
    }

    [[nodiscard]] collection::sample_collector& collector() noexcept
    {
        return m_collector;
    }

private:
    std::chrono::nanoseconds     m_interval;
    sink_t                       m_sink;
    collection::sample_collector m_collector;
};

/** The process-wide sampler, its lifecycle and the sources queued before setup(). */
class process_sampler_state
{
public:
    void add_source(std::unique_ptr<collection::source_interface> source)
    {
        const std::lock_guard lock(m_mutex);
        if(!m_sampler)
        {
            m_pending.push_back(std::move(source));
            return;
        }
        if(m_state.load() == state::process::Finalized)
        {
            LOG_DEBUG("Source not added: the background sampler is shut down.");
            return;
        }
        m_sampler->add(std::move(source));
        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        m_sampler->collector().start();
    }

    void setup(std::chrono::nanoseconds interval, std::uint64_t deadline_ns)
    {
        const std::lock_guard lock(m_mutex);
        take_sampler().reset();
        m_state.store(state::process::PreInit);

        auto next = std::make_unique<sampler>(interval, deadline_ns, m_gate,
                                              [this] { return is_sampling_allowed(); });
        for(auto& source : pmc::make_default_sources())
        {
            next->add(std::move(source));
        }
        for(auto& source : std::exchange(m_pending, {}))
        {
            next->add(std::move(source));
        }
        install(std::move(next));

        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        m_sampler->collector().start();
        m_state.store(state::process::Active);
    }

    void shutdown()
    {
        if(g_on_poll_thread)
        {
            // No thread can be joined from here. The other sources' threads keep running
            // but skip their ticks (allowed() is false); their polls in flight are
            // drained so finalize does not run under them. Our own pass is discounted.
            m_state.store(state::process::Finalized);
            m_gate.block();
            m_gate.unblock();
            return;
        }
        const std::lock_guard lock(m_mutex);
        m_state.store(state::process::Finalized);
        if(m_sampler)
        {
            m_sampler->collector().stop();
        }
    }

    void post_process()
    {
        if(g_on_poll_thread)
        {
            return;
        }
        const std::lock_guard lock(m_mutex);
        take_sampler().reset();
    }

    void pause()
    {
        const std::lock_guard lock(m_swap_mutex);
        m_paused = true;
        if(m_sampler)
        {
            m_sampler->collector().pause();
        }
    }

    void resume()
    {
        const std::lock_guard lock(m_swap_mutex);
        m_paused = false;
        if(m_sampler)
        {
            m_sampler->collector().resume();
        }
    }

    [[nodiscard]] collection::fork_gate& gate() noexcept { return m_gate; }

private:
    [[nodiscard]] bool is_sampling_allowed() const
    {
        return m_state.load() == state::process::Active &&
               state::process::get() == state::process::Active;
    }

    /** Caller holds m_mutex. The returned sampler is stopped and destroyed by the caller
     * outside m_swap_mutex, so pause() and resume() never wait for it. */
    [[nodiscard]] std::unique_ptr<sampler> take_sampler()
    {
        const std::lock_guard lock(m_swap_mutex);
        return std::exchange(m_sampler, nullptr);
    }

    /** Caller holds m_mutex. A sampler installed while paused starts paused. */
    void install(std::unique_ptr<sampler> next)
    {
        const std::lock_guard lock(m_swap_mutex);
        m_sampler = std::move(next);
        if(m_paused)
        {
            m_sampler->collector().pause();
        }
    }

    /** Lifecycle: setup, shutdown, add_source, post_process. */
    std::mutex m_mutex;
    /** Short: m_sampler (written with m_mutex held too) and m_paused. */
    std::mutex                         m_swap_mutex;
    std::atomic<state::process::State> m_state{ state::process::PreInit };
    bool                               m_paused = false;
    std::vector<std::unique_ptr<collection::source_interface>> m_pending;
    /** Outlives every sampler. */
    collection::fork_gate    m_gate;
    std::unique_ptr<sampler> m_sampler;
};

/** Never destroyed: finalize can run from atexit() after the static destructors. */
process_sampler_state*&
state_slot()
{
    static process_sampler_state* s_state =
        std::make_unique<process_sampler_state>().release();
    return s_state;
}

process_sampler_state&
instance()
{
    return *state_slot();
}

/** Timeline ns at which sampling ends; max when no duration is configured. */
std::uint64_t
sampling_deadline_ns()
{
    auto duration = config::get_process_sampling_duration();
    if(duration < 0.0)
    {
        duration = config::get_sampling_duration();
    }
    if(duration <= 0.0)
    {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{ duration });
    return clocks::timeline::now_ns() + static_cast<std::uint64_t>(duration_ns.count());
}
}  // namespace

void
add_source(std::unique_ptr<collection::source_interface> source)
{
    instance().add_source(std::move(source));
}

void
setup()
{
    if(!get_use_process_sampling())
    {
        LOG_DEBUG("Background sampler is disabled...");
        return;
    }

    const auto interval = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{ 1.0 / get_process_sampling_freq() });
    LOG_DEBUG(
        "Background process sampling polling at an interval of {:.2f} seconds...",
        std::chrono::duration_cast<std::chrono::duration<double>>(interval).count());

    instance().setup(interval, sampling_deadline_ns());
}

void
shutdown()
{
    instance().shutdown();
}

void
post_process()
{
    instance().post_process();
}

void
pause()
{
    instance().pause();
}

void
resume()
{
    instance().resume();
}

void
block()
{
    instance().gate().block();
}

void
unblock()
{
    instance().gate().unblock();
}

void
postfork_child()
{
    // The old state is leaked on purpose: its threads, mutexes and providers belong to
    // threads that do not exist in the child.
    state_slot() = std::make_unique<process_sampler_state>().release();
}
}  // namespace rocprofsys::process_sampler
