// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/process_sampler.hpp"

#include "core/collection/sample_collector.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/source_interface.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/trace_cache/collection_sink.hpp"
#include "library/collection/sources/cpu_source.hpp"
#include "library/pmc/sources.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::process_sampler
{
namespace
{
void
init_sampler_thread()
{
    threading::offset_this_id(true);
    threading::set_thread_name("omni.sampler");
    state::thread::push(state::thread::Internal);
}

/** The schedulers of the running sources, so a fork can wait for their polls. It has its
 * own lock: the fork can come from inside `setup()`, which holds the sampler lock. */
class scheduler_registry
{
public:
    void add(collection::scheduler<>& scheduler)
    {
        const std::lock_guard lock(m_mutex);
        m_schedulers.push_back(&scheduler);
    }

    void clear()
    {
        const std::lock_guard lock(m_mutex);
        m_schedulers.clear();
    }

    void block()
    {
        for(auto* scheduler : snapshot())
        {
            scheduler->block();
        }
    }

    void unblock()
    {
        for(auto* scheduler : snapshot())
        {
            scheduler->unblock();
        }
    }

private:
    // Copied so that no lock is held while a scheduler waits for its callback.
    std::vector<collection::scheduler<>*> snapshot()
    {
        const std::lock_guard lock(m_mutex);
        return m_schedulers;
    }

    std::mutex                            m_mutex;
    std::vector<collection::scheduler<>*> m_schedulers;
};

/** What the owner gives the sampler: when sources may poll, and where their schedulers
 * are registered. */
struct sampler_context
{
    std::function<bool()> is_allowed;
    scheduler_registry&   schedulers;
};

/** Destroyed in the reverse order of the members: the collector first, then the
 * providers its sources use. */
class sampler
{
public:
    sampler(std::chrono::nanoseconds period, std::chrono::nanoseconds duration,
            const std::vector<source_factory>& added_sources, sampler_context context)
    : m_interval(period)
    , m_context(std::move(context))
    , m_end(duration > std::chrono::nanoseconds::zero()
                ? std::optional{ std::chrono::steady_clock::now() + duration }
                : std::nullopt)
    , m_collector(m_sink)
    {
        m_collector.add(std::make_unique<collection::sources::cpu_source<>>(
            period, make_scheduler()));
        if(config::get_use_amd_smi())
        {
            m_collector.add(
                pmc::make_gpu_smi_source(period, m_providers, make_scheduler()));
            add_if_built(pmc::make_nic_smi_source(period, m_providers, make_scheduler()));
        }
        add_if_built(pmc::make_hipfile_source(period, m_providers, make_scheduler()));

        for(const auto& factory : added_sources)
        {
            add_from_factory(factory);
        }
    }

    ~sampler() { m_context.schedulers.clear(); }

    sampler(const sampler&)            = delete;
    sampler& operator=(const sampler&) = delete;

    // The collector has to be started for the source to take effect.
    void add_from_factory(const source_factory& factory)
    {
        add_if_built(factory(m_interval, make_scheduler()));
    }

    collection::sample_collector& collector() { return m_collector; }

    // Stops the sources, then their providers.
    void stop()
    {
        m_collector.stop();
        m_providers.shutdown();
        m_providers.release();
    }

    void release_providers() { m_providers.release(); }

private:
    void add_if_built(std::unique_ptr<collection::source_interface> source)
    {
        if(source)
        {
            m_collector.add(std::move(source));
        }
    }

    // The schedulers of the sources: the thread setup, and a gate that skips the ticks
    // while the profiler is not active or the sampling duration is over.
    pmc::scheduler_factory_t make_scheduler()
    {
        return [this](std::string name) {
            auto scheduler = std::make_unique<collection::scheduler<>>(
                std::move(name), &init_sampler_thread, [this] {
                    return m_context.is_allowed() &&
                           (!m_end || std::chrono::steady_clock::now() < *m_end);
                });
            m_context.schedulers.add(*scheduler);
            return scheduler;
        };
    }

    std::chrono::nanoseconds                             m_interval;
    sampler_context                                      m_context;
    std::optional<std::chrono::steady_clock::time_point> m_end;  // none: no limit
    pmc::providers                                       m_providers;
    trace_cache::collection_sink                         m_sink;
    collection::sample_collector                         m_collector;
};

/** The process-wide sampler: the state, the lock and the sources added before it exists.
 */
class process_sampler_state
{
public:
    void add_source(source_factory factory)
    {
        const std::lock_guard lock(m_mutex);
        if(!m_sampler)
        {
            m_added_factories.push_back(std::move(factory));
            return;
        }
        if(m_state.load() == state::process::Finalized)
        {
            LOG_DEBUG("Source not added: the background sampler is shut down.");
            return;
        }
        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        m_sampler->add_from_factory(factory);
        m_sampler->collector().start();
    }

    void setup(std::chrono::nanoseconds interval, std::chrono::nanoseconds duration)
    {
        const std::lock_guard lock(m_mutex);
        stop_locked();  // restart if it is already running
        m_sampler.reset();
        m_state.store(state::process::PreInit);

        m_sampler = std::make_unique<sampler>(
            interval, duration, std::exchange(m_added_factories, {}),
            sampler_context{ .is_allowed = [this] { return is_sampling_allowed(); },
                             .schedulers = m_schedulers });

        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        m_sampler->collector().start();
        m_state.store(state::process::Active);
    }

    void shutdown()
    {
        const std::lock_guard lock(m_mutex);
        m_state.store(state::process::Finalized);
        stop_locked();
    }

    void pause()
    {
        const std::lock_guard lock(m_mutex);
        if(m_sampler)
        {
            m_sampler->collector().pause();
        }
    }

    void resume()
    {
        const std::lock_guard lock(m_mutex);
        if(m_sampler)
        {
            m_sampler->collector().resume();
        }
    }

    void post_process()
    {
        const std::lock_guard lock(m_mutex);
        if(m_sampler)
        {
            m_sampler->release_providers();
            m_sampler.reset();
        }
    }

    // No lock: the fork can come from a poll that runs inside `setup()`.
    void block() { m_schedulers.block(); }

    void unblock() { m_schedulers.unblock(); }

    void postfork_child()
    {
        // The sampler thread was not copied into the child: destroying the collector
        // would join a thread that does not exist, so it is leaked on purpose.
        static_cast<void>(m_sampler.release());
        m_schedulers.clear();
    }

private:
    [[nodiscard]] bool is_sampling_allowed() const
    {
        return m_state.load() == state::process::Active &&
               state::process::get() == state::process::Active;
    }

    void stop_locked()
    {
        if(m_sampler)
        {
            m_sampler->stop();
        }
    }

    std::mutex                         m_mutex;  // the sampler, the state transitions
    std::atomic<state::process::State> m_state{ state::process::PreInit };
    std::vector<source_factory>        m_added_factories;
    scheduler_registry                 m_schedulers;
    std::unique_ptr<sampler>           m_sampler;
};

process_sampler_state&
instance()
{
    static process_sampler_state s_state;
    return s_state;
}

std::chrono::nanoseconds
get_sampling_duration()
{
    auto duration = config::get_process_sampling_duration();
    if(duration < 0.0)
    {
        duration = config::get_sampling_duration();
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{ duration > 0.0 ? duration : 0.0 });
}
}  // namespace

void
add_source(source_factory factory)
{
    instance().add_source(std::move(factory));
}

void
setup()
{
    if(!get_use_process_sampling())
    {
        LOG_DEBUG("Background sampler is disabled...");
        return;
    }

    LOG_DEBUG("Setting up background sampler...");

    const auto interval = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{ 1.0 / get_process_sampling_freq() });
    LOG_DEBUG(
        "Background process sampling polling at an interval of {:.2f} seconds...",
        std::chrono::duration_cast<std::chrono::duration<double>>(interval).count());

    instance().setup(interval, get_sampling_duration());
}

void
shutdown()
{
    instance().shutdown();
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
post_process()
{
    instance().post_process();
}

void
block()
{
    instance().block();
}

void
unblock()
{
    instance().unblock();
}

void
postfork_child()
{
    instance().postfork_child();
}
}  // namespace rocprofsys::process_sampler
