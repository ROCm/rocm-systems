// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/detail/source_runner.hpp"
#include "common/clocks.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::collection::detail
{
namespace
{
/** Calls func(); a throw is rethrown as std::runtime_error prefixed with `step`, so the
 * scheduler's "stopped polling" line names the step that failed. */
template <typename Fn>
auto
rethrow_named(std::string_view step, Fn&& func)
{
    try
    {
        return std::forward<Fn>(func)();
    } catch(const std::exception& error)
    {
        throw std::runtime_error(fmt::format("{} failed: {}", step, error.what()));
    } catch(...)
    {
        throw std::runtime_error(fmt::format("{} failed: unknown exception", step));
    }
}
}  // namespace

source_runner::source_runner(std::unique_ptr<source_interface> source,
                             std::chrono::nanoseconds          interval,
                             const run_context& context, sink_interface& out)
: m_source(std::move(source))
, m_interval(interval)
, m_context(context)
, m_out(out)
, m_scheduler(std::string{ m_source->name() })
{}

source_interface&
source_runner::source() noexcept
{
    return *m_source;
}

void
source_runner::start()
{
    m_scheduler.start(
        m_interval, [this](std::uint64_t timeline_ns) { return tick(timeline_ns); },
        [this] { init_thread(); });
}

void
source_runner::init_thread()
{
    m_poll_thread.store(std::this_thread::get_id());
    const auto& thread_init = m_context.options.thread_init;
    if(!thread_init)
    {
        return;
    }
    const fork_gate::pass pass{ m_context.gate, fork_gate::pass::k_wait };
    rethrow_named("thread_init()", [&] { thread_init(m_source->name()); });
}

bool
source_runner::is_allowed() const
{
    const auto& allowed = m_context.options.allowed;
    return !allowed || rethrow_named("allowed()", [&] { return allowed(); });
}

void
source_runner::pause()
{
    m_scheduler.pause();
}

void
source_runner::resume()
{
    m_scheduler.resume();
}

void
source_runner::stop()
{
    m_scheduler.stop();
    if(m_poll_thread.load() == std::this_thread::get_id())
    {
        LOG_WARNING("collection: source '{}' stopped by its own poll; release skipped",
                    m_source->name());
        return;
    }
    m_source->release();
}

bool
source_runner::tick(std::uint64_t timeline_ns)
{
    if(timeline_ns >= m_context.options.deadline_ns)
    {
        return false;
    }
    // Checked inside the pass: a fork-gate drain then also waits for a tick that saw
    // allowed() before it changed
    const fork_gate::pass pass{ m_context.gate };
    if(!pass || m_context.paused.load(std::memory_order_acquire) || !is_allowed())
    {
        return true;
    }
    m_source->poll(clocks::timeline::now_ns(), m_out);
    return true;
}
}  // namespace rocprofsys::collection::detail
