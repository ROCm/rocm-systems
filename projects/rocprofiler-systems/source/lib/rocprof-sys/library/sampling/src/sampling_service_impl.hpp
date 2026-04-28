// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Template method bodies for sampling_service<Policies>.
// Included from sampling_service.hpp after the class definition.
// May use platform types (sigset_t, etc.) via the signal_dispatcher policy.

#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sample_parser.hpp"
#include "sampling/src/sampling_config_fwd.hpp"
#include "sampling/src/signal_set.hpp"
#include "sampling/src/symbol_resolver.hpp"

#include "logger/debug.hpp"

// POSIX defines sigmask(sig) as a 1-arg macro; undefine it so our 3-arg
// policy method named 'sigmask' is not subject to macro expansion.
#ifdef sigmask
#    undef sigmask
#endif

#include <algorithm>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <time.h>
#include <unistd.h>

namespace rocprofsys::sampling
{

// ── Construction / Destruction ─────────────────────────────────────────────

template <class Policies>
sampling_service<Policies>::sampling_service()
: pause_registry_(clock_)
, duration_controller_([this]() {
    duration_disabled_ = true;
    block_samples();
})
{}

template <class Policies>
sampling_service<Policies>::~sampling_service() = default;

// ── is_paused / is_blocked ─────────────────────────────────────────────────

template <class Policies>
bool
sampling_service<Policies>::is_paused() const noexcept
{
    return pause_registry_.is_paused();
}

template <class Policies>
bool
sampling_service<Policies>::is_blocked() const noexcept
{
    return blocked_.load(std::memory_order_relaxed);
}

template <class Policies>
size_t
sampling_service<Policies>::dropped_samples() const noexcept
{
    size_t total = 0;
    // Sum dropped counts across all active per-thread states (DEC-15).
    registry_.each(
        [&total](int64_t /*tid*/, thread_sampler_state<Policies> const& state) noexcept {
            total += state.dropped_count();
        });
    return total;
}

// ── block_samples / unblock_samples ───────────────────────────────────────

template <class Policies>
void
sampling_service<Policies>::block_samples()
{
    LOG_DEBUG("Blocking sampling...");
    blocked_.store(true, std::memory_order_release);
}

template <class Policies>
void
sampling_service<Policies>::unblock_samples()
{
    LOG_DEBUG("Unblocking sampling...");
    blocked_.store(false, std::memory_order_release);
}

// ── block_signals / unblock_signals ───────────────────────────────────────

template <class Policies>
void
sampling_service<Policies>::block_signals(std::set<int> sigs)
{
    auto calling_tid = static_cast<int64_t>(::gettid());
    if(sigs.empty())
    {
        sigs = get_signal_types(calling_tid);
    }
    if(sigs.empty())
    {
        LOG_DEBUG("No signals to block...");
        return;
    }

    std::ostringstream sig_buf;
    for(auto s : sigs)
    {
        if(sig_buf.tellp() > 0) sig_buf << ", ";
        sig_buf << s;
    }
    LOG_DEBUG("Blocking signals [{}] on thread #{}...", sig_buf.str(), calling_tid);

    signal_set ss(sigs);
    int        err = signal_dispatcher_.sigmask(SIG_BLOCK, ss.get(), nullptr);
    if(err != 0)
    {
        fatal_.fatal(__FILE__, __LINE__, "pthread_sigmask failed: errno={}", err);
    }
}

template <class Policies>
void
sampling_service<Policies>::unblock_signals(std::set<int> sigs)
{
    auto calling_tid = static_cast<int64_t>(::gettid());
    if(sigs.empty())
    {
        sigs = get_signal_types(calling_tid);
    }
    if(sigs.empty())
    {
        LOG_DEBUG("No signals to unblock...");
        return;
    }

    std::ostringstream sig_buf;
    for(auto s : sigs)
    {
        if(sig_buf.tellp() > 0) sig_buf << ", ";
        sig_buf << s;
    }
    LOG_DEBUG("Unblocking signals [{}] on thread #{}...", sig_buf.str(), calling_tid);

    signal_set ss(sigs);
    int        err = signal_dispatcher_.sigmask(SIG_UNBLOCK, ss.get(), nullptr);
    if(err != 0)
    {
        fatal_.fatal(__FILE__, __LINE__, "pthread_sigmask failed: errno={}", err);
    }
}

// ── pause / resume ────────────────────────────────────────────────────────

template <class Policies>
void
sampling_service<Policies>::pause()
{
    if(pause_registry_.pause())
    {
        block_samples();
    }
}

template <class Policies>
void
sampling_service<Policies>::resume()
{
    if(pause_registry_.resume())
    {
        unblock_samples();
    }
}

// ── get_signal_types ──────────────────────────────────────────────────────

template <class Policies>
std::set<int>
sampling_service<Policies>::get_signal_types(int64_t tid)
{
    std::lock_guard<std::mutex> lk(signal_types_mutex_);
    auto                        it = signal_types_.find(tid);
    if(it != signal_types_.end()) return it->second;

    // Lazy-initialize from config (matches today's signal_type_instances behavior).
    auto sigs          = rocprofsys::get_sampling_signals(tid);
    signal_types_[tid] = sigs;
    return sigs;
}

// ── setup ─────────────────────────────────────────────────────────────────

template <class Policies>
std::set<int>
sampling_service<Policies>::setup(int64_t tid)
{
    // AC-19: causal profiling guard.
    if(causal_mode_test_ || rocprofsys::get_use_causal())
    {
        throw std::runtime_error("Internal error! configuring sampling not permitted "
                                 "when causal profiling is enabled");
    }

    // AC-5: duration already fired.
    if(duration_disabled_) return {};

    // I-12: production thread-state guards (ThreadState::Disabled, offset thread).
    // No-op in the generic template; explicit specialization for
    // default_sampling_policies checks get_thread_state() and thread_info::get()
    // (requires main-lib headers).
    if(!setup_check_thread_guards(tid)) return {};

    // Compute signal set for this thread (one lock acquisition — C-6 fix).
    std::set<int> sigs;
    {
        std::lock_guard<std::mutex> lk(signal_types_mutex_);
        auto                        it = signal_types_.find(tid);
        if(it == signal_types_.end())
        {
            sigs               = rocprofsys::get_sampling_signals(tid);
            signal_types_[tid] = sigs;
        }
        else
        {
            sigs = it->second;
        }
    }
    if(sigs.empty()) return {};

    LOG_DEBUG("Configuring sampler for thread {}", tid);

    // Log the "will be triggered" message for each signal (SI-7).
    for(auto sig : sigs)
    {
        LOG_INFO("[SIG{}] Sampler for thread {} will be triggered by signal {}", sig, tid,
                 sig);
    }

    // Create per-thread state, record signal set, mark running (C-2).
    registry_.emplace(tid);
    auto* state = registry_.at(tid);
    if(state)
    {
        state->set_signal_types(sigs);
        state->start();
    }

    // Production wiring: TLS pointer setup, timer arming, overflow trigger config.
    // No-op in the generic template; explicit specialization for
    // default_sampling_policies does the real work (requires main-lib headers; defined in
    // library/sampling_production_policies/sampling_service_production_hooks.hpp).
    setup_production_wiring(tid, state, sigs);

    // Block signals on the calling thread before the trigger is armed.
    block_signals(sigs);

    return sigs;
}

// ── shutdown ──────────────────────────────────────────────────────────────

template <class Policies>
std::set<int>
sampling_service<Policies>::shutdown(int64_t tid)
{
    // AC-20: child process — release state without post-process.
    if(child_process_test_)
    {
        // In child mode we do NOT call post_process; just return.
        return {};
    }

    LOG_DEBUG("Stopping sampler for thread {}...", tid);

    // Retrieve the signal set before destroying state.
    auto sigs = get_signal_types(tid);

    // Block signals on the calling thread so no new samples arrive.
    if(!sigs.empty()) block_signals(sigs);

    if(auto* state = registry_.at(tid))
    {
        // Stop all POSIX timers before clearing the running flag so no new signals
        // are delivered after stop() (DEC-3: separate realtime + cputime slots).
        if(state->realtime_trigger().has_value()) state->realtime_trigger()->stop();
        if(state->cputime_trigger().has_value()) state->cputime_trigger()->stop();
        if(state->overflow_trigger().has_value()) state->overflow_trigger()->stop();

        state->stop();
        // Busy-wait for any in-flight handler to complete (architecture § 7, 5s timeout).
        if(!state->wait_for_in_flight_zero(5000))
        {
            LOG_DEBUG("Warning: in-flight handler for thread {} did not finish in 5s",
                      tid);
        }

        // Drain any remaining ring-buffer records to the offload store.
        // Safe here: signals are blocked and in-flight count is zero.
        offload_.write(tid, state->ring_buffer(), fatal_);
    }

    // Clear thread-local signal-handler pointers so a stale signal after
    // state destruction is a no-op. No-op in generic template; explicit
    // specialization for default_sampling_policies clears tl_sampler_state_vp,
    // tl_offload_vp, and tl_logical_tid (defined in
    // library/sampling_production_policies/sampling_service_production_hooks.hpp).
    shutdown_production_wiring(tid);

    // Destroy the per-thread state for this tid.
    registry_.erase(tid);

    LOG_DEBUG("Sampler destroyed for thread {}...", tid);
    return sigs;
}

// ── post_process ──────────────────────────────────────────────────────────

template <class Policies>
void
sampling_service<Policies>::post_process()
{
    sample_parser   parser;
    symbol_resolver resolver;

    // Track any exception to re-throw after cleanup.
    std::exception_ptr pending_exc;

    // Per-thread fan-out: drain offload → parse → report_writer + perfetto_sink +
    // trace_sink. DEC-9: offload.read() must precede report_writer.write*(). DEC-9:
    // offload.reset() must be called in all paths (exception safety). trace_sink_
    // receives post-parse resolved samples (store_timer/store_overflow) so data_processor
    // gets symbol names + real time ranges, not raw PC addresses. Use offload_.tids()
    // instead of registry_.each() because shutdown() erases registry entries before
    // post_process() runs.
    for(int64_t tid : offload_.tids())
    {
        if(pending_exc) break;  // skip remaining tids after first failure

        auto records = offload_.read(tid);
        LOG_DEBUG("[post_process] tid={} offload returned {} records", tid,
                  records.size());
        if(records.empty()) continue;

        // Split TIMER vs OVERFLOW records.
        // First TIMER record is the init/base record (provides beg_ns for sample1).
        std::vector<backtrace_record> timer_raw;
        std::vector<backtrace_record> overflow_raw;
        timer_raw.reserve(records.size());
        overflow_raw.reserve(records.size());
        for(auto const& r : records)
        {
            if(r.trigger == trigger_type::TIMER)
                timer_raw.push_back(r);
            else
                overflow_raw.push_back(r);
        }

        try
        {
            if(timer_raw.size() >= 2)
            {
                backtrace_record              init_rec = timer_raw.front();
                std::vector<backtrace_record> tail(timer_raw.begin() + 1,
                                                   timer_raw.end());
                auto                          timer_samples =
                    parser.parse_timer(tid, init_rec, tail, pause_registry_);
                if(!timer_samples.empty())
                {
                    // R-A1: resolve raw PCs to symbol names post-signal.
                    for(auto& s : timer_samples)
                        for(auto& f : s.stack)
                            if(f.name.empty()) f.name = resolver.resolve(f.address);
                    report_writer_.write_timer_samples(tid, timer_samples);
                    perfetto_sink_.emit_timer(tid, nullptr, timer_samples);
                    trace_sink_.store_timer(tid, timer_samples);
                }
            }

            if(!overflow_raw.empty())
            {
                auto overflow_samples =
                    parser.parse_overflow(tid, overflow_raw, pause_registry_);
                if(!overflow_samples.empty())
                {
                    for(auto& s : overflow_samples)
                        for(auto& f : s.stack)
                            if(f.name.empty()) f.name = resolver.resolve(f.address);
                    report_writer_.write_overflow_samples(tid, overflow_samples);
                    perfetto_sink_.emit_overflow(tid, nullptr, overflow_samples);
                    trace_sink_.store_overflow(tid, overflow_samples);
                }
            }
        } catch(...)
        {
            pending_exc = std::current_exception();
        }
    }

    // Final teardown: open output files (production), flush writer, reset offload.
    // DEC-9 ordering: open → flush → reset.
    // offload_.reset() must run even if flush() throws (C-15).
    open_report_writer_streams();
    try
    {
        report_writer_.flush();
    } catch(...)
    {
        if(!pending_exc) pending_exc = std::current_exception();
    }
    offload_.reset();

    // Propagate any exception from the fan-out loop or flush via fatal_error_policy
    // so production callers get the noreturn abort and test doubles get
    // sampling_fatal_error.
    if(pending_exc)
    {
        try
        {
            std::rethrow_exception(pending_exc);
        } catch(std::exception const& e)
        {
            fatal_.fatal(__FILE__, __LINE__, e.what());
        } catch(...)
        {
            fatal_.fatal(__FILE__, __LINE__, "unknown exception in post_process fan-out");
        }
    }
}

// ── postfork_parent_reinit / postfork_child_cleanup ───────────────────────

template <class Policies>
void
sampling_service<Policies>::postfork_parent_reinit()
{
    // AC-17: parent process — timers survive fork() per POSIX, so no re-arming needed.
    // Delegate to the PMC layer when process sampling is active (production hook).
    postfork_production_parent_reinit();
}

template <class Policies>
void
sampling_service<Policies>::postfork_child_cleanup()
{
    // NFR-TS-5: block sampling signals on the calling thread BEFORE touching any state
    // so no signal handler fires against partially-destroyed per-thread state in the
    // child.
    {
        std::set<int> all_sigs;
        registry_.each([&all_sigs](int64_t /*tid*/, thread_sampler_state<Policies>& s) {
            for(int sig : s.signal_types())
                all_sigs.insert(sig);
        });
        if(!all_sigs.empty()) block_signals(all_sigs);
    }

    // Stop all POSIX timers across all threads — inherited file descriptors must be
    // closed in the child to avoid interfering with the parent's timer delivery.
    registry_.each([](int64_t /*tid*/, thread_sampler_state<Policies>& s) {
        if(s.realtime_trigger().has_value()) s.realtime_trigger()->stop();
        if(s.cputime_trigger().has_value()) s.cputime_trigger()->stop();
        if(s.overflow_trigger().has_value()) s.overflow_trigger()->stop();
    });

    // Drop all per-thread state without calling post_process (AC-20).
    registry_.reset();

    // Delegate to the PMC layer when process sampling is active (production hook).
    postfork_production_child_cleanup();

    LOG_DEBUG("[postfork_child_cleanup] child process sampling state released");
}

// ── postfork production hooks — no-op generic definitions ─────────────────

template <class Policies>
void
sampling_service<Policies>::postfork_production_parent_reinit()
{
    // No-op in generic template. Production specialization for
    // default_sampling_policies calls pmc::postfork_parent_reinit() when applicable.
}

template <class Policies>
void
sampling_service<Policies>::postfork_production_child_cleanup()
{
    // No-op in generic template. Production specialization for
    // default_sampling_policies calls pmc::postfork_child_cleanup() when applicable.
}

// ── Production wiring hooks — generic (no-op) definitions ─────────────────
// These are called from setup() and shutdown(). The generic template does nothing;
// explicit full specializations for default_sampling_policies live in
// library/sampling_production_policies/sampling_service_production_hooks.hpp
// and are included from library/sampling_production_policies.hpp.

template <class Policies>
bool
sampling_service<Policies>::setup_check_thread_guards(int64_t /*tid*/)
{
    return true;  // proceed
}

template <class Policies>
void
sampling_service<Policies>::setup_production_wiring(
    int64_t /*tid*/, thread_sampler_state<Policies>* /*state*/,
    std::set<int> const& /*sigs*/)
{}

template <class Policies>
void
sampling_service<Policies>::shutdown_production_wiring(int64_t /*tid*/)
{}

template <class Policies>
void
sampling_service<Policies>::open_report_writer_streams()
{}

}  // namespace rocprofsys::sampling
