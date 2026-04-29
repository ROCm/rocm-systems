// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Template method bodies for sampling_service<Policies, ProductionHooks, TestHooks>.
// Included from sampling_service.hpp after the class definition.
// May use platform types (sigset_t, etc.) via the signal_dispatcher policy.

#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sample_parser.hpp"
#include "sampling/src/sampling_config_fwd.hpp"
#include "sampling/src/signal_set.hpp"
#include "sampling/src/symbol_resolver.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <time.h>
#include <unistd.h>

namespace rocprofsys::sampling
{

// ── Construction / Destruction ─────────────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
sampling_service<Policies, ProductionHooks, TestHooks>::sampling_service()
: pause_registry_(clock_)
, duration_controller_([this]() {
    duration_disabled_ = true;
    block_samples();
})
{
    using namespace detail;
    static_assert(is_clock_policy_v<clock>, "Policies::clock must satisfy ClockPolicy "
                                            "(requires now_ns() and now_steady())");
    static_assert(is_timer_trigger_policy_v<timer_trigger>,
                  "Policies::timer_trigger must satisfy TimerTriggerPolicy "
                  "(requires start(), stop(), is_armed())");
    static_assert(is_overflow_trigger_policy_v<overflow_trigger>,
                  "Policies::overflow_trigger must satisfy OverflowTriggerPolicy "
                  "(requires start(), stop(), is_open())");
    static_assert(is_signal_dispatcher_policy_v<signal_dispatcher>,
                  "Policies::signal_dispatcher must satisfy SignalDispatcherPolicy "
                  "(requires apply_sigmask(int, void const*, void*))");
    static_assert(is_unwinder_policy_v<unwinder>,
                  "Policies::unwinder must satisfy UnwinderPolicy "
                  "(requires unwind(void const*) and static valid_pc(uintptr_t))");
    static_assert(is_emitter_policy_v<offload>,
                  "Policies::offload must satisfy EmitterPolicy "
                  "(requires read(), tids(), reset(), erase())");
    static_assert(is_trace_sink_policy_v<trace_sink>,
                  "Policies::trace_sink must satisfy TraceSinkPolicy "
                  "(requires store_timer() and store_overflow())");
    static_assert(is_perfetto_sink_policy_v<perfetto_sink>,
                  "Policies::perfetto_sink must satisfy PerfettoSinkPolicy "
                  "(requires emit_timer() and emit_overflow())");
    static_assert(is_report_writer_policy_v<report_writer>,
                  "Policies::report_writer must satisfy ReportWriterPolicy "
                  "(requires write_timer_samples(), write_overflow_samples(), flush())");
    static_assert(is_fatal_error_policy_v<fatal_error>,
                  "Policies::fatal_error must satisfy FatalErrorPolicy "
                  "(requires non-void type with fatal() template method)");
    static_assert(is_production_hooks_policy_v<ProductionHooks>,
                  "ProductionHooks must satisfy ProductionHooksPolicy "
                  "(requires bool check_thread_guards(int64_t))");
    static_assert(
        is_test_hooks_policy_v<TestHooks>,
        "TestHooks must satisfy TestHooksPolicy "
        "(requires override_duration_disabled / _causal_mode / _child_process)");
}

template <class Policies, class ProductionHooks, class TestHooks>
sampling_service<Policies, ProductionHooks, TestHooks>::~sampling_service() = default;

// ── is_paused / is_blocked ─────────────────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
bool
sampling_service<Policies, ProductionHooks, TestHooks>::is_paused() const noexcept
{
    return pause_registry_.is_paused();
}

template <class Policies, class ProductionHooks, class TestHooks>
bool
sampling_service<Policies, ProductionHooks, TestHooks>::is_blocked() const noexcept
{
    return blocked_.load(std::memory_order_relaxed);
}

template <class Policies, class ProductionHooks, class TestHooks>
size_t
sampling_service<Policies, ProductionHooks, TestHooks>::dropped_samples() const noexcept
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

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::block_samples()
{
    LOG_DEBUG("Blocking sampling...");
    blocked_.store(true, std::memory_order_release);
}

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::unblock_samples()
{
    LOG_DEBUG("Unblocking sampling...");
    blocked_.store(false, std::memory_order_release);
}

// ── block_signals / unblock_signals ───────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::apply_signal_mask(
    int how, std::set<int> sigs, char const* verb_capitalized)
{
    auto calling_tid = static_cast<int64_t>(::gettid());
    if(sigs.empty()) sigs = get_signal_types(calling_tid);
    if(sigs.empty())
    {
        // Lowercase form for "No signals to {block,unblock}..." parity with legacy.
        std::string lower{ verb_capitalized };
        if(!lower.empty()) lower[0] = static_cast<char>(std::tolower(lower[0]));
        LOG_DEBUG("No signals to {}...", lower);
        return;
    }

    LOG_DEBUG("{}ing signals [{}] on thread #{}...", verb_capitalized,
              join_with_comma(sigs), calling_tid);

    signal_set ss(sigs);
    int        err = signal_dispatcher_.apply_sigmask(how, ss.get(), nullptr);
    if(err != 0)
    {
        fatal_.fatal(__FILE__, __LINE__, "pthread_sigmask failed: errno={}", err);
    }
}

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::block_signals(std::set<int> sigs)
{
    apply_signal_mask(SIG_BLOCK, std::move(sigs), "Block");
}

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::unblock_signals(
    std::set<int> sigs)
{
    apply_signal_mask(SIG_UNBLOCK, std::move(sigs), "Unblock");
}

// ── pause / resume ────────────────────────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::pause()
{
    if(pause_registry_.pause())
    {
        block_samples();
    }
}

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::resume()
{
    if(pause_registry_.resume())
    {
        unblock_samples();
    }
}

// ── get_signal_types ──────────────────────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
std::set<int>
sampling_service<Policies, ProductionHooks, TestHooks>::get_signal_types(int64_t tid)
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

template <class Policies, class ProductionHooks, class TestHooks>
std::set<int>
sampling_service<Policies, ProductionHooks, TestHooks>::setup(int64_t tid)
{
    // AC-19: causal profiling guard. test_hooks_ override lets unit tests
    // exercise the guard without enabling actual causal profiling.
    if(test_hooks_.override_causal_mode() || rocprofsys::get_use_causal())
    {
        throw std::runtime_error("Internal error! configuring sampling not permitted "
                                 "when causal profiling is enabled");
    }

    // AC-5: duration already fired (production state) OR test override.
    if(duration_disabled_ || test_hooks_.override_duration_disabled()) return {};

    // I-12: production thread-state guards (ThreadState::Disabled, offset thread).
    // Generic ProductionHooks default returns true; real_production_hooks checks
    // get_thread_state() and thread_info::get() (requires main-lib headers).
    if(!production_hooks_.check_thread_guards(tid)) return {};

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

    // L02 — matches legacy: "Requesting allocator for sampler on thread {}"
    LOG_DEBUG("Requesting allocator for sampler on thread {}", tid);

    LOG_DEBUG("Configuring sampler for thread {}", tid);

    // Create per-thread state, record signal set, mark running (C-2).
    registry_.emplace(tid);
    auto* state = registry_.at(tid);
    if(state)
    {
        state->set_signal_types(sigs);
        state->start();
    }

    // Production wiring: TLS pointer setup, timer arming, overflow trigger config.
    // Generic ProductionHooks default is a no-op; real_production_hooks does the
    // real work (requires main-lib headers; defined in
    // sampling/policies/real_production_hooks.hpp).
    production_hooks_.setup_wiring(*this, tid, state, sigs);

    // Block signals on the calling thread before the trigger is armed.
    block_signals(sigs);

    return sigs;
}

// ── shutdown ──────────────────────────────────────────────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
std::set<int>
sampling_service<Policies, ProductionHooks, TestHooks>::shutdown(int64_t tid)
{
    // AC-20: child process — release state without per-tid processing.
    if(child_process_mode_ || test_hooks_.override_child_process()) return {};

    LOG_DEBUG("Stopping sampler for thread {}...", tid);

    // Retrieve the signal set before destroying state.
    auto sigs = get_signal_types(tid);

    // Block signals on the calling thread so no new samples arrive.
    if(!sigs.empty()) block_signals(sigs);

    if(auto* state = registry_.at(tid))
    {
        // Stop all POSIX timers before clearing the running flag so no new signals
        // are delivered after stop() (DEC-3: separate realtime + cputime slots).
        state->stop_all_triggers();

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

    // Variant 2: parse + resolve + emit to trace_cache immediately.
    // Generic ProductionHooks default is a no-op; real_production_hooks reads from
    // offload_, parses, resolves symbols, and emits backtrace_region_sample to
    // trace_cache.
    production_hooks_.emit_resolved(*this, tid);

    // Clear thread-local signal-handler pointers so a stale signal after
    // state destruction is a no-op.
    production_hooks_.shutdown_wiring(*this, tid);

    // Destroy the per-thread state for this tid.
    registry_.erase(tid);

    LOG_DEBUG("Sampler destroyed for thread {}...", tid);
    return sigs;
}

// ── postfork_parent_reinit / postfork_child_cleanup ───────────────────────

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::postfork_parent_reinit()
{
    // AC-17: parent process — timers survive fork() per POSIX, so no re-arming needed.
    // Delegate to the PMC layer when process sampling is active (production hook).
    production_hooks_.postfork_parent_reinit(*this);
}

template <class Policies, class ProductionHooks, class TestHooks>
void
sampling_service<Policies, ProductionHooks, TestHooks>::postfork_child_cleanup()
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
        s.stop_all_triggers();
    });

    // Drop all per-thread state without per-tid processing (AC-20).
    registry_.reset();

    // Delegate to the PMC layer when process sampling is active (production hook).
    production_hooks_.postfork_child_cleanup(*this);

    LOG_DEBUG("[postfork_child_cleanup] child process sampling state released");
}

}  // namespace rocprofsys::sampling
