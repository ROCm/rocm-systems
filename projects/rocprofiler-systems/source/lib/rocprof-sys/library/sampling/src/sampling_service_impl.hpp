// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Template method bodies for sampling_service<Policies>.
// Included from sampling_service.hpp after the class definition.
// May use platform types (sigset_t, etc.) via the signal_dispatcher policy.
//
// The lifecycle orchestration methods at the bottom of this file
// (check_thread_guards, do_setup_wiring, do_shutdown_wiring, do_emit_resolved,
// do_postfork_parent_reinit, do_postfork_child_cleanup) were folded out of
// real_production_hooks. Configuration is read from sampling_config (injected
// at construction). Remaining main-library deps (core/perf.hpp,
// core/trace_cache/cache_manager.hpp, library/thread_info.hpp) plus Linux
// kernel headers (linux/perf_event.h, libunwind.h) are still included here;
// these will be removed in subsequent decoupling tasks (Tasks 4–6).

#include "sampling/data/track_name.hpp"
#include "sampling/data/track_traits.hpp"
#include "sampling/policies/real_trace_cache_sink.hpp"
#include "sampling/policies/stack_frame_json.hpp"
#include "sampling/policies/tl_state.hpp"
#include "sampling/src/linux/signal_set.hpp"
#include "sampling/src/linux/symbol_resolver.hpp"
#include "sampling/src/pause_interval_registry.hpp"
#include "sampling/src/sample_parser.hpp"

#include "rocprofiler-systems/categories.h"

#include "logger/debug.hpp"

#include <dlfcn.h>
#include <libunwind.h>
#include <linux/perf_event.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rocprofsys::sampling
{

// ── Construction / Destruction ─────────────────────────────────────────────

template <class Policies>
sampling_service<Policies>::sampling_service(sampling_config config)
: config_(std::move(config))
, thread_info_resolver_(config_.resolve_thread_info)
, trace_sink_(thread_info_resolver_)
, perfetto_sink_(thread_info_resolver_, config_.use_perfetto,
                 config_.perfetto_annotations)
, pause_registry_(clock_)
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
                  "(requires store_timer(), store_overflow(), store_thread_counters())");
    static_assert(is_perfetto_sink_policy_v<perfetto_sink>,
                  "Policies::perfetto_sink must satisfy PerfettoSinkPolicy "
                  "(requires emit_timer() and emit_overflow())");
    static_assert(is_report_writer_policy_v<report_writer>,
                  "Policies::report_writer must satisfy ReportWriterPolicy "
                  "(requires write_timer_samples(), write_overflow_samples(), flush())");
    static_assert(is_fatal_error_policy_v<fatal_error>,
                  "Policies::fatal_error must satisfy FatalErrorPolicy "
                  "(requires non-void type with fatal() template method)");
}

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
sampling_service<Policies>::apply_signal_mask(int how, std::set<int> sigs,
                                              char const* verb_capitalized)
{
    auto calling_tid = static_cast<int64_t>(config_.get_sys_tid());
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

template <class Policies>
void
sampling_service<Policies>::block_signals(std::set<int> sigs)
{
    apply_signal_mask(SIG_BLOCK, std::move(sigs), "Block");
}

template <class Policies>
void
sampling_service<Policies>::unblock_signals(std::set<int> sigs)
{
    apply_signal_mask(SIG_UNBLOCK, std::move(sigs), "Unblock");
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
    auto sigs          = config_.resolve_signals(tid);
    signal_types_[tid] = sigs;
    return sigs;
}

// ── setup ─────────────────────────────────────────────────────────────────

template <class Policies>
std::set<int>
sampling_service<Policies>::setup(int64_t tid)
{
    // AC-19: causal profiling guard.
    if(config_.use_causal)
    {
        throw std::runtime_error("Internal error! configuring sampling not permitted "
                                 "when causal profiling is enabled");
    }

    // AC-5: duration already fired (production state) OR test override.
    if(duration_disabled_) return {};

    // I-12: production thread-state guards (ThreadState::Disabled, offset thread).
    if(!check_thread_guards(tid)) return {};

    // Compute signal set for this thread (one lock acquisition — C-6 fix).
    std::set<int> sigs;
    {
        std::lock_guard<std::mutex> lk(signal_types_mutex_);
        auto                        it = signal_types_.find(tid);
        if(it == signal_types_.end())
        {
            sigs               = config_.resolve_signals(tid);
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
    do_setup_wiring(tid, state, sigs);

    // Block signals on the calling thread before the trigger is armed.
    block_signals(sigs);

    return sigs;
}

// ── shutdown ──────────────────────────────────────────────────────────────

template <class Policies>
std::set<int>
sampling_service<Policies>::shutdown(int64_t tid)
{
    // AC-20: child process — release state without per-tid processing.
    if(child_process_mode_) return {};

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
    do_emit_resolved(tid);

    // Clear thread-local signal-handler pointers so a stale signal after
    // state destruction is a no-op.
    do_shutdown_wiring(tid);

    // Destroy the per-thread state for this tid.
    registry_.erase(tid);

    LOG_DEBUG("Sampler destroyed for thread {}...", tid);
    return sigs;
}

// ── postfork_parent_reinit / postfork_child_cleanup ───────────────────────

template <class Policies>
void
sampling_service<Policies>::postfork_parent_reinit()
{
    // AC-17: parent process — timers survive fork() per POSIX, so no re-arming needed.
    // Delegate to the PMC layer when process sampling is active.
    do_postfork_parent_reinit();
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
        s.stop_all_triggers();
    });

    // Drop all per-thread state without per-tid processing (AC-20).
    registry_.reset();

    // Delegate to the PMC layer when process sampling is active.
    do_postfork_child_cleanup();

    LOG_DEBUG("[postfork_child_cleanup] child process sampling state released");
}

// ── Lifecycle orchestration (folded out of real_production_hooks) ─────────

namespace detail
{

// TF-4 helper: resolve a stack-frame IP to a human-readable label.
// Cascade: cached resolver (dladdr) → libunwind proc-name → dladdr fname
// (e.g. "[libgcc_s.so.1+0x1234]") → "<unresolved>" placeholder.
// Eliminates raw "0xADDR" hex strings from the TSV output entirely.
inline std::string
resolve_symbol(symbol_resolver& resolver, std::uintptr_t addr)
{
    if(addr == 0) return "<null>";

    auto name = resolver.resolve(addr);
    if(!name.empty()) return name;

    // libunwind reads ELF symtab directly — finds static binary symbols
    // that dladdr (dynamic-symtab only) misses.
    {
        char       buf[512] = {};
        unw_word_t off      = 0;
        if(::unw_get_proc_name_by_ip(unw_local_addr_space, static_cast<unw_word_t>(addr),
                                     buf, sizeof(buf), &off, nullptr) == 0 &&
           buf[0] != '\0')
        {
            std::string sym{ buf };
            resolver.inject(addr, sym);
            return resolver.resolve(addr);
        }
    }

    // Last-resort: report the containing module name + offset, instead of
    // a bare hex address. Covers vDSO / libgcc_s / signal-trampoline frames.
    {
        ::Dl_info info{};
        if(::dladdr(reinterpret_cast<void const*>(addr), &info) != 0 &&
           info.dli_fname != nullptr)
        {
            const char* slash  = std::strrchr(info.dli_fname, '/');
            const char* base   = slash ? slash + 1 : info.dli_fname;
            auto        offset = static_cast<std::uintptr_t>(addr) -
                          reinterpret_cast<std::uintptr_t>(info.dli_fbase);
            return "[" + std::string{ base } + "+0x" + fmt::format("{:X}", offset) + "]";
        }
    }

    return "<unresolved>";
}

// Split raw records into TIMER and OVERFLOW vectors.
inline std::pair<std::vector<backtrace_record>, std::vector<backtrace_record>>
split_records(std::vector<backtrace_record> const& records)
{
    std::pair<std::vector<backtrace_record>, std::vector<backtrace_record>> out;
    out.first.reserve(records.size());
    out.second.reserve(records.size());
    for(auto const& r : records)
    {
        if(r.trigger == trigger_type::TIMER)
            out.first.push_back(r);
        else
            out.second.push_back(r);
    }
    return out;
}

// Resolve unnamed stack frames in-place via the cascade resolver.
template <class Sample>
inline void
resolve_stacks(std::vector<Sample>& samples, symbol_resolver& resolver)
{
    for(auto& s : samples)
        for(auto& f : s.stack)
            if(f.name.empty()) f.name = resolve_symbol(resolver, f.address);
}

}  // namespace detail

// I-12 thread-state guards. Returns false to skip setup() for a thread.
template <class Policies>
bool
sampling_service<Policies>::check_thread_guards(int64_t tid)
{
    return config_.is_thread_eligible(tid);
}

template <class Policies>
void
sampling_service<Policies>::do_setup_wiring(int64_t tid, thread_state_t* state,
                                            std::set<int> const& sigs)
{
    if(!state) return;

    wire_tls(tid, state);
    arm_timer_triggers(tid, state, sigs);
    arm_overflow_trigger(tid, state, sigs);
    start_duration_controller(tid);
    register_trace_cache_tracks(tid);
}

template <class Policies>
void
sampling_service<Policies>::wire_tls(int64_t tid, thread_state_t* state)
{
    using tls        = tl_state<Policies>;
    tls::sampler     = state;
    tls::offload     = &offload_;
    tls::logical_tid = tid;
}

template <class Policies>
void
sampling_service<Policies>::arm_timer_triggers(int64_t tid, thread_state_t* state,
                                               std::set<int> const& sigs)
{
    pid_t sys_tid  = static_cast<pid_t>(config_.get_sys_tid());
    int   rt_sig   = config_.realtime_signal;
    int   cpu_sig  = config_.cputime_signal;
    auto& rt_slot  = state->realtime_trigger();
    auto& cpu_slot = state->cputime_trigger();

    if(sigs.count(rt_sig) > 0)
    {
        double rt_freq = config_.realtime_freq;
        rt_slot.emplace();
        rt_slot->configure(tid, sys_tid, rt_sig, CLOCK_REALTIME, rt_freq,
                           config_.realtime_delay);
        rt_slot->start();
        LOG_DEBUG("thread {} realtime timer armed={} (sig={})", tid, rt_slot->is_armed(),
                  rt_sig);
        double rt_period_ms = (rt_freq > 0.0) ? (1000.0 / rt_freq) : 0.0;
        LOG_INFO("[SIG{}] Sampler for thread {} will be triggered {:.1f}x per "
                 "second of {}-time (every {:.3e} milliseconds)...",
                 rt_sig, tid, rt_freq, "wall", rt_period_ms);
    }

    if(sigs.count(cpu_sig) > 0)
    {
        double cpu_freq = config_.cputime_freq;
        cpu_slot.emplace();
        cpu_slot->configure(tid, sys_tid, cpu_sig, CLOCK_THREAD_CPUTIME_ID, cpu_freq,
                            config_.cputime_delay);
        cpu_slot->start();
        LOG_DEBUG("thread {} cputime timer armed={} (sig={} sys_tid={})", tid,
                  cpu_slot->is_armed(), cpu_sig, sys_tid);
        double cpu_period_ms = (cpu_freq > 0.0) ? (1000.0 / cpu_freq) : 0.0;
        LOG_INFO("[SIG{}] Sampler for thread {} will be triggered {:.1f}x per "
                 "second of {}-time (every {:.3e} milliseconds)...",
                 cpu_sig, tid, cpu_freq, "CPU", cpu_period_ms);
    }
}

template <class Policies>
void
sampling_service<Policies>::arm_overflow_trigger(int64_t tid, thread_state_t* state,
                                                 std::set<int> const& sigs)
{
    int ovfl_sig = config_.overflow_signal;
    if(sigs.count(ovfl_sig) == 0) return;

    pid_t           sys_tid    = static_cast<pid_t>(config_.get_sys_tid());
    perf_event_attr pe_attr    = {};
    std::string     event_name = config_.overflow_event;
    double          ovfl_freq  = config_.overflow_freq;
    config_.configure_overflow_pe_attr(&pe_attr, event_name, ovfl_freq);
    pe_attr.sample_type   = PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN;
    pe_attr.wakeup_events = 1;

    auto& ovfl_slot = state->overflow_trigger();
    ovfl_slot.emplace();
    ovfl_slot->configure(tid, sys_tid, ovfl_sig, &pe_attr, fatal_);
    ovfl_slot->start();
    LOG_DEBUG("thread {} overflow armed={} (sig={})", tid, ovfl_slot->is_open(),
              ovfl_sig);
    LOG_INFO("[SIG{}] Sampler for thread {} will be triggered every {:.1f} "
             "{} events...",
             ovfl_sig, tid, ovfl_freq, event_name);
}

template <class Policies>
void
sampling_service<Policies>::start_duration_controller(int64_t tid)
{
    if(tid == 0)
    {
        double dur = config_.duration;
        if(dur > 0.0) duration_controller_.start(dur);
    }
}

template <class Policies>
void
sampling_service<Policies>::register_trace_cache_tracks(int64_t tid)
{
    auto thread_inf = thread_info_resolver_.resolve(tid);
    if(!thread_inf) return;

    const std::size_t sys_id = thread_inf->system_value;
    const std::size_t seq_id = thread_inf->sequent_value;
    const std::string timer_track =
        make_thread_track_name(timer_track_tag{}, seq_id, sys_id);
    const std::string overflow_track =
        make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);

    config_.register_sampling_categories();
    config_.register_thread_info(static_cast<int>(getppid()), static_cast<int>(getpid()),
                                 sys_id);
    config_.register_track(timer_track, sys_id);
    config_.register_track(overflow_track, sys_id);

    const std::string seq_suffix = " [" + std::to_string(seq_id) + "]";
    for(char const* prefix : thread_counter_prefixes)
    {
        config_.register_track(std::string{ prefix } + seq_suffix, sys_id);
    }

    LOG_DEBUG("thread {} registered trace_cache tracks: '{}' / '{}'", tid, timer_track,
              overflow_track);
}

// Clear thread-local signal-handler pointers so a stale signal after state
// destruction is a no-op.
template <class Policies>
void
sampling_service<Policies>::do_shutdown_wiring(int64_t /*tid*/) noexcept
{
    using tls        = tl_state<Policies>;
    tls::sampler     = nullptr;
    tls::offload     = nullptr;
    tls::logical_tid = -1;
}

// Variant 2 (Task #30): parse + resolve + emit ring records to trace_cache.
// Called from shutdown(tid) after offload_.write(). Reads raw records from
// offload_, parses with sample_parser, resolves symbols, emits
// backtrace_region_sample to trace_cache::buffer_storage, then clears the
// tid from offload_ to prevent double-emission when post_process() iterates
// offload_.tids().
template <class Policies>
void
sampling_service<Policies>::do_emit_resolved(int64_t tid)
{
    // L43 — refactor variant of legacy "[{}] Post-processing data for timemory..."
    LOG_DEBUG("[{}] Post-processing data for native report...", tid);

    auto records = offload_.read(tid);
    if(records.empty()) return;

    auto [timer_raw, overflow_raw] = detail::split_records(records);

    sample_parser   parser;
    symbol_resolver resolver;
    const bool      legacy = config_.trace_legacy;

    // Timer + cputime + thread-counter tracks (AC-11 discards single-sample buffer).
    if(timer_raw.size() >= 2)
    {
        backtrace_record              init_rec = timer_raw.front();
        std::vector<backtrace_record> tail(timer_raw.begin() + 1, timer_raw.end());
        auto timer_samples = parser.parse_timer(tid, init_rec, tail, pause_registry_);
        if(!timer_samples.empty())
        {
            detail::resolve_stacks(timer_samples, resolver);
            trace_sink_.store_timer(tid, timer_samples);
            trace_sink_.store_thread_counters(tid, timer_samples);
            if(legacy) perfetto_sink_.emit_timer(tid, nullptr, timer_samples);
        }
    }

    // Overflow region samples.
    if(!overflow_raw.empty())
    {
        auto overflow_samples = parser.parse_overflow(tid, overflow_raw, pause_registry_);
        if(!overflow_samples.empty())
        {
            detail::resolve_stacks(overflow_samples, resolver);
            trace_sink_.store_overflow(tid, overflow_samples);
            if(legacy) perfetto_sink_.emit_overflow(tid, nullptr, overflow_samples);
        }
    }

    offload_.erase(tid);
}

template <class Policies>
void
sampling_service<Policies>::do_postfork_parent_reinit()
{
    if(config_.use_process_sampling && config_.use_amd_smi)
    {
        LOG_DEBUG("[postfork_parent_reinit] delegating to pmc::postfork_parent_reinit");
        config_.postfork_parent_reinit();
    }
}

template <class Policies>
void
sampling_service<Policies>::do_postfork_child_cleanup()
{
    if(config_.use_process_sampling && config_.use_amd_smi)
    {
        LOG_DEBUG("[postfork_child_cleanup] delegating to pmc::postfork_child_cleanup");
        config_.postfork_child_cleanup();
    }
}

}  // namespace rocprofsys::sampling
