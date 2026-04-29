// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// real_production_hooks — production specialization of the ProductionHooksPolicy
// concept. Holds no state itself; each hook reaches sampling_service members
// (offload, fatal_error, pause_registry, perfetto_sink, duration_controller)
// through public service accessors.
//
// Lives in library/ because the wiring depends on main-lib symbols
// (core/config.hpp, core/perf.hpp, core/state.hpp, library/thread_info.hpp,
// linux/perf_event.h, library/pmc/sampler.hpp, trace_cache).
// Included from sampling/default_policies.hpp.

#include "sampling/data/track_name.hpp"
#include "sampling/policies/real_trace_cache_sink.hpp"
#include "sampling/policies/stack_frame_json.hpp"
#include "sampling/policies/tl_state.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/linux/symbol_resolver.hpp"
#include "sampling/src/sample_parser.hpp"

#include "core/config.hpp"
#include "core/perf.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "library/pmc/sampler.hpp"
#include "library/thread_info.hpp"

#include "rocprofiler-systems/categories.h"

#include <dlfcn.h>
#include <libunwind.h>
#include <linux/perf_event.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace rocprofsys::sampling
{

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

// AC-16: register sampling category strings exactly once per process.
inline void
register_sampling_categories_once()
{
    static std::once_flag s_init;
    std::call_once(s_init, [] {
        auto& reg = trace_cache::get_metadata_registry();
        reg.add_string("timer_sampling");
        reg.add_string("overflow_sampling");
    });
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

// ── ProductionHooksPolicy implementation ───────────────────────────────────

class real_production_hooks
{
public:
    // I-12 thread-state guards. Returns false to skip setup() for a thread.
    bool check_thread_guards(int64_t tid)
    {
        // I-12, Guard 1: thread state disabled.
        if(get_thread_state() == ThreadState::Disabled) return false;

        // I-12, Guard 2: offset thread — internally-created, not user code.
        const auto& info = thread_info::get(tid, SequentTID);
        if(info && info->is_offset) return false;

        return true;
    }

    // Wire per-thread TLS, arm POSIX timers + perf_event overflow trigger,
    // start the duration controller (tid==0 only), register trace_cache tracks.
    template <class Service>
    void setup_wiring(Service& svc, int64_t tid, typename Service::thread_state_t* state,
                      std::set<int> const& sigs)
    {
        if(!state) return;

        using Policies = typename Service::policies;

        // Wire per-thread signal-handler state. Must be set on the calling
        // thread so the handler can access them without a mutex (NFR-TS-2).
        using tls        = tl_state<Policies>;
        tls::sampler     = state;
        tls::offload     = &svc.get_offload();
        tls::logical_tid = tid;

        pid_t sys_tid   = static_cast<pid_t>(::gettid());
        int   rt_sig    = rocprofsys::get_sampling_realtime_signal();
        int   cpu_sig   = rocprofsys::get_sampling_cputime_signal();
        int   ovfl_sig  = rocprofsys::get_sampling_overflow_signal();
        auto& rt_slot   = state->realtime_trigger();
        auto& cpu_slot  = state->cputime_trigger();
        auto& ovfl_slot = state->overflow_trigger();

        // AC-1: arm realtime timer if subscribed (independent of cputime — DEC-3).
        if(sigs.count(rt_sig) > 0)
        {
            double rt_freq = rocprofsys::get_sampling_realtime_freq();
            rt_slot.emplace();
            rt_slot->configure(tid, sys_tid, rt_sig, CLOCK_REALTIME, rt_freq,
                               rocprofsys::get_sampling_realtime_delay());
            rt_slot->start();
            LOG_DEBUG("thread {} realtime timer armed={} (sig={})", tid,
                      rt_slot->is_armed(), rt_sig);
            // L06 — matches legacy: "[SIG{}] Sampler for thread {} will be triggered
            // {:.1f}x per second of {}-time (every {:.3e} milliseconds)..."
            double rt_period_ms = (rt_freq > 0.0) ? (1000.0 / rt_freq) : 0.0;
            LOG_INFO("[SIG{}] Sampler for thread {} will be triggered {:.1f}x per "
                     "second of {}-time (every {:.3e} milliseconds)...",
                     rt_sig, tid, rt_freq, "wall", rt_period_ms);
        }
        // AC-2: arm cputime timer if subscribed (independent of realtime — DEC-3).
        if(sigs.count(cpu_sig) > 0)
        {
            double cpu_freq = rocprofsys::get_sampling_cputime_freq();
            cpu_slot.emplace();
            cpu_slot->configure(tid, sys_tid, cpu_sig, CLOCK_THREAD_CPUTIME_ID, cpu_freq,
                                rocprofsys::get_sampling_cputime_delay());
            cpu_slot->start();
            LOG_DEBUG("thread {} cputime timer armed={} (sig={} sys_tid={})", tid,
                      cpu_slot->is_armed(), cpu_sig, sys_tid);
            // L06 — matches legacy: same pattern for CPU-time
            double cpu_period_ms = (cpu_freq > 0.0) ? (1000.0 / cpu_freq) : 0.0;
            LOG_INFO("[SIG{}] Sampler for thread {} will be triggered {:.1f}x per "
                     "second of {}-time (every {:.3e} milliseconds)...",
                     cpu_sig, tid, cpu_freq, "CPU", cpu_period_ms);
        }

        if(sigs.count(ovfl_sig) > 0)
        {
            perf_event_attr pe_attr        = {};
            auto            event_name_opt = rocprofsys::get_setting_value<std::string>(
                "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT");
            std::string event_name = event_name_opt.value_or("PERF_COUNT_SW_CPU_CLOCK");
            double      ovfl_freq  = rocprofsys::get_sampling_overflow_freq();
            rocprofsys::perf::config_overflow_sampling(pe_attr, event_name, ovfl_freq);
            pe_attr.sample_type   = PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN;
            pe_attr.wakeup_events = 1;

            ovfl_slot.emplace();
            ovfl_slot->configure(tid, sys_tid, ovfl_sig, &pe_attr, svc.get_fatal_error());
            ovfl_slot->start();
            LOG_DEBUG("thread {} overflow armed={} (sig={})", tid, ovfl_slot->is_open(),
                      ovfl_sig);
            // L05 — matches legacy: "[SIG{}] Sampler for thread {} will be triggered
            // every {:.1f} {} events..."
            LOG_INFO("[SIG{}] Sampler for thread {} will be triggered every {:.1f} "
                     "{} events...",
                     ovfl_sig, tid, ovfl_freq, event_name);
        }

        // Start the duration controller once on the main thread (tid==0).
        if(tid == 0)
        {
            double dur = rocprofsys::get_sampling_duration();
            if(dur > 0.0) svc.duration_controller().start(dur);
        }

        // Register per-thread sampling tracks in the trace_cache metadata registry so
        // data_processor can resolve track names when inserting backtrace_region_sample
        // records. Must happen regardless of use_perfetto.
        const auto& thread_inf = thread_info::get(tid, SequentTID);
        if(thread_inf)
        {
            const std::size_t sys_id = thread_inf->index_data->system_value;
            const std::size_t seq_id = thread_inf->index_data->sequent_value;
            const std::string timer_track =
                make_thread_track_name(timer_track_tag{}, seq_id, sys_id);
            const std::string overflow_track =
                make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);

            auto& reg = trace_cache::get_metadata_registry();

            detail::register_sampling_categories_once();

            // AC-16: register thread info so rocpd has ppid/pid/tid for this thread.
            reg.add_thread_info({ getppid(), getpid(), sys_id, 0, 0, "{}" });

            reg.add_track({ timer_track, sys_id, "{}" });
            reg.add_track({ overflow_track, sys_id, "{}" });

            // TF-3 follow-up: per-thread process-sampling counter tracks emitted
            // by real_trace_cache_sink::store_thread_counters as
            // pmc_event_with_sample. Prefixes come from the sink so the two
            // sides cannot drift.
            const std::string seq_suffix = " [" + std::to_string(seq_id) + "]";
            for(char const* prefix : real_trace_cache_sink::thread_counter_prefixes)
            {
                reg.add_track({ std::string{ prefix } + seq_suffix, sys_id, "{}" });
            }

            LOG_DEBUG("thread {} registered trace_cache tracks: '{}' / '{}'", tid,
                      timer_track, overflow_track);
        }
    }

    // Clear thread-local signal-handler pointers so a stale signal after state
    // destruction is a no-op.
    template <class Service>
    void shutdown_wiring(Service& /*svc*/, int64_t /*tid*/) noexcept
    {
        using Policies   = typename Service::policies;
        using tls        = tl_state<Policies>;
        tls::sampler     = nullptr;
        tls::offload     = nullptr;
        tls::logical_tid = -1;
    }

    // Variant 2 (Task #30): parse + resolve + emit ring records to trace_cache.
    // Called from shutdown(tid) after offload_.write(). Reads raw records from
    // offload_, parses with sample_parser, resolves symbols, emits
    // backtrace_region_sample to trace_cache::buffer_storage, then clears
    // the tid from offload_ to prevent double-emission when post_process()
    // iterates offload_.tids().
    template <class Service>
    void emit_resolved(Service& svc, int64_t tid)
    {
        // L43 — refactor variant of legacy "[{}] Post-processing data for timemory..."
        LOG_DEBUG("[{}] Post-processing data for native report...", tid);

        auto records = svc.get_offload().read(tid);
        if(records.empty()) return;

        auto [timer_raw, overflow_raw] = detail::split_records(records);

        sample_parser   parser;
        symbol_resolver resolver;
        const bool      legacy = rocprofsys::get_use_sampling_trace_legacy();
        auto&           sink   = svc.get_trace_sink();

        // Timer + cputime + thread-counter tracks (AC-11 discards single-sample buffer).
        if(timer_raw.size() >= 2)
        {
            backtrace_record              init_rec = timer_raw.front();
            std::vector<backtrace_record> tail(timer_raw.begin() + 1, timer_raw.end());
            auto                          timer_samples =
                parser.parse_timer(tid, init_rec, tail, svc.pause_registry());
            if(!timer_samples.empty())
            {
                detail::resolve_stacks(timer_samples, resolver);
                sink.store_timer(tid, timer_samples);
                sink.store_thread_counters(tid, timer_samples);
                if(legacy)
                    svc.get_perfetto_sink().emit_timer(tid, nullptr, timer_samples);
            }
        }

        // Overflow region samples.
        if(!overflow_raw.empty())
        {
            auto overflow_samples =
                parser.parse_overflow(tid, overflow_raw, svc.pause_registry());
            if(!overflow_samples.empty())
            {
                detail::resolve_stacks(overflow_samples, resolver);
                sink.store_overflow(tid, overflow_samples);
                if(legacy)
                    svc.get_perfetto_sink().emit_overflow(tid, nullptr, overflow_samples);
            }
        }

        svc.get_offload().erase(tid);
    }

    template <class Service>
    void postfork_parent_reinit(Service& /*svc*/)
    {
        if(rocprofsys::config::get_use_process_sampling() &&
           rocprofsys::config::get_use_amd_smi())
        {
            LOG_DEBUG(
                "[postfork_parent_reinit] delegating to pmc::postfork_parent_reinit");
            rocprofsys::pmc::postfork_parent_reinit();
        }
    }

    template <class Service>
    void postfork_child_cleanup(Service& /*svc*/)
    {
        if(rocprofsys::config::get_use_process_sampling() &&
           rocprofsys::config::get_use_amd_smi())
        {
            LOG_DEBUG(
                "[postfork_child_cleanup] delegating to pmc::postfork_child_cleanup");
            rocprofsys::pmc::postfork_child_cleanup();
        }
    }
};

}  // namespace rocprofsys::sampling
