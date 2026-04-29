// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Explicit full specializations of sampling_service<default_sampling_policies>
// production wiring hooks. These replace the ROCPROFSYS_INTERNAL_BUILD guards
// that formerly lived in sampling/src/sampling_service_impl.hpp.
//
// Included from library/sampling_production_policies.hpp — never from the
// sampling/ subtree. Requires main-lib headers (core/config.hpp, core/perf.hpp,
// core/state.hpp, library/thread_info.hpp, linux/perf_event.h).

#include "sampling/data/stack_frame_json.hpp"
#include "sampling/data/track_name.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/sample_parser.hpp"
#include "sampling/src/symbol_resolver.hpp"

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
#include <string>
#include <utility>

namespace rocprofsys::sampling
{

// Typed thread-local signal-handler state lives in sampling/policies/tl_state.hpp.

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

// ── setup_check_thread_guards ──────────────────────────────────────────────

template <>
inline bool
sampling_service<default_sampling_policies>::setup_check_thread_guards(int64_t tid)
{
    // I-12, Guard 1: thread state disabled.
    if(get_thread_state() == ThreadState::Disabled) return false;

    // I-12, Guard 2: offset thread — internally-created, not user code.
    const auto& info = thread_info::get(tid, SequentTID);
    if(info && info->is_offset) return false;

    return true;
}

namespace detail
{

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

// Per-thread metric descriptor for the 4 process-sampling counter tracks
// emitted by emit_thread_counters. Single source of truth for the metric
// name, perfetto category, valid-bit index, and value transform.
struct thread_metric_descriptor
{
    char const* track_prefix;
    std::size_t category_enum;
    std::size_t valid_bit;
    double (*read)(backtrace_metrics_data const&);
};

inline constexpr std::array<thread_metric_descriptor, 4> thread_metric_descriptors = {
    thread_metric_descriptor{ "thread_cpu_time", ROCPROFSYS_CATEGORY_THREAD_CPU_TIME, 0,
                              [](backtrace_metrics_data const& m) {
                                  return static_cast<double>(m.cpu_ns) * 1.0e-9;
                              } },
    thread_metric_descriptor{ "thread_peak_memory",
                              ROCPROFSYS_CATEGORY_THREAD_PEAK_MEMORY, 1,
                              [](backtrace_metrics_data const& m) {
                                  return static_cast<double>(m.mem_peak_kb) / 1024.0;
                              } },
    thread_metric_descriptor{
        "thread_context_switch", ROCPROFSYS_CATEGORY_THREAD_CONTEXT_SWITCH, 2,
        [](backtrace_metrics_data const& m) { return static_cast<double>(m.ctx_swch); } },
    thread_metric_descriptor{
        "thread_page_fault", ROCPROFSYS_CATEGORY_THREAD_PAGE_FAULT, 3,
        [](backtrace_metrics_data const& m) { return static_cast<double>(m.page_flt); } },
};

}  // namespace detail

// ── setup_production_wiring ────────────────────────────────────────────────

template <>
inline void
sampling_service<default_sampling_policies>::setup_production_wiring(
    int64_t tid, thread_sampler_state<default_sampling_policies>* state,
    std::set<int> const& sigs)
{
    if(!state) return;

    // Wire per-thread signal-handler state.
    // Must be set on the calling thread so the handler can access them
    // without a mutex (NFR-TS-2).
    using tls        = tl_state<default_sampling_policies>;
    tls::sampler     = state;
    tls::offload     = &offload_;
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
        LOG_DEBUG("thread {} realtime timer armed={} (sig={})", tid, rt_slot->is_armed(),
                  rt_sig);
        // L06 — matches legacy: "[SIG{}] Sampler for thread {} will be triggered {:.1f}x
        // per second of {}-time (every {:.3e} milliseconds)..."
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
        ovfl_slot->configure(tid, sys_tid, ovfl_sig, &pe_attr, fatal_);
        ovfl_slot->start();
        LOG_DEBUG("thread {} overflow armed={} (sig={})", tid, ovfl_slot->is_open(),
                  ovfl_sig);
        // L05 — matches legacy: "[SIG{}] Sampler for thread {} will be triggered every
        // {:.1f} {} events..."
        LOG_INFO("[SIG{}] Sampler for thread {} will be triggered every {:.1f} "
                 "{} events...",
                 ovfl_sig, tid, ovfl_freq, event_name);
    }

    // Start the duration controller once on the main thread (tid==0).
    if(tid == 0)
    {
        double dur = rocprofsys::get_sampling_duration();
        if(dur > 0.0) duration_controller_.start(dur);
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
        // by emit_thread_counters as pmc_event_with_sample.
        const std::string seq_suffix = " [" + std::to_string(seq_id) + "]";
        for(auto const& descriptor : detail::thread_metric_descriptors)
        {
            reg.add_track(
                { std::string{ descriptor.track_prefix } + seq_suffix, sys_id, "{}" });
        }

        LOG_DEBUG("thread {} registered trace_cache tracks: '{}' / '{}'", tid,
                  timer_track, overflow_track);
    }
}

// ── shutdown_production_wiring ─────────────────────────────────────────────

template <>
inline void
sampling_service<default_sampling_policies>::shutdown_production_wiring(int64_t /*tid*/)
{
    using tls        = tl_state<default_sampling_policies>;
    tls::sampler     = nullptr;
    tls::offload     = nullptr;
    tls::logical_tid = -1;
}

// ── postfork production hooks ──────────────────────────────────────────────

template <>
inline void
sampling_service<default_sampling_policies>::postfork_production_parent_reinit()
{
    if(rocprofsys::config::get_use_process_sampling() &&
       rocprofsys::config::get_use_amd_smi())
    {
        LOG_DEBUG("[postfork_parent_reinit] delegating to pmc::postfork_parent_reinit");
        rocprofsys::pmc::postfork_parent_reinit();
    }
}

template <>
inline void
sampling_service<default_sampling_policies>::postfork_production_child_cleanup()
{
    if(rocprofsys::config::get_use_process_sampling() &&
       rocprofsys::config::get_use_amd_smi())
    {
        LOG_DEBUG("[postfork_child_cleanup] delegating to pmc::postfork_child_cleanup");
        rocprofsys::pmc::postfork_child_cleanup();
    }
}

namespace detail
{

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

// Build a single trace_cache::backtrace_region_sample for one frame.
inline trace_cache::backtrace_region_sample
make_region_sample(uint32_t category_id, std::size_t sys_id,
                   std::string const& track_name, std::string const& name,
                   std::uint64_t beg_ns, std::uint64_t end_ns, char const* category_str,
                   stack_frame const& frame, int depth)
{
    return trace_cache::backtrace_region_sample{ category_id,
                                                 static_cast<uint64_t>(sys_id),
                                                 track_name,
                                                 name,
                                                 beg_ns,
                                                 end_ns,
                                                 category_str,
                                                 make_call_stack_json(frame),
                                                 make_line_info_json(frame),
                                                 make_extdata_json(depth) };
}

// Emit timer + cpu-time region samples for one tid.
inline void
emit_timer_to_trace_cache(int64_t /*tid*/, std::optional<thread_info> const& info,
                          std::vector<timer_sample> const& timer_samples)
{
    if(timer_samples.empty()) return;

    constexpr auto category_id =
        static_cast<uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING);
    constexpr auto category_str     = "timer_sampling";
    constexpr auto cpu_category_str = "cputime_sampling";

    const std::size_t sys_id = info->index_data->system_value;
    const std::size_t seq_id = info->index_data->sequent_value;
    const std::string track_name =
        make_thread_track_name(timer_track_tag{}, seq_id, sys_id);

    for(auto const& sample : timer_samples)
    {
        if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

        const bool has_cpu = sample.metrics.valid.test(0) && sample.metrics.cpu_ns > 0;

        int depth = 0;
        for(auto const& frame : sample.stack)
        {
            std::string name = frame.name.empty()
                                   ? ("0x" + fmt::format("{:X}", frame.address))
                                   : frame.name;

            // Wall-clock timer sample.
            trace_cache::get_buffer_storage().store(
                make_region_sample(category_id, sys_id, track_name, name, sample.beg_ns,
                                   sample.end_ns, category_str, frame, depth));

            // CPU-time sample: emitted only when cpu_ns delta is available.
            // Duration encoded as [0, cpu_ns] so tsv_processor computes the
            // correct cpu-time duration from end_timestamp - start_timestamp.
            if(has_cpu)
            {
                trace_cache::get_buffer_storage().store(make_region_sample(
                    category_id, sys_id, track_name, name, std::uint64_t{ 0 },
                    static_cast<std::uint64_t>(sample.metrics.cpu_ns), cpu_category_str,
                    frame, depth));
            }

            ++depth;
        }
    }
}

// Emit per-thread process-sampling counter tracks for one tid.
inline void
emit_thread_counters(std::optional<thread_info> const& info,
                     std::vector<timer_sample> const&  timer_samples)
{
    const std::size_t sys_id = info->index_data->system_value;
    const std::size_t seq_id = info->index_data->sequent_value;

    for(auto const& sample : timer_samples)
    {
        if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;
        const std::uint64_t mid_ns = sample.end_ns;

        for(auto const& descriptor : thread_metric_descriptors)
        {
            if(!sample.metrics.valid.test(descriptor.valid_bit)) continue;
            std::string track = std::string{ descriptor.track_prefix } + " [" +
                                std::to_string(seq_id) + "]";
            trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
                descriptor.category_enum, std::move(track),
                static_cast<std::size_t>(mid_ns), std::string{ "{}" },
                /*stack_id*/ 0, /*parent_stack_id*/ 0,
                /*correlation_id*/ 0, /*call_stack*/ std::string{},
                /*line_info*/ std::string{}, static_cast<uint32_t>(sys_id),
                /*device_type*/ uint8_t{ 0 }, std::string{ descriptor.track_prefix },
                descriptor.read(sample.metrics), std::optional<int64_t>{} });
        }
    }
}

// Emit overflow region samples for one tid.
inline void
emit_overflow_to_trace_cache(int64_t /*tid*/, std::optional<thread_info> const& info,
                             std::vector<overflow_sample> const& overflow_samples)
{
    if(overflow_samples.empty()) return;

    constexpr auto category_id =
        static_cast<uint32_t>(ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING);
    constexpr auto category_str = "overflow_sampling";

    const std::size_t sys_id = info->index_data->system_value;
    const std::size_t seq_id = info->index_data->sequent_value;
    const std::string track_name =
        make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);

    for(auto const& sample : overflow_samples)
    {
        if(!info->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

        int depth = 0;
        for(auto const& frame : sample.stack)
        {
            std::string name = frame.name.empty()
                                   ? ("0x" + fmt::format("{:X}", frame.address))
                                   : frame.name;
            trace_cache::get_buffer_storage().store(
                make_region_sample(category_id, sys_id, track_name, name, sample.beg_ns,
                                   sample.end_ns, category_str, frame, depth));
            ++depth;
        }
    }
}

// tid may be either an internal_value (when called from tracing.cpp /
// library.cpp via utility::get_thread_index()) or a sequent_value (when
// called from pthread_create_gotcha which uses _info->sequent_value).
inline std::optional<thread_info> const&
resolve_tid_info(int64_t tid)
{
    auto const& info = thread_info::get(tid, SequentTID);
    if(info) return info;
    return thread_info::get(tid, InternalTID);
}

}  // namespace detail

// ── emit_resolved_to_trace_cache ───────────────────────────────────────────
// Variant 2 (Task #30): after draining the ring buffer in shutdown(tid), parse
// and resolve the raw backtrace_records and emit backtrace_region_sample directly
// to trace_cache::buffer_storage. Clears the tid from offload_ to prevent
// double-emission when post_process() iterates offload_.tids().

template <>
inline void
sampling_service<default_sampling_policies>::emit_resolved_to_trace_cache(int64_t tid)
{
    // L43 — refactor variant of legacy "[{}] Post-processing data for timemory..."
    LOG_DEBUG("[{}] Post-processing data for native report...", tid);

    auto records = offload_.read(tid);
    if(records.empty()) return;

    auto const& info = detail::resolve_tid_info(tid);
    if(!info)
    {
        offload_.erase(tid);
        return;
    }

    auto [timer_raw, overflow_raw] = detail::split_records(records);

    sample_parser   parser;
    symbol_resolver resolver;
    const bool      legacy = rocprofsys::get_use_sampling_trace_legacy();

    // Timer + cputime + thread-counter tracks (AC-11 discards single-sample buffer).
    if(timer_raw.size() >= 2)
    {
        backtrace_record              init_rec = timer_raw.front();
        std::vector<backtrace_record> tail(timer_raw.begin() + 1, timer_raw.end());
        auto timer_samples = parser.parse_timer(tid, init_rec, tail, pause_registry_);
        if(!timer_samples.empty())
        {
            detail::resolve_stacks(timer_samples, resolver);
            detail::emit_timer_to_trace_cache(tid, info, timer_samples);
            detail::emit_thread_counters(info, timer_samples);
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
            detail::emit_overflow_to_trace_cache(tid, info, overflow_samples);
            if(legacy) perfetto_sink_.emit_overflow(tid, nullptr, overflow_samples);
        }
    }

    offload_.erase(tid);
}

}  // namespace rocprofsys::sampling
