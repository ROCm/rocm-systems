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

#include <nlohmann/json.hpp>

#include <linux/perf_event.h>
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <mutex>

namespace rocprofsys::sampling
{

// Thread-local signal-handler state — defined in services_accessor.cpp.
// Declared here at namespace scope so the static TLS model is used
// (avoids dynamic-TLS fault in dlopen-loaded libraries).
extern thread_local void*   tl_sampler_state_vp;
extern thread_local void*   tl_offload_vp;
extern thread_local int64_t tl_logical_tid;

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
    tl_sampler_state_vp = static_cast<void*>(state);
    tl_offload_vp       = static_cast<void*>(&offload_);
    tl_logical_tid      = tid;

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
    // records. Must happen regardless of use_perfetto — real_trace_cache_sink::store()
    // always pushes records using these exact track names (Bug 4 root cause fix).
    // Track name format must exactly match real_trace_cache_sink::store().
    const auto& thread_inf = thread_info::get(tid, SequentTID);
    if(thread_inf)
    {
        size_t sys_id = thread_inf->index_data->system_value;
        size_t seq_id = thread_inf->index_data->sequent_value;

        std::string timer_track =
            "Thread " + std::to_string(seq_id) + " (S) " + std::to_string(sys_id);
        std::string overflow_track = "Thread " + std::to_string(seq_id) +
                                     " Overflow (S) " + std::to_string(sys_id);

        auto& reg = trace_cache::get_metadata_registry();

        // AC-16: register sampling category strings once per process.
        static std::once_flag s_category_init;
        std::call_once(s_category_init, [&reg]() {
            reg.add_string("timer_sampling");
            reg.add_string("overflow_sampling");
        });

        // AC-16: register thread info so rocpd has ppid/pid/tid for this thread.
        reg.add_thread_info({ getppid(), getpid(), sys_id, 0, 0, "{}" });

        reg.add_track({ timer_track, sys_id, "{}" });
        reg.add_track({ overflow_track, sys_id, "{}" });

        LOG_DEBUG("thread {} registered trace_cache tracks: '{}' / '{}'", tid,
                  timer_track, overflow_track);
    }
}

// ── shutdown_production_wiring ─────────────────────────────────────────────

template <>
inline void
sampling_service<default_sampling_policies>::shutdown_production_wiring(int64_t /*tid*/)
{
    tl_sampler_state_vp = nullptr;
    tl_offload_vp       = nullptr;
    tl_logical_tid      = -1;
}

// ── open_report_writer_streams ─────────────────────────────────────────────
// Opens sampling_wall_clock.tsv, sampling_cpu_clock.tsv, sampling_percent.tsv,
// and trip_count.tsv in the configured output directory, then injects them into
// report_writer_ so flush() has real file handles.

template <>
inline void
sampling_service<default_sampling_policies>::open_report_writer_streams()
{
    // Ofstreams are kept alive as members so report_writer_ can hold non-owning
    // pointers until flush() completes.
    auto open = [](std::ofstream& ofs, std::string_view name) {
        auto path = rocprofsys::get_sampling_output_filepath(name);
        if(!path.empty())
        {
            auto parent = std::filesystem::path{ path }.parent_path();
            if(!parent.empty()) std::filesystem::create_directories(parent);
        }
        ofs.open(path);
        if(!ofs.is_open()) LOG_CRITICAL("[native_report_writer] cannot open {}", path);
    };

    if(!m_tsv_wall.is_open()) open(m_tsv_wall, "sampling_wall_clock");
    if(!m_tsv_cpu.is_open()) open(m_tsv_cpu, "sampling_cpu_clock");
    if(!m_tsv_pct.is_open()) open(m_tsv_pct, "sampling_percent");
    if(!m_tsv_trip.is_open()) open(m_tsv_trip, "trip_count");

    report_writer_.set_streams(&m_tsv_wall, &m_tsv_cpu, &m_tsv_pct, &m_tsv_trip);
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

// ── emit_resolved_to_trace_cache ───────────────────────────────────────────
// Variant 2 (Task #30): after draining the ring buffer in shutdown(tid), parse
// and resolve the raw backtrace_records and emit backtrace_region_sample directly
// to trace_cache::buffer_storage. Clears the tid from offload_ to prevent
// double-emission when post_process() iterates offload_.tids().

template <>
inline void
sampling_service<default_sampling_policies>::emit_resolved_to_trace_cache(int64_t tid)
{
    auto records = offload_.read(tid);
    if(records.empty()) return;

    const auto& info = thread_info::get(tid, SequentTID);
    if(!info)
    {
        offload_.erase(tid);
        return;
    }

    size_t sys_id = info->index_data->system_value;
    size_t seq_id = info->index_data->sequent_value;

    sample_parser   parser;
    symbol_resolver resolver;

    // Split TIMER vs OVERFLOW records.
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

    // Parse + resolve + emit timer samples.
    if(timer_raw.size() >= 2)
    {
        backtrace_record              init_rec = timer_raw.front();
        std::vector<backtrace_record> tail(timer_raw.begin() + 1, timer_raw.end());
        auto timer_samples = parser.parse_timer(tid, init_rec, tail, pause_registry_);

        if(!timer_samples.empty())
        {
            for(auto& s : timer_samples)
                for(auto& f : s.stack)
                    if(f.name.empty()) f.name = resolver.resolve(f.address);

            constexpr uint32_t category_id =
                static_cast<uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING);
            constexpr auto category_str = "timer_sampling";
            std::string    track_name =
                "Thread " + std::to_string(seq_id) + " (S) " + std::to_string(sys_id);

            for(auto const& s : timer_samples)
            {
                if(!info->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

                int depth = 0;
                for(auto const& frame : s.stack)
                {
                    std::string    name = frame.name.empty()
                                              ? ("0x" + fmt::format("{:X}", frame.address))
                                              : frame.name;
                    nlohmann::json cs_j;
                    cs_j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address)
                                                      : frame.name;
                    cs_j["pc"]   = fmt::format("{:X}", frame.address);
                    cs_j["file"] = frame.location;
                    nlohmann::json li_j;
                    li_j["line_address"] = fmt::format("{:X}", frame.line_address);
                    li_j["name"]         = cs_j["name"];
                    if(!frame.inlines.empty())
                    {
                        nlohmann::json inlined;
                        auto const&    top  = frame.inlines.front();
                        inlined["name"]     = top.name;
                        inlined["location"] = top.location;
                        inlined["line"]     = std::to_string(top.line);
                        li_j["inlined"]     = inlined;
                    }
                    nlohmann::json ext;
                    ext["depth"] = depth;

                    trace_cache::get_buffer_storage().store(
                        trace_cache::backtrace_region_sample{
                            category_id, static_cast<uint64_t>(sys_id), track_name, name,
                            s.beg_ns, s.end_ns, category_str, cs_j.dump(), li_j.dump(),
                            ext.dump() });
                    ++depth;
                }
            }
        }
    }

    // Parse + resolve + emit overflow samples.
    if(!overflow_raw.empty())
    {
        auto overflow_samples = parser.parse_overflow(tid, overflow_raw, pause_registry_);

        if(!overflow_samples.empty())
        {
            for(auto& s : overflow_samples)
                for(auto& f : s.stack)
                    if(f.name.empty()) f.name = resolver.resolve(f.address);

            constexpr uint32_t category_id =
                static_cast<uint32_t>(ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING);
            constexpr auto category_str = "overflow_sampling";
            std::string    track_name   = "Thread " + std::to_string(seq_id) +
                                     " Overflow (S) " + std::to_string(sys_id);

            for(auto const& s : overflow_samples)
            {
                if(!info->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

                int depth = 0;
                for(auto const& frame : s.stack)
                {
                    std::string    name = frame.name.empty()
                                              ? ("0x" + fmt::format("{:X}", frame.address))
                                              : frame.name;
                    nlohmann::json cs_j;
                    cs_j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address)
                                                      : frame.name;
                    cs_j["pc"]   = fmt::format("{:X}", frame.address);
                    cs_j["file"] = frame.location;
                    nlohmann::json li_j;
                    li_j["line_address"] = fmt::format("{:X}", frame.line_address);
                    li_j["name"]         = cs_j["name"];
                    if(!frame.inlines.empty())
                    {
                        nlohmann::json inlined;
                        auto const&    top  = frame.inlines.front();
                        inlined["name"]     = top.name;
                        inlined["location"] = top.location;
                        inlined["line"]     = std::to_string(top.line);
                        li_j["inlined"]     = inlined;
                    }
                    nlohmann::json ext;
                    ext["depth"] = depth;

                    trace_cache::get_buffer_storage().store(
                        trace_cache::backtrace_region_sample{
                            category_id, static_cast<uint64_t>(sys_id), track_name, name,
                            s.beg_ns, s.end_ns, category_str, cs_j.dump(), li_j.dump(),
                            ext.dump() });
                    ++depth;
                }
            }
        }
    }

    // Clear processed tid from offload_ to prevent double-emission in post_process().
    offload_.erase(tid);
}

}  // namespace rocprofsys::sampling
