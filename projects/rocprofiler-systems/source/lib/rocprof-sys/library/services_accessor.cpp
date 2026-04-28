// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Meyers-singleton definitions for services::sampling() and services::causal_sampling().
// DEC-10: both accessors return the same default_sampling_service instance.
// DEC-11: causal_sampling() is a thin alias — same object, causal path throws at setup().
//
// Also owns the thread-local signal-handler state pointers and the
// rocprofsys_sampling_signal_handler free function (ODR: exactly one definition).
//
// Lives in library/ so it can include library/sampling_production_policies.hpp which
// depends on main-library symbols (perf.hpp, tracing.hpp, trace_cache).
// Must NOT be compiled into standalone test binaries.

#if defined(__linux__)
#    include "library/sampling_production_policies.hpp"
#endif

#include "sampling/sampling_service.hpp"

namespace rocprofsys::services
{

#if defined(__linux__)

rocprofsys::sampling::default_sampling_service&
sampling()
{
    static rocprofsys::sampling::default_sampling_service instance;
    return instance;
}

rocprofsys::sampling::default_sampling_service&
causal_sampling()
{
    return sampling();
}

#endif

}  // namespace rocprofsys::services

// ── Thread-local signal-handler state (single definition) ──────────────────
// Declared extern in sampling_production_policies.hpp; defined here so exactly
// one TU owns them — avoids ODR violations when multiple TUs include the header.

#if defined(__linux__)

namespace rocprofsys::sampling
{

thread_local void*   tl_sampler_state_vp = nullptr;
thread_local void*   tl_offload_vp       = nullptr;
thread_local int64_t tl_logical_tid      = -1;

}  // namespace rocprofsys::sampling

// ── Signal handler definition (single TU) ──────────────────────────────────
// Installed via sigaction in real_timer_trigger::start().
// All state accessed through thread-local pointers — zero mutexes in handler (NFR-TS-2).

#    include <csignal>
#    include <ctime>
#    include <libunwind.h>
#    include <sys/resource.h>
#    include <ucontext.h>

extern "C" void
rocprofsys_sampling_signal_handler(int sig, siginfo_t* /*info*/, void* ucontext)
{
    using namespace rocprofsys::sampling;

    // Fast-path blocked check — no mutex.
    if(rocprofsys::services::sampling().is_blocked()) return;

    default_state_t* state = tl_sampler_state();
    if(!state || !state->is_running()) return;

    // Re-entry guard (DEC-15) — drop sample if already in handler.
    if(state->try_enter_handler())
    {
        state->increment_dropped();
        return;
    }

    state->enter_in_flight();

    // Build backtrace_record in-place — no heap allocation.
    backtrace_record rec{};
    rec.tid          = tl_logical_tid;
    rec.timestamp_ns = rocprofsys::services::sampling().get_clock().now_ns();
    rec.trigger      = (sig == rocprofsys::get_sampling_overflow_signal())
                           ? trigger_type::OVERFLOW
                           : trigger_type::TIMER;
    rec.pc_count     = 0;

    // Unwind directly into rec.raw_pcs[] using the signal ucontext.
    {
        unw_cursor_t  cursor = {};
        unw_context_t uctx   = {};
        if(ucontext)
        {
            uctx = *reinterpret_cast<unw_context_t const*>(
                static_cast<ucontext_t const*>(ucontext));
        }
        else
        {
            unw_getcontext(&uctx);
        }
        if(unw_init_local(&cursor, &uctx) == 0)
        {
            constexpr uint8_t max_depth = libunwind_unwinder::max_depth;
            while(rec.pc_count < max_depth)
            {
                unw_word_t ip = 0;
                if(unw_get_reg(&cursor, UNW_REG_IP, &ip) != 0) break;
                if(ip == 0) break;
                rec.raw_pcs[rec.pc_count++] = static_cast<uintptr_t>(ip);
                if(unw_step(&cursor) <= 0) break;
            }
        }
    }

    // Capture per-thread CPU time (async-signal-safe per POSIX.1-2017 §2.4.3).
    // Bit 0 of metrics.valid marks cpu_ns as populated.
    {
        struct timespec _cputime_ts = { 0, 0 };
        if(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &_cputime_ts) == 0)
        {
            rec.metrics.cpu_ns =
                (_cputime_ts.tv_sec * INT64_C(1'000'000'000)) + _cputime_ts.tv_nsec;
            rec.metrics.valid.set(0);
        }
    }

    // TF-3: capture per-thread rusage so emit_resolved_to_trace_cache can
    // produce the legacy thread_peak_memory / thread_context_switch /
    // thread_page_fault Perfetto counter tracks. RUSAGE_THREAD is a Linux
    // extension; getrusage is a thin syscall wrapper and behaves like an
    // async-signal-safe operation in glibc (matches the legacy
    // backtrace_metrics::sample call from develop).
    {
        struct rusage _ru = {};
        if(::getrusage(RUSAGE_THREAD, &_ru) == 0)
        {
            rec.metrics.mem_peak_kb = static_cast<int64_t>(_ru.ru_maxrss);
            rec.metrics.ctx_swch    = static_cast<int64_t>(_ru.ru_nvcsw + _ru.ru_nivcsw);
            rec.metrics.page_flt    = static_cast<int64_t>(_ru.ru_minflt + _ru.ru_majflt);
            rec.metrics.valid.set(1);  // mem_peak_kb
            rec.metrics.valid.set(2);  // ctx_swch
            rec.metrics.valid.set(3);  // page_flt
        }
    }

    if(!state->ring_buffer().try_push(rec))
    {
        state->increment_dropped();
    }

    state->exit_in_flight();
    state->exit_handler();
}

#endif  // __linux__
