// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Meyers-singleton definitions for services::sampling() and services::causal_sampling().
// DEC-10: both accessors return the same default_sampling_service instance.
// DEC-11: causal_sampling() is a thin alias — same object, causal path throws at setup().
//
// Also owns the thread-local signal-handler state pointers and the
// rocprofsys_sampling_signal_handler free function (ODR: exactly one definition).
//
// Lives in library/ so it can include sampling/default_policies.hpp which
// depends on main-library symbols (perf.hpp, tracing.hpp, trace_cache).
// Must NOT be compiled into standalone test binaries.

#if defined(__linux__)
#    include "sampling/default_policies.hpp"
#endif

#include "library/sampling_service_instantiation.hpp"
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

// ── Thin caller-facing wrappers ──────────────────────────────────────────────
// Defined here (the single TU with the full sampling_service template
// instantiation) so caller TUs can stay free of main-lib hook deps.

std::set<int>
sampling_setup(int64_t tid)
{
    return sampling().setup(tid);
}

std::set<int>
sampling_shutdown(int64_t tid)
{
    return sampling().shutdown(tid);
}

void
sampling_block_samples()
{
    sampling().block_samples();
}

void
sampling_unblock_samples()
{
    sampling().unblock_samples();
}

void
sampling_block_signals(std::set<int> sigs)
{
    sampling().block_signals(std::move(sigs));
}

void
sampling_unblock_signals(std::set<int> sigs)
{
    sampling().unblock_signals(std::move(sigs));
}

void
sampling_pause()
{
    sampling().pause();
}

void
sampling_resume()
{
    sampling().resume();
}

void
sampling_postfork_parent_reinit()
{
    sampling().postfork_parent_reinit();
}

void
sampling_postfork_child_cleanup()
{
    sampling().postfork_child_cleanup();
}

void
sampling_enter_child_process_mode()
{
    sampling().enter_child_process_mode();
}

std::set<int>
causal_sampling_setup(int64_t tid)
{
    return causal_sampling().setup(tid);
}

std::set<int>
causal_sampling_shutdown(int64_t tid)
{
    return causal_sampling().shutdown(tid);
}

void
causal_sampling_block_signals(std::set<int> sigs)
{
    causal_sampling().block_signals(std::move(sigs));
}

void
causal_sampling_unblock_signals(std::set<int> sigs)
{
    causal_sampling().unblock_signals(std::move(sigs));
}

void
causal_sampling_pause()
{
    causal_sampling().pause();
}

void
causal_sampling_resume()
{
    causal_sampling().resume();
}

void
sampling_shutdown_in_child_mode(int64_t tid)
{
    auto& svc = sampling();
    svc.enter_child_process_mode();
    svc.shutdown(tid);
}

#endif

}  // namespace rocprofsys::services

// Typed thread-local signal-handler state lives in sampling/policies/tl_state.hpp
// (definitions are inline thread_local template members).

#if defined(__linux__)

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

    default_state_t* state = default_tl::sampler;
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
    rec.tid          = default_tl::logical_tid;
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
            while(rec.pc_count < static_cast<uint8_t>(MAX_STACK_DEPTH))
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
