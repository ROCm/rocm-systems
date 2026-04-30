// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Template implementation of the sampling signal handler body.
// Extracted from services_accessor.cpp so tests can instantiate with
// test policies and exercise the full unwind->ring-buffer flow without
// real signals.
//
// The extern "C" trampoline in services_accessor.cpp is a one-liner
// that delegates to this template.

#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/limits.hpp"
#include "sampling/policies/tl_state.hpp"
#include "sampling/sampling_service.hpp"

#include <libunwind.h>
#include <sys/resource.h>
#include <ucontext.h>

#include <cstdint>
#include <ctime>

namespace rocprofsys::sampling
{

template <class Policies>
void
sampling_signal_handler_body(int sig, void* ucontext, sampling_service<Policies>& svc)
{
    if(svc.is_blocked()) return;

    using state_t = thread_sampler_state<Policies>;
    using tls     = tl_state<Policies>;

    state_t* state = tls::sampler;
    if(!state || !state->is_running()) return;

    if(state->try_enter_handler())
    {
        state->increment_dropped();
        return;
    }

    state->enter_in_flight();

    backtrace_record rec{};
    rec.tid          = tls::logical_tid;
    rec.timestamp_ns = svc.get_clock().now_ns();
    rec.trigger      = (sig == svc.config().overflow_signal) ? trigger_type::OVERFLOW
                                                             : trigger_type::TIMER;
    rec.pc_count     = 0;

    {
        static_assert(sizeof(unw_context_t) == sizeof(ucontext_t),
                      "unw_context_t / ucontext_t size mismatch — ucontext "
                      "reinterpret_cast in the sampling signal handler is unsafe");

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

    {
        struct timespec cputime_ts = { 0, 0 };
        if(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cputime_ts) == 0)
        {
            rec.metrics.cpu_ns =
                (cputime_ts.tv_sec * INT64_C(1'000'000'000)) + cputime_ts.tv_nsec;
            rec.metrics.valid.set(0);
        }
    }

    {
        struct rusage ru = {};
        if(::getrusage(RUSAGE_THREAD, &ru) == 0)
        {
            rec.metrics.mem_peak_kb = static_cast<int64_t>(ru.ru_maxrss);
            rec.metrics.ctx_swch    = static_cast<int64_t>(ru.ru_nvcsw + ru.ru_nivcsw);
            rec.metrics.page_flt    = static_cast<int64_t>(ru.ru_minflt + ru.ru_majflt);
            rec.metrics.valid.set(1);
            rec.metrics.valid.set(2);
            rec.metrics.valid.set(3);
        }
    }

    if(!state->ring_buffer().try_push(rec))
    {
        state->increment_dropped();
    }

    state->exit_in_flight();
    state->exit_handler();
}

}  // namespace rocprofsys::sampling
