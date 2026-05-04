// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Async-signal-safe helpers for the sampling signal handler.
// Each function populates fields of a backtrace_record in-place.

#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/limits.hpp"

#include <libunwind.h>
#include <sys/resource.h>
#include <ucontext.h>

#include <cstdint>
#include <ctime>

namespace rocprofsys::sampling
{

inline void
capture_stack_trace(backtrace_record& rec, void* ucontext)
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
        constexpr int skip_frames = 3;
        for(int skip = 0; skip < skip_frames; ++skip)
        {
            if(unw_step(&cursor) <= 0) return;
        }

        while(rec.pc_count < static_cast<uint8_t>(MAX_STACK_DEPTH))
        {
            if(unw_is_signal_frame(&cursor))
            {
                if(unw_step(&cursor) <= 0) break;
                continue;
            }

            unw_word_t ip = 0;
            if(unw_get_reg(&cursor, UNW_REG_IP, &ip) != 0) break;
            if(ip == 0) break;
            rec.raw_pcs[rec.pc_count++] = static_cast<uintptr_t>(ip);

            const int step_rc = unw_step(&cursor);
            if(step_rc == 0) break;
            if(step_rc < 0)
            {
                if(-step_rc == UNW_ENOINFO || -step_rc == UNW_EBADVERSION ||
                   -step_rc == UNW_EINVALIDIP || -step_rc == UNW_EBADFRAME)
                    continue;
                break;
            }
        }
    }
}

inline void
capture_cpu_time(backtrace_record& rec)
{
    struct timespec cputime_ts = { 0, 0 };
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cputime_ts) == 0)
    {
        rec.metrics.cpu_ns =
            (cputime_ts.tv_sec * INT64_C(1'000'000'000)) + cputime_ts.tv_nsec;
        rec.metrics.valid.set(0);
    }
}

inline void
capture_thread_rusage(backtrace_record& rec)
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

}  // namespace rocprofsys::sampling
