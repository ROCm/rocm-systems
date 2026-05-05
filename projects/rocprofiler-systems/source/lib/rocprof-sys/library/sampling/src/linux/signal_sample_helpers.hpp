// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Async-signal-safe helpers for the sampling signal handler.
// Each function populates fields of a backtrace_record in-place.

#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/limits.hpp"

#include <libunwind.h>
#include <sys/resource.h>

#include <cstdint>
#include <ctime>

namespace rocprofsys::sampling
{

// Modeled after baseline's timemory get_unw_stack<Depth, Offset, WSignalFrame=false>().
// Uses unw_getcontext() and skips handler frames via ignore_depth.
// Bottom-of-stack may include _start/__clone — these are valid frames that
// the baseline omitted only due to its deeper handler chain.
inline void
capture_stack_trace(backtrace_record& rec)
{
    unw_cursor_t  cursor = {};
    unw_context_t uctx   = {};
    unw_getcontext(&uctx);
    if(unw_init_local(&cursor, &uctx) != 0) return;

    constexpr int64_t ignore_depth = 3;
    int64_t           total_idx    = 0;

    while(true)
    {
        const int step_rc = unw_step(&cursor);
        if(step_rc == 0) break;
        if(step_rc < 0)
        {
            switch(-step_rc)
            {
                case UNW_ENOINFO:
                case UNW_EBADVERSION:
                case UNW_EINVALIDIP:
                case UNW_EBADFRAME: continue;
                default: goto done;
            }
        }

        if(total_idx < ignore_depth)
        {
            ++total_idx;
            continue;
        }

        if(unw_is_signal_frame(&cursor)) continue;

        if(total_idx >= static_cast<int64_t>(MAX_STACK_DEPTH) + ignore_depth) break;

        ++total_idx;

        unw_word_t ip = 0;
        if(unw_get_reg(&cursor, UNW_REG_IP, &ip) != 0) break;
        if(ip == 0) break;
        rec.raw_pcs[rec.pc_count++] = static_cast<uintptr_t>(ip);
    }
done:;
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
