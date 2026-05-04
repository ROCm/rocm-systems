// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Template implementation of the sampling signal handler body.
// The extern "C" trampoline in services_accessor.cpp delegates to this.

#include "sampling/data/backtrace_record.hpp"
#include "sampling/policies/tl_state.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/linux/signal_sample_helpers.hpp"

#include "core/state.hpp"

#include <cerrno>

namespace rocprofsys::sampling
{

template <class Policies>
void
sampling_signal_handler_body(int sig, void* ucontext, sampling_service<Policies>& svc)
{
    if(svc.is_blocked()) return;

    const int saved_errno = errno;

    using state_t = thread_sampler_state<Policies>;
    using tls     = tl_state<Policies>;

    state_t* state = tls::sampler;
    if(!state || !state->is_running())
    {
        errno = saved_errno;
        return;
    }

    if(state->try_enter_handler())
    {
        state->increment_dropped();
        errno = saved_errno;
        return;
    }

    state->enter_in_flight();

    backtrace_record rec{};
    rec.tid          = tls::logical_tid;
    rec.timestamp_ns = svc.get_clock().now_ns();
    rec.trigger      = (sig == svc.config().overflow_signal) ? trigger_type::OVERFLOW
                                                             : trigger_type::TIMER;
    rec.pc_count     = 0;

    auto prev_state = set_thread_state(ThreadState::Internal);
    capture_stack_trace(rec, ucontext);
    set_thread_state(prev_state);

    capture_cpu_time(rec);
    capture_thread_rusage(rec);

    auto read_hw = svc.callbacks().read_hw_counters;
    if(read_hw)
    {
        auto n = read_hw(rec.tid, rec.metrics.hw_counter.data(),
                         rec.metrics.hw_counter.size());
        if(n > 0) rec.metrics.valid.set(4);
    }

    if(!state->ring_buffer().try_push(rec))
    {
        state->increment_dropped();
    }

    state->exit_in_flight();
    state->exit_handler();
    errno = saved_errno;
}

}  // namespace rocprofsys::sampling
