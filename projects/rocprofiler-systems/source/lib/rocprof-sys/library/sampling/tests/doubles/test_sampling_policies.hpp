// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "fake_clock.hpp"
#include "in_memory_emitter.hpp"
#include "mock_overflow_trigger.hpp"
#include "mock_thread_info_resolver.hpp"
#include "mock_timer_trigger.hpp"
#include "mock_unwinder.hpp"
#include "noop_perfetto_sink.hpp"
#include "noop_report_writer.hpp"
#include "recording_signal_dispatcher.hpp"
#include "recording_trace_sink.hpp"
#include "throwing_fatal_error_policy.hpp"

#include "sampling/sampling_config.hpp"
#include "sampling/sampling_service.hpp"

#include <csignal>

namespace rocprofsys::sampling::test
{

inline sampling_config
make_test_config()
{
    sampling_config cfg;
    cfg.realtime_signal = SIGPROF;
    cfg.cputime_signal  = SIGALRM;
    cfg.overflow_signal = SIGUSR1;
    cfg.realtime_freq   = 100.0;
    cfg.cputime_freq    = 100.0;
    cfg.resolve_signals = [](int64_t) { return std::set<int>{ SIGPROF, SIGALRM }; };
    return cfg;
}

// All-test-double policy bundle — satisfies the sampling_policies_traits concept.
// Type aliases use lower_case to match sampling_policies_traits<> member names.
// Lifecycle orchestration (setup_wiring, emit_resolved, postfork_*) lives
// directly on sampling_service<Policies>; tests change behaviour by swapping
// the 10 strategy policies, not by replacing the orchestration layer.
struct test_sampling_policies
{
    using unwinder             = mock_unwinder;
    using offload              = in_memory_emitter;
    using trace_sink           = recording_trace_sink;
    using timer_trigger        = mock_timer_trigger;
    using overflow_trigger     = mock_overflow_trigger;
    using clock                = fake_clock;
    using signal_dispatcher    = recording_signal_dispatcher;
    using report_writer        = noop_report_writer;
    using perfetto_sink        = noop_perfetto_sink;
    using fatal_error          = throwing_fatal_error_policy;
    using thread_info_resolver = mock_thread_info_resolver;
};

// sampling_service alias for unit tests.
using test_service = sampling_service<test_sampling_policies>;

}  // namespace rocprofsys::sampling::test

// Template method bodies — bottom-included so test TUs that instantiate
// sampling_service<test_sampling_policies> see the full method definitions.
#include "sampling/src/sampling_service_impl.hpp"
