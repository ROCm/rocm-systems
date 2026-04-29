// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "fake_clock.hpp"
#include "in_memory_emitter.hpp"
#include "mock_overflow_trigger.hpp"
#include "mock_timer_trigger.hpp"
#include "mock_unwinder.hpp"
#include "noop_perfetto_sink.hpp"
#include "noop_report_writer.hpp"
#include "recording_signal_dispatcher.hpp"
#include "recording_trace_sink.hpp"
#include "throwing_fatal_error_policy.hpp"

#include "sampling/policies/production_hooks_policy.hpp"
#include "sampling/policies/test_hooks_policy.hpp"
#include "sampling/sampling_service.hpp"

namespace rocprofsys::sampling::test
{

// All-test-double policy bundle — satisfies the sampling_policies_traits concept.
// Type aliases use lower_case to match sampling_policies_traits<> member names.
struct test_sampling_policies
{
    using unwinder          = mock_unwinder;
    using offload           = in_memory_emitter;
    using trace_sink        = recording_trace_sink;
    using timer_trigger     = mock_timer_trigger;
    using overflow_trigger  = mock_overflow_trigger;
    using clock             = fake_clock;
    using signal_dispatcher = recording_signal_dispatcher;
    using report_writer     = noop_report_writer;
    using perfetto_sink     = noop_perfetto_sink;
    using fatal_error       = throwing_fatal_error_policy;
};

// sampling_service alias for unit tests: noop production hooks (no main-lib
// deps) + recording test hooks (so tests can flip override flags).
using test_service =
    sampling_service<test_sampling_policies, noop_production_hooks, recording_test_hooks>;

}  // namespace rocprofsys::sampling::test
