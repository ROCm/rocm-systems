// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Policy bundles used for post_process() TDD tests.
// Each bundle swaps in recording doubles for the sinks/writer while keeping
// the rest of the standard test doubles.

#include "fake_clock.hpp"
#include "in_memory_offload.hpp"
#include "mock_overflow_trigger.hpp"
#include "mock_timer_trigger.hpp"
#include "mock_unwinder.hpp"
#include "recording_perfetto_sink.hpp"
#include "recording_report_writer.hpp"
#include "recording_signal_dispatcher.hpp"
#include "recording_trace_sink.hpp"
#include "throwing_fatal_error_policy.hpp"

namespace rocprofsys::sampling::test
{

// Standard post_process test bundle — recording sinks, in-memory offload.
struct post_process_policies
{
    using unwinder          = mock_unwinder;
    using offload           = in_memory_offload;
    using trace_sink        = recording_trace_sink;
    using timer_trigger     = mock_timer_trigger;
    using overflow_trigger  = mock_overflow_trigger;
    using clock             = fake_clock;
    using signal_dispatcher = recording_signal_dispatcher;
    using report_writer     = recording_report_writer;
    using perfetto_sink     = recording_perfetto_sink;
    using fatal_error       = throwing_fatal_error_policy;
};

// Exception-safety variant — throws from report_writer.write_timer_samples().
struct throwing_writer_policies
{
    using unwinder          = mock_unwinder;
    using offload           = in_memory_offload;
    using trace_sink        = recording_trace_sink;
    using timer_trigger     = mock_timer_trigger;
    using overflow_trigger  = mock_overflow_trigger;
    using clock             = fake_clock;
    using signal_dispatcher = recording_signal_dispatcher;
    using report_writer     = throwing_report_writer;
    using perfetto_sink     = recording_perfetto_sink;
    using fatal_error       = throwing_fatal_error_policy;
};

// Exception-safety variant — throws from report_writer.flush() (C-15).
struct throwing_flush_policies
{
    using unwinder          = mock_unwinder;
    using offload           = in_memory_offload;
    using trace_sink        = recording_trace_sink;
    using timer_trigger     = mock_timer_trigger;
    using overflow_trigger  = mock_overflow_trigger;
    using clock             = fake_clock;
    using signal_dispatcher = recording_signal_dispatcher;
    using report_writer     = throwing_flush_report_writer;
    using perfetto_sink     = recording_perfetto_sink;
    using fatal_error       = throwing_fatal_error_policy;
};

}  // namespace rocprofsys::sampling::test
