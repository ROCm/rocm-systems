// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// DEC-6: sampling_policies_traits struct grouping.
// sampling_service is templated on one Policies type rather than 10 separate parameters.

#include "sampling/policies/policy_checks.hpp"

namespace rocprofsys::sampling
{

template <class UnwinderT, class OffloadT, class TraceSinkT, class TimerTriggerT,
          class OverflowTriggerT, class ClockT, class SignalDispatcherT,
          class ReportWriterT, class PerfettoSinkT, class FatalErrorT>
struct sampling_policies_traits
{
    using unwinder          = UnwinderT;
    using offload           = OffloadT;
    using trace_sink        = TraceSinkT;
    using timer_trigger     = TimerTriggerT;
    using overflow_trigger  = OverflowTriggerT;
    using clock             = ClockT;
    using signal_dispatcher = SignalDispatcherT;
    using report_writer     = ReportWriterT;
    using perfetto_sink     = PerfettoSinkT;
    using fatal_error       = FatalErrorT;
};

// Production types are Linux-only. Forward-declared here; defined in src/linux/.
// Non-Linux gate is enforced at sampling_service<Policies> instantiation time
// via sampling/platform_guard.hpp (single source of truth — NFR-PORT-3).
#if defined(__linux__)
class libunwind_unwinder;
class trace_cache_offload_adapter;
class real_trace_cache_sink;
class real_timer_trigger;
class real_overflow_trigger;
class steady_clock;
class real_signal_dispatcher;
class native_report_writer;
class real_perfetto_sink;
class real_fatal_error_policy;

using default_sampling_policies = sampling_policies_traits<
    libunwind_unwinder, trace_cache_offload_adapter, real_trace_cache_sink,
    real_timer_trigger, real_overflow_trigger, steady_clock, real_signal_dispatcher,
    native_report_writer, real_perfetto_sink, real_fatal_error_policy>;
#endif

}  // namespace rocprofsys::sampling
