// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// DEC-6: sampling_policies_traits struct grouping.
// sampling_service is templated on one Policies type rather than 10 separate parameters.

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
#if defined(__linux__)
class libunwind_unwinder;
class tmpfile_offload_store;
class real_trace_cache_sink;
class real_timer_trigger;
class real_overflow_trigger;
class steady_clock;
class real_signal_dispatcher;
class native_report_writer;
class real_perfetto_sink;
class real_fatal_error_policy;

using default_sampling_policies = sampling_policies_traits<
    libunwind_unwinder, tmpfile_offload_store, real_trace_cache_sink, real_timer_trigger,
    real_overflow_trigger, steady_clock, real_signal_dispatcher, native_report_writer,
    real_perfetto_sink, real_fatal_error_policy>;
#else
// Non-Linux: any TU that reaches this point gets a hard compile error (C-3 fix).
// This fires on #include of sampling_policies.hpp on non-Linux even without
// instantiation — which is the correct gate for default_sampling_policies.
static_assert(false,
              "Thread sampling is Linux-only in this build. "
              "Do not include sampling/sampling_policies.hpp on non-Linux targets.");
#endif

}  // namespace rocprofsys::sampling
