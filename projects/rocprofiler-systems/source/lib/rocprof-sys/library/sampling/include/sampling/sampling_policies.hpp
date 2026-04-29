// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// DEC-6: sampling_policies_traits struct grouping.
// sampling_service is templated on a single Policies type. The bundle now also
// carries the production-hooks and test-hooks types as nested aliases so the
// service template needs only one parameter (P4 — single source of truth for
// dependent types).

#include "sampling/policies/policy_checks.hpp"
#include "sampling/policies/production_hooks_policy.hpp"
#include "sampling/policies/test_hooks_policy.hpp"

namespace rocprofsys::sampling
{

template <class UnwinderT, class OffloadT, class TraceSinkT, class TimerTriggerT,
          class OverflowTriggerT, class ClockT, class SignalDispatcherT,
          class ReportWriterT, class PerfettoSinkT, class FatalErrorT,
          class ProductionHooksT = noop_production_hooks,
          class TestHooksT       = noop_test_hooks>
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
    using production_hooks  = ProductionHooksT;
    using test_hooks        = TestHooksT;
};

// Production types are Linux-only. Forward-declared here; defined in src/linux/.
// Non-Linux gate is enforced at sampling_service<Policies> instantiation time
// via sampling/platform_guard.hpp (single source of truth — NFR-PORT-3).
// libunwind is a hard requirement on Linux — enforced at CMake configure time
// in sampling/CMakeLists.txt (find_package(LibUnwind REQUIRED)).
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

// Forward-declared production hooks type — definition lives in the linux
// policies tree (real_production_hooks.hpp) and is included from
// default_policies.hpp before this template alias is instantiated.
class real_production_hooks;

using default_sampling_policies = sampling_policies_traits<
    libunwind_unwinder, trace_cache_offload_adapter, real_trace_cache_sink,
    real_timer_trigger, real_overflow_trigger, steady_clock, real_signal_dispatcher,
    native_report_writer, real_perfetto_sink, real_fatal_error_policy,
    real_production_hooks, noop_test_hooks>;
#endif

}  // namespace rocprofsys::sampling
