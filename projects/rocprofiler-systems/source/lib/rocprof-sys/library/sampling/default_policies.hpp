// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Production-complete aggregator for the 10 sampling policy types plus the
// per-thread TLS pointers and the production hooks policy.
//
// Lives in library/ because the heavy policies depend on main-library symbols
// (config.hpp, tracing.hpp, trace_cache, thread_info, perf.hpp).
//
// Only include this header from translation units that are compiled as part of
// the main rocprofiler-systems library (NOT from standalone test binaries).
//
// Linux-only enforcement happens via sampling/platform_guard.hpp at
// sampling_service<Policies> instantiation time (NFR-PORT-3 single gate).

// ── Light policies (sampling/ — no main-lib deps) ────────────────────────────
#include "sampling/src/linux/libunwind_unwinder.hpp"
#include "sampling/src/linux/real_signal_dispatcher.hpp"
#include "sampling/src/steady_clock.hpp"

// ── Production-only trigger policies (not test-accessible) ───────────────────
#include "sampling/policies/real_overflow_trigger.hpp"
#include "sampling/policies/real_timer_trigger.hpp"

// ── EmitterPolicy (production) ───────────────────────────────────────────────
// Lightweight header — no libunwind / AMD-SMI deps; also included by test TUs.
#include "sampling/policies/trace_cache_offload_adapter.hpp"

// ── TSV report writer (no main-lib deps; test-accessible) ────────────────────
#include "sampling/src/native_report_writer.hpp"

// ── Heavy policies (one class per file under sampling/policies/) ─────────────
#include "sampling/policies/real_fatal_error_policy.hpp"
#include "sampling/policies/real_perfetto_sink.hpp"
#include "sampling/policies/real_trace_cache_sink.hpp"

// ── TLS state + service template + production hooks policy ──────────────────
#include "sampling/policies/tl_state.hpp"
#include "sampling/sampling_service.hpp"

// real_production_hooks (T18a) replaces the explicit-specialization layer
// that formerly lived in sampling_service_production_hooks.hpp.
#include "sampling/policies/real_production_hooks.hpp"

#include <csignal>
#include <sys/types.h>

namespace rocprofsys::sampling
{

using default_state_t = thread_sampler_state<default_sampling_policies>;
using default_tl      = tl_state<default_sampling_policies>;

}  // namespace rocprofsys::sampling

// Forward declaration — definition in services_accessor.cpp.
extern "C" void
rocprofsys_sampling_signal_handler(int, siginfo_t*, void*);
