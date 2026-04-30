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
// The aggregator body is also wrapped in #if defined(__linux__) so this
// header is parseable (a no-op) on a hypothetical non-Linux preprocessor pass
// even though the build system never compiles sampling on non-Linux.

#if defined(__linux__)

// ── Light policies (sampling/ — no main-lib deps) ────────────────────────────
// libunwind is a hard requirement on Linux (sampling/CMakeLists.txt enforces
// find_package(LibUnwind REQUIRED)), so the unwinder include is unconditional.
#    include "sampling/src/linux/libunwind_unwinder.hpp"
#    include "sampling/src/linux/real_signal_dispatcher.hpp"
#    include "sampling/src/steady_clock.hpp"

// ── Production-only trigger policies (not test-accessible) ───────────────────
#    include "sampling/policies/linux/real_overflow_trigger.hpp"
#    include "sampling/policies/linux/real_timer_trigger.hpp"

// ── EmitterPolicy (production) ───────────────────────────────────────────────
// Lightweight header — no libunwind / AMD-SMI deps; also included by test TUs.
#    include "sampling/policies/trace_cache_offload_adapter.hpp"

// ── TSV report writer (no main-lib deps; test-accessible) ────────────────────
#    include "sampling/src/native_report_writer.hpp"

// ── Heavy policies (one class per file under sampling/policies/) ─────────────
#    include "sampling/policies/real_fatal_error_policy.hpp"
#    include "sampling/policies/real_perfetto_sink.hpp"
#    include "sampling/policies/real_thread_info_resolver.hpp"
#    include "sampling/policies/real_trace_cache_sink.hpp"

// ── TLS state + service template ────────────────────────────────────────────
// Lifecycle orchestration (formerly real_production_hooks) is now inlined
// into sampling_service<Policies>::do_setup_wiring / do_emit_resolved /
// do_postfork_* — see src/sampling_service_impl.hpp.
#    include "sampling/policies/tl_state.hpp"
#    include "sampling/sampling_service.hpp"

#    include <csignal>
#    include <sys/types.h>

namespace rocprofsys::sampling
{

using default_state_t = thread_sampler_state<default_sampling_policies>;
using default_tl      = tl_state<default_sampling_policies>;

}  // namespace rocprofsys::sampling

#    include "sampling/policies/linux/sampling_signal_handler_fwd.hpp"

// Template method bodies — bottom-included after the class definition and all
// policy types are visible. Lives in src/ so platform types (sigset_t, etc.)
// stay out of the public include/sampling/ headers (NFR-PORT-1).
#    include "sampling/src/sampling_service_impl.hpp"

#endif  // defined(__linux__)
