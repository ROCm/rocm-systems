// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Thin caller-facing API for the production sampling_service<default_sampling_policies,
// real_production_hooks, noop_test_hooks> singleton. The full template instantiation
// lives in services_accessor.cpp (the only TU that includes
// sampling/default_policies.hpp); caller TUs include this header instead and route
// method calls through the free-function wrappers below.
//
// Rationale: real_production_hooks has main-lib deps (config.hpp, perf.hpp,
// thread_info.hpp, pmc/sampler.hpp). Pulling that header chain through 7 caller TUs
// is wasteful. The wrappers reduce each caller's include footprint to <set> +
// <cstdint>.
//
// Linux-only: all wrappers are no-ops on non-Linux (the singleton itself does not
// exist; the platform_guard at sampling_service<Policies> instantiation time is
// the single non-Linux gate).

#include <cstdint>
#include <set>
#include <utility>

#if defined(__linux__)

namespace rocprofsys::services
{

// Driver API — mirrors sampling_service public methods on services::sampling().
[[nodiscard]] std::set<int>
sampling_setup(int64_t tid);
[[nodiscard]] std::set<int>
sampling_shutdown(int64_t tid);

void
sampling_block_samples();
void
sampling_unblock_samples();
void
sampling_block_signals(std::set<int> sigs = {});
void
sampling_unblock_signals(std::set<int> sigs = {});

void
sampling_pause();
void
sampling_resume();

void
sampling_postfork_parent_reinit();
void
sampling_postfork_child_cleanup();

void
sampling_enter_child_process_mode();

// Causal variants — services::causal_sampling() returns the same singleton
// (DEC-10); the causal-mode guard at setup() differentiates behavior. Kept as
// thin aliases so callers in causal-only paths (causal/delay.cpp,
// causal/sampling.cpp) read meaningfully.
[[nodiscard]] std::set<int>
causal_sampling_setup(int64_t tid);
[[nodiscard]] std::set<int>
causal_sampling_shutdown(int64_t tid);
void
causal_sampling_block_signals(std::set<int> sigs = {});
void
causal_sampling_unblock_signals(std::set<int> sigs = {});
void
causal_sampling_pause();
void
causal_sampling_resume();

// Diamond-collapsing accessors: callers that previously branched on
// causal-vs-sampling mode (library.cpp, tracing.cpp, pthread_create_gotcha.cpp)
// route through these. The underlying singleton is the same — the wrapper
// exists so the diamond stays out of caller code.
inline std::set<int>
sampling_setup_for_thread(int64_t tid)
{
    return sampling_setup(tid);
}
inline std::set<int>
sampling_shutdown_for_thread(int64_t tid)
{
    return sampling_shutdown(tid);
}
inline void
sampling_block_signals_for_thread(std::set<int> sigs = {})
{
    sampling_block_signals(std::move(sigs));
}
inline void
sampling_unblock_signals_for_thread(std::set<int> sigs = {})
{
    sampling_unblock_signals(std::move(sigs));
}

bool
sampling_is_setup_for_current_thread();

// Encapsulates the fork-child 2-step:
//   services::sampling().enter_child_process_mode();
//   services::sampling().shutdown(tid);
// Used by fork_gotcha postfork_child to release per-thread state without
// per-tid processing (AC-20).
void
sampling_shutdown_in_child_mode(int64_t tid);

}  // namespace rocprofsys::services

#endif  // __linux__
