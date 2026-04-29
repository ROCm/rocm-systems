// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ProductionHooksPolicy concept — injection point for production-only side
// effects (TLS wiring, timer arming, perf-event configuration, trace_cache
// emission, PMC postfork delegation).
//
// Generic template defaults to noop_production_hooks so sampling unit tests
// can instantiate sampling_service without dragging in main-library symbols
// (config.hpp, perf.hpp, trace_cache, thread_info, pmc/sampler).
//
// The production specialization (real_production_hooks) lives in
// sampling/policies/real_production_hooks.hpp and includes the heavy
// `library/` deps. Hooks reach back into sampling_service via public
// accessors (offload(), pause_registry(), perfetto_sink(), fatal_error(),
// duration_controller()) — no friend, no private access.

#include <cstdint>
#include <set>

namespace rocprofsys::sampling
{

// noop_production_hooks: generic default. All hooks are no-ops; the unit-test
// build does not need any `library/` includes.
struct noop_production_hooks
{
    static constexpr bool check_thread_guards(int64_t /*tid*/) noexcept { return true; }

    template <class Service>
    static void setup_wiring(Service& /*svc*/, int64_t /*tid*/,
                             typename Service::thread_state_t* /*state*/,
                             std::set<int> const& /*sigs*/) noexcept
    {}

    template <class Service>
    static void shutdown_wiring(Service& /*svc*/, int64_t /*tid*/) noexcept
    {}

    template <class Service>
    static void emit_resolved(Service& /*svc*/, int64_t /*tid*/) noexcept
    {}

    template <class Service>
    static void postfork_parent_reinit(Service& /*svc*/) noexcept
    {}

    template <class Service>
    static void postfork_child_cleanup(Service& /*svc*/) noexcept
    {}
};

}  // namespace rocprofsys::sampling
