// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// noop_production_hooks — test-only ProductionHooksPolicy implementation.
// Lives in tests/doubles/ because production code never instantiates it; the
// production wiring is real_production_hooks (library/sampling/policies/linux/).
// Unit-test bundles use this so they can instantiate sampling_service without
// dragging in main-library symbols (config.hpp, perf.hpp, trace_cache, ...).

#include <cstdint>
#include <set>

namespace rocprofsys::sampling::test
{

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

}  // namespace rocprofsys::sampling::test
