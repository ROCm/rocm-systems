// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Smoke tests for sampling_service<default_sampling_policies, real_production_hooks>
// production hooks.
//
// Tests that the real_production_hooks policy is wired correctly via the
// ProductionHooksPolicy template slot. The observable behavior tested here is:
//   - shutdown_wiring() clears the three thread-local state pointers
//     (tl_state<default_sampling_policies>::sampler/offload/logical_tid) so no
//     stale pointers remain after a thread's sampling session ends
//     (AC-11, NFR-TS-2).
//
// Requires rocprofiler-systems-core-library (real config, thread_info, trace_cache)
// so this is a separate binary that does NOT include config_stubs.cpp.
//
// Config initialization: configure_settings(false) via GTest Environment to
// avoid the recursive_init_error in the magic-static get_config() path.

#include <gtest/gtest.h>

#include "core/config.hpp"
// sampling/default_policies.hpp provides ALL 10 policy types as complete definitions
// plus the real_production_hooks policy class. Required because the production
// sampling_service<default_sampling_policies, real_production_hooks> instantiation
// needs every policy type.
#include "sampling/default_policies.hpp"
#include "sampling/sampling_service.hpp"

#include <cstdint>

// ── Config environment ────────────────────────────────────────────────────────
namespace
{
class HooksConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override { rocprofsys::configure_settings(false); }
};
}  // namespace

// ── Compile-time specialization checks ───────────────────────────────────────

namespace
{
using prod_service = rocprofsys::sampling::default_sampling_service;
}  // namespace

// ── shutdown_wiring: TLS pointers cleared ───────────────────────────────────
// This is the only production hook whose outcome is directly observable without
// OS resources (perf events, Perfetto runtime, thread_info registry).

TEST(sampling_service_production_hooks, shutdown_clears_tls_sampler_state)
{
    using namespace rocprofsys::sampling;
    using tls = tl_state<default_sampling_policies>;

    // Simulate TLS state that was set by setup_wiring.
    using state_t = thread_sampler_state<default_sampling_policies>;
    state_t dummy_state{};
    tls::sampler     = &dummy_state;
    tls::offload     = nullptr;
    tls::logical_tid = 99;

    // Call shutdown() on a default-constructed service.
    // shutdown() calls: offload_.write(tid) → production_hooks_.emit_resolved(tid)
    //                   → production_hooks_.shutdown_wiring(tid)
    // Ring buffer is empty, so write() is a no-op; the hook still runs.
    prod_service svc;
    svc.shutdown(0);

    EXPECT_EQ(tls::sampler, nullptr)
        << "shutdown_production_wiring must set tl_state::sampler to nullptr";
    EXPECT_EQ(tls::offload, nullptr)
        << "shutdown_production_wiring must set tl_state::offload to nullptr";
    EXPECT_EQ(tls::logical_tid, -1)
        << "shutdown_production_wiring must set tl_state::logical_tid to -1";
}

// ── Custom main ───────────────────────────────────────────────────────────────
int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new HooksConfigEnv);
    return RUN_ALL_TESTS();
}
