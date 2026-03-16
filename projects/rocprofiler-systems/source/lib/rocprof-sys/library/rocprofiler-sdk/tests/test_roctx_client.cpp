// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprof-sys/library/rocprofiler-sdk/roctx_client.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/trace_control.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

// ============================================================================
// roctx_client construction tests
// ============================================================================

class roctx_client_test : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(roctx_client_test, constructor_creates_controller)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config config{ true, true, true, false, "TestRegion" };
    roctx_client<>            client(config);
    EXPECT_NE(client.get_controller(), nullptr);
}

TEST_F(roctx_client_test, constructor_without_region_filter)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config config{ true, true, true, false, "" };
    roctx_client<>            client(config);
    EXPECT_NE(client.get_controller(), nullptr);
    EXPECT_FALSE(client.get_controller()->region_filter_active());
}

TEST_F(roctx_client_test, constructor_with_region_filter)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config config{ true, true, true, false, "Region 1" };
    roctx_client<>            client(config);
    EXPECT_TRUE(client.get_controller()->region_filter_active());
}

TEST_F(roctx_client_test, should_write_requires_write_enabled)
{
    using namespace rocprofsys::rocprofiler_sdk;

    // is_write_enabled = false, no region filter
    const roctx_client_config config{ false, true, true, false, "" };
    const roctx_client<>      client(config);
    EXPECT_FALSE(client.should_write_markers());
}

TEST_F(roctx_client_test, should_write_no_filter_and_enabled)
{
    using namespace rocprofsys::rocprofiler_sdk;

    // is_write_enabled = true, no region filter => always write
    const roctx_client_config config{ true, true, true, false, "" };
    const roctx_client<>      client(config);
    EXPECT_TRUE(client.should_write_markers());
}

TEST_F(roctx_client_test, should_write_with_filter_not_in_region)
{
    using namespace rocprofsys::rocprofiler_sdk;

    // is_write_enabled = true, has region filter, but no active region
    const roctx_client_config config{ true, true, true, false, "Region 1" };
    const roctx_client<>      client(config);
    EXPECT_FALSE(client.should_write_markers());
}

// ============================================================================
// Integration tests: events driven through the client's controller
//
// Each test creates a roctx_client, obtains its trace_control via
// get_controller(), registers callback counters, and simulates events
// by calling the controller's public handle_* methods. Assertions verify
// client.should_write_markers() returns the correct value at each point
// and that start/stop callbacks fire at the right times.
// ============================================================================

class roctx_client_control_test : public ::testing::Test
{
protected:
    using roctx_client_t = rocprofsys::rocprofiler_sdk::roctx_client<>;
    using roctx_config_t = rocprofsys::rocprofiler_sdk::roctx_client_config;

    int start_count = 0;
    int stop_count  = 0;

    /// Create a client and register callback counters on its controller.
    /// Uses is_write_enabled=true with no backends (perfetto/timemory off)
    /// so should_write_markers() purely reflects controller state.
    std::unique_ptr<roctx_client_t> make_client(const std::string& regions)
    {
        const roctx_config_t config{ true, false, false, false, regions };
        auto                 client = std::make_unique<roctx_client_t>(config);

        auto ctrl = client->get_controller();
        ctrl->register_region_start_callback([this]() { start_count++; });
        ctrl->register_region_stop_callback([this]() { stop_count++; });

        return client;
    }
};

// ---------------------------------------------------------------------------
// Pause / Resume (no region filter)
// ---------------------------------------------------------------------------

// Scenario (no region filter):
//   CodeZ            => profiled
//   CodeA            => profiled
//   roctx_pause      => stop callback fires (controls main tracing context)
//   CodeB            => markers still written (no filter => should_write always true)
//   roctx_resume     => start callback fires
//   CodeC            => profiled
//   CodeD            => profiled
//
// Without a region filter, should_write_markers() always returns true.
// Pause/resume only affects the main tracing context via callbacks.
TEST_F(roctx_client_control_test, pause_resume_no_filter)
{
    auto client = make_client("");
    auto ctrl   = client->get_controller();

    EXPECT_FALSE(ctrl->region_filter_active());
    EXPECT_TRUE(client->should_write_markers());

    // Pause: stop callback fires, but should_write stays true (no filter)
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_TRUE(client->should_write_markers());

    // Resume: start callback fires
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());
}

// ---------------------------------------------------------------------------
// Selective Region Tracing - Example 1: Normal Case
// ---------------------------------------------------------------------------

// Scenario:
//   Code-Block A                         => NOT profiled (outside region)
//   Region-Start "Region 1" (id=1)       => start callback
//     Code-Block B                       => profiled
//     Region-Start "Region 2" (id=2)     => ignored (not target)
//       Code-Block C                     => profiled (Region 1 still active)
//     Region-Stop "Region 2" (id=2)      => ignored
//     Code-Block D                       => profiled
//   Region-Stop "Region 1" (id=1)        => stop callback
//   Region-Start "Region 3" (id=3)       => ignored (not target)
//     Code-Block E                       => NOT profiled
//   Region-Stop "Region 3" (id=3)        => ignored
//   Region-Start "Region 1" (id=4)       => start callback
//     Code-Block F                       => profiled
//   Region-Stop "Region 1" (id=4)        => stop callback
//   Code-Block G                         => NOT profiled
//
// Expected profiled: {B, C, D, F}
TEST_F(roctx_client_control_test, selective_region_normal)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    EXPECT_TRUE(ctrl->region_filter_active());

    // Code-Block A: outside target region
    EXPECT_FALSE(client->should_write_markers());

    // Region-Start "Region 1"
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());  // B

    // Region-Start "Region 2" (not a target)
    ctrl->handle_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);                    // no new callback
    EXPECT_TRUE(client->should_write_markers());  // C (Region 1 still active)

    // Region-Stop "Region 2" (not tracked)
    ctrl->handle_range_stop(2);
    EXPECT_TRUE(client->should_write_markers());  // D

    // Region-Stop "Region 1"
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());

    // Region-Start "Region 3" (not a target)
    ctrl->handle_range_start(3, "Region 3");
    EXPECT_FALSE(client->should_write_markers());  // E: not profiled

    // Region-Stop "Region 3"
    ctrl->handle_range_stop(3);
    EXPECT_FALSE(client->should_write_markers());

    // Region-Start "Region 1" again (new range id)
    ctrl->handle_range_start(4, "Region 1");
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(client->should_write_markers());  // F

    // Region-Stop "Region 1"
    ctrl->handle_range_stop(4);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(client->should_write_markers());  // G: not profiled
}

// ---------------------------------------------------------------------------
// Selective Region + Pause/Resume - Example 2
// ---------------------------------------------------------------------------

// Scenario:
//   CodeZ                          => NOT profiled (outside region)
//   Push Region1 (id=1)            => start callback
//   CodeA                          => profiled
//   roctx_pause                    => stop callback; paused
//   CodeB                          => NOT profiled (paused)
//   roctx_resume                   => start callback; resumed
//   CodeC                          => profiled
//   Pop Region1 (id=1)             => stop callback
//   CodeD                          => NOT profiled
//
// Expected profiled: {A, C}
TEST_F(roctx_client_control_test, selective_region_pause_resume_inside)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    // CodeZ: outside region
    EXPECT_FALSE(client->should_write_markers());

    // Push Region1
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());  // CodeA

    // roctx_pause
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());  // CodeB: not profiled

    // roctx_resume (paused is true, inside region => succeeds)
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(client->should_write_markers());  // CodeC

    // Pop Region1
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(client->should_write_markers());  // CodeD
}

// ---------------------------------------------------------------------------
// Selective Region + Pause/Resume - Example 3
// ---------------------------------------------------------------------------

// Scenario:
//   roctx_pause                    => outside region => ignored
//   CodeZ                          => NOT profiled (outside region)
//   Push Region1 (id=1)            => start callback (pause was ignored)
//   CodeA                          => profiled
//   CodeB                          => profiled
//   roctx_resume                   => not paused => ignored
//   CodeC                          => profiled
//   Pop Region1 (id=1)             => stop callback
//   CodeD                          => NOT profiled
//
// Expected profiled: {A, B, C}
TEST_F(roctx_client_control_test, selective_region_pause_outside_resume_inside)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    // roctx_pause outside region: ignored (region filter active, no active ranges)
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 0);  // no callback fired

    // CodeZ: outside region
    EXPECT_FALSE(client->should_write_markers());

    // Push Region1 (pause was ignored, so not paused)
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());  // CodeA

    // CodeB: still profiled
    EXPECT_TRUE(client->should_write_markers());

    // roctx_resume: not paused => ignored
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);  // no new callback

    // CodeC: still profiled
    EXPECT_TRUE(client->should_write_markers());

    // Pop Region1
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());  // CodeD
}

// ---------------------------------------------------------------------------
// Selective Region + Pause/Resume - Example 4
// ---------------------------------------------------------------------------

// Scenario:
//   Push Region1 (id=1)           => start callback
//   CodeA                         => profiled
//   roctx_pause                   => stop callback; paused
//   CodeC                         => NOT profiled (paused)
//   Pop Region1 (id=1)            => region ends while paused; warning;
//                                    paused reset to false; stop callback
//   CodeD                         => NOT profiled (outside region)
//   roctx_resume                  => outside region => ignored
//
// Expected profiled: {A}
TEST_F(roctx_client_control_test, selective_region_pause_then_region_ends)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    // Push Region1
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());  // CodeA

    // roctx_pause
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());  // CodeC: not profiled

    // Pop Region1: region ends while paused.
    // handle_range_stop sees user_paused=true => logs warning,
    // resets paused to false, triggers stop callbacks.
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(client->should_write_markers());  // CodeD: outside region

    // roctx_resume: paused was reset to false by range_stop,
    // also outside region => ignored
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);  // no new callback
    EXPECT_FALSE(client->should_write_markers());
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

TEST_F(roctx_client_control_test, double_pause_is_ignored)
{
    auto client = make_client("");
    auto ctrl   = client->get_controller();

    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);

    // Second pause is ignored (already paused)
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);

    // No region filter => should_write always true; pause only affects callbacks
    EXPECT_TRUE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, resume_without_pause_is_ignored)
{
    auto client = make_client("");
    auto ctrl   = client->get_controller();

    // Resume without prior pause
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 0);
    EXPECT_TRUE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, nested_target_regions)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    EXPECT_FALSE(client->should_write_markers());

    // First instance
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());

    // Nested second instance (same region name, different range id)
    ctrl->handle_range_start(2, "Region 1");
    EXPECT_EQ(start_count, 1);  // already active, no extra callback
    EXPECT_TRUE(client->should_write_markers());

    // Stop first - still have second
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // not yet empty
    EXPECT_TRUE(client->should_write_markers());

    // Stop second - now empty
    ctrl->handle_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, multiple_target_regions)
{
    auto client = make_client("Region 1,Region 2");
    auto ctrl   = client->get_controller();

    EXPECT_TRUE(ctrl->region_filter_active());
    EXPECT_FALSE(client->should_write_markers());

    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(client->should_write_markers());

    ctrl->handle_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);  // already active
    EXPECT_TRUE(client->should_write_markers());

    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // Region 2 still active
    EXPECT_TRUE(client->should_write_markers());

    ctrl->handle_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, shutdown_clears_state)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_start(1, "Region 1");
    EXPECT_TRUE(client->should_write_markers());

    client->shutdown();

    // After shutdown: region filter cleared => should_write returns true
    EXPECT_FALSE(ctrl->region_filter_active());
    EXPECT_TRUE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, stop_unknown_range_is_noop)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_stop(999);
    EXPECT_EQ(stop_count, 0);
    EXPECT_FALSE(client->should_write_markers());
}

TEST_F(roctx_client_control_test, start_with_null_message_is_ignored)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_start(1, nullptr);
    EXPECT_EQ(start_count, 0);
    EXPECT_FALSE(client->should_write_markers());
}
