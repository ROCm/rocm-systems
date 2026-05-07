// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/rocprofiler-sdk/roctx_client.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/trace_control.hpp"

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

// ============================================================================
// Mock marker policy — shared by all fixtures
// ============================================================================

namespace
{

using rocprofsys::rocprofiler_sdk::annotation_entry;
using rocprofsys::trace_cache::region_sample;

using ::testing::NiceMock;

class mock_api
{
public:
    MOCK_METHOD(void, push_timemory, (std::string_view name));
    MOCK_METHOD(void, pop_timemory, (std::string_view name));
    MOCK_METHOD(void, push_perfetto_ts,
                (const char* name, std::uint64_t ts, std::uint64_t flow_id,
                 const std::vector<annotation_entry>& annotations));
    MOCK_METHOD(void, pop_perfetto_ts,
                (const char* name, std::uint64_t ts,
                 const std::vector<annotation_entry>& annotations));
    MOCK_METHOD(void, add_string, (std::string_view string_value));
    MOCK_METHOD(void, store_region, (const region_sample& sample));
    MOCK_METHOD(void, add_thread_info,
                (const rocprofsys::trace_cache::info::thread& thread_info));
};

struct mock_marker_policy
{
    static inline std::unique_ptr<NiceMock<mock_api>> api{};

    static void reset() { api = std::make_unique<NiceMock<mock_api>>(); }

    static void push_timemory(std::string_view name) { api->push_timemory(name); }
    static void pop_timemory(std::string_view name) { api->pop_timemory(name); }

    static void push_perfetto_ts(const char* name, std::uint64_t ts,
                                 std::uint64_t                        flow_id,
                                 const std::vector<annotation_entry>& annotations)
    {
        api->push_perfetto_ts(name, ts, flow_id, annotations);
    }

    static void pop_perfetto_ts(const char* name, std::uint64_t ts,
                                const std::vector<annotation_entry>& annotations)
    {
        api->pop_perfetto_ts(name, ts, annotations);
    }

    static void add_string(std::string_view string_value)
    {
        api->add_string(string_value);
    }
    static void store_region(const region_sample& sample) { api->store_region(sample); }
    static void add_thread_info(const rocprofsys::trace_cache::info::thread& thread_info)
    {
        api->add_thread_info(thread_info);
    }
};

}  // namespace

class mock_cleanup_base : public ::testing::Test
{
protected:
    void SetUp() override { mock_marker_policy::reset(); }
    void TearDown() override { mock_marker_policy::api.reset(); }
};

// ============================================================================
// roctx_client construction tests
// ============================================================================

class roctx_client_test : public mock_cleanup_base
{};

TEST_F(roctx_client_test, constructor_creates_controller)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config        config{ true, true, true, false, "TestRegion" };
    roctx_client<mock_marker_policy> client(config);
    EXPECT_NE(client.get_controller(), nullptr);
}

TEST_F(roctx_client_test, constructor_without_region_filter)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config        config{ true, true, true, false, "" };
    roctx_client<mock_marker_policy> client(config);
    EXPECT_NE(client.get_controller(), nullptr);
    EXPECT_FALSE(client.get_controller()->region_filter_active());
}

TEST_F(roctx_client_test, constructor_with_region_filter)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config        config{ true, true, true, false, "Region 1" };
    roctx_client<mock_marker_policy> client(config);
    EXPECT_TRUE(client.get_controller()->region_filter_active());
}

TEST_F(roctx_client_test, should_write_no_filter)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config              config{ true, true, true, false, "" };
    const roctx_client<mock_marker_policy> client(config);
    EXPECT_TRUE(client.get_controller()->should_write_markers());
}

TEST_F(roctx_client_test, should_write_with_filter_not_in_region)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config              config{ true, true, true, false, "Region 1" };
    const roctx_client<mock_marker_policy> client(config);
    EXPECT_FALSE(client.get_controller()->should_write_markers());
}

// ============================================================================
// Integration tests: events driven through the client's controller
//
// Each test creates a roctx_client, obtains its trace_control via
// get_controller(), registers callback counters, and simulates events
// by calling the controller's public handle_* methods. Assertions verify
// ctrl->should_write_markers() returns the correct value at each point
// and that start/stop callbacks fire at the right times.
// ============================================================================

class roctx_client_control_test : public mock_cleanup_base
{
protected:
    using roctx_client_t = rocprofsys::rocprofiler_sdk::roctx_client<mock_marker_policy>;
    using roctx_config_t = rocprofsys::rocprofiler_sdk::roctx_client_config;

    int start_count = 0;
    int stop_count  = 0;

    /// Create a client and register callback counters on its controller.
    /// Uses is_write_enabled=true with no backends (perfetto/timemory off)
    /// so ctrl->should_write_markers() purely reflects controller state.
    std::unique_ptr<roctx_client_t> make_client(const std::string& regions)
    {
        const roctx_config_t config{ true, false, false, false, regions };
        auto                 client = std::make_unique<roctx_client_t>(config);

        auto ctrl = client->get_controller();
        ctrl->register_region_pauser_resume_callbacks([this]() { start_count++; },
                                                      [this]() { stop_count++; });

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
    EXPECT_TRUE(ctrl->should_write_markers());

    // Pause: stop callback fires, but should_write stays true (no filter)
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());

    // Resume: start callback fires
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());
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
    EXPECT_FALSE(ctrl->should_write_markers());

    // Region-Start "Region 1"
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());  // B

    // Region-Start "Region 2" (not a target)
    ctrl->handle_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);                  // no new callback
    EXPECT_TRUE(ctrl->should_write_markers());  // C (Region 1 still active)

    // Region-Stop "Region 2" (not tracked)
    ctrl->handle_range_stop(2);
    EXPECT_TRUE(ctrl->should_write_markers());  // D

    // Region-Stop "Region 1"
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());

    // Region-Start "Region 3" (not a target)
    ctrl->handle_range_start(3, "Region 3");
    EXPECT_FALSE(ctrl->should_write_markers());  // E: not profiled

    // Region-Stop "Region 3"
    ctrl->handle_range_stop(3);
    EXPECT_FALSE(ctrl->should_write_markers());

    // Region-Start "Region 1" again (new range id)
    ctrl->handle_range_start(4, "Region 1");
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(ctrl->should_write_markers());  // F

    // Region-Stop "Region 1"
    ctrl->handle_range_stop(4);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(ctrl->should_write_markers());  // G: not profiled
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
    EXPECT_FALSE(ctrl->should_write_markers());

    // Push Region1
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());  // CodeA

    // roctx_pause
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());  // CodeB: not profiled

    // roctx_resume (paused is true, inside region => succeeds)
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(ctrl->should_write_markers());  // CodeC

    // Pop Region1
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(ctrl->should_write_markers());  // CodeD
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
    EXPECT_FALSE(ctrl->should_write_markers());

    // Push Region1 (pause was ignored, so not paused)
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());  // CodeA

    // CodeB: still profiled
    EXPECT_TRUE(ctrl->should_write_markers());

    // roctx_resume: not paused => ignored
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);  // no new callback

    // CodeC: still profiled
    EXPECT_TRUE(ctrl->should_write_markers());

    // Pop Region1
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());  // CodeD
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
//                                    paused reset to false; NO stop callback
//                                    (already fired by pause)
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
    EXPECT_TRUE(ctrl->should_write_markers());  // CodeA

    // roctx_pause
    ctrl->handle_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());  // CodeC: not profiled

    // Pop Region1: region ends while paused.
    // handle_range_stop sees user_paused=true => logs warning,
    // resets paused to false. Stop callbacks NOT fired (already fired by pause).
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 1);                    // no double-stop
    EXPECT_FALSE(ctrl->should_write_markers());  // CodeD: outside region

    // roctx_resume: paused was reset to false by range_stop,
    // also outside region => ignored
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 1);  // no new callback
    EXPECT_FALSE(ctrl->should_write_markers());
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
    EXPECT_TRUE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, resume_without_pause_is_ignored)
{
    auto client = make_client("");
    auto ctrl   = client->get_controller();

    // Resume without prior pause
    ctrl->handle_resume();
    EXPECT_EQ(start_count, 0);
    EXPECT_TRUE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, nested_target_regions)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    EXPECT_FALSE(ctrl->should_write_markers());

    // First instance
    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());

    // Nested second instance (same region name, different range id)
    ctrl->handle_range_start(2, "Region 1");
    EXPECT_EQ(start_count, 1);  // already active, no extra callback
    EXPECT_TRUE(ctrl->should_write_markers());

    // Stop first - still have second
    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // not yet empty
    EXPECT_TRUE(ctrl->should_write_markers());

    // Stop second - now empty
    ctrl->handle_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, multiple_target_regions)
{
    auto client = make_client("Region 1,Region 2");
    auto ctrl   = client->get_controller();

    EXPECT_TRUE(ctrl->region_filter_active());
    EXPECT_FALSE(ctrl->should_write_markers());

    ctrl->handle_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(ctrl->should_write_markers());

    ctrl->handle_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);  // already active
    EXPECT_TRUE(ctrl->should_write_markers());

    ctrl->handle_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // Region 2 still active
    EXPECT_TRUE(ctrl->should_write_markers());

    ctrl->handle_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, shutdown_clears_state)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_start(1, "Region 1");
    EXPECT_TRUE(ctrl->should_write_markers());

    ctrl->shutdown();

    EXPECT_FALSE(ctrl->region_filter_active());
    EXPECT_TRUE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, stop_unknown_range_is_noop)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_stop(999);
    EXPECT_EQ(stop_count, 0);
    EXPECT_FALSE(ctrl->should_write_markers());
}

TEST_F(roctx_client_control_test, start_with_null_message_is_ignored)
{
    auto client = make_client("Region 1");
    auto ctrl   = client->get_controller();

    ctrl->handle_range_start(1, nullptr);
    EXPECT_EQ(start_count, 0);
    EXPECT_FALSE(ctrl->should_write_markers());
}

// ============================================================================
// Marker write data propagation tests
//
// These tests verify that marker_writer correctly propagates data fields
// (name, timestamps, correlation IDs, args, thread info) to the storage
// layer. Uses a mock marker policy to capture and verify the data flow.
// ============================================================================

namespace
{

using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StrEq;

MATCHER_P2(IsAnnotation, key, value, "")
{
    return std::string(arg.key) == key &&
           std::holds_alternative<std::uint64_t>(arg.value) &&
           std::get<std::uint64_t>(arg.value) == static_cast<std::uint64_t>(value);
}

using mock_marker_writer = rocprofsys::rocprofiler_sdk::marker_writer<mock_marker_policy>;

rocprofiler_callback_tracing_record_t
make_record(std::uint64_t thread_id, std::uint64_t corr_internal,
            std::uint64_t corr_external)
{
    rocprofiler_callback_tracing_record_t record{};
    record.thread_id                     = thread_id;
    record.correlation_id.internal       = corr_internal;
    record.correlation_id.external.value = corr_external;
    return record;
}

}  // namespace

class marker_write_test : public mock_cleanup_base
{};

TEST_F(marker_write_test, all_backends_with_annotations)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_record(42, 100, 200);

    constexpr auto str = "rocm_marker_api";
    EXPECT_CALL(mock, add_string(std::string_view(str)));
    EXPECT_CALL(mock, push_timemory(std::string_view("my_region")));
    EXPECT_CALL(mock, pop_timemory(std::string_view("my_region")));
    EXPECT_CALL(mock, push_perfetto_ts(StrEq("my_region"), 1000, 100,
                                       ElementsAre(IsAnnotation("begin_ns", 1000u),
                                                   IsAnnotation("stack_id", 100u))));
    EXPECT_CALL(mock, pop_perfetto_ts(StrEq("my_region"), 2000,
                                      ElementsAre(IsAnnotation("end_ns", 2000u))));
    const auto thread_info =
        rocprofsys::trace_cache::info::thread{ getppid(), getpid(), 42u, 0, 0, "{}" };
    EXPECT_CALL(mock, add_thread_info(thread_info));
    EXPECT_CALL(mock,
                store_region(AllOf(Field(&region_sample::thread_id, 42u),
                                   Field(&region_sample::name, "my_region"),
                                   Field(&region_sample::start_timestamp, 1000u),
                                   Field(&region_sample::end_timestamp, 2000u),
                                   Field(&region_sample::correlation_id_internal, 100u),
                                   Field(&region_sample::correlation_id_ancestor, 200u),
                                   Field(&region_sample::args_str, "arg1=val1"),
                                   Field(&region_sample::call_stack, "{}"),
                                   Field(&region_sample::category, "rocm_marker_api"))));

    const mock_marker_writer writer(true, true, true);
    writer.write_begin("my_region");
    writer.write_end("my_region", 1000, 2000, "arg1=val1", record);
}

TEST_F(marker_write_test, perfetto_disabled)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_record(1, 10, 20);

    EXPECT_CALL(mock, push_timemory(_));
    EXPECT_CALL(mock, pop_timemory(_));
    EXPECT_CALL(mock, push_perfetto_ts(_, _, _, _)).Times(0);
    EXPECT_CALL(mock, pop_perfetto_ts(_, _, _)).Times(0);
    EXPECT_CALL(mock, store_region(_));

    const mock_marker_writer writer(false, true, false);
    writer.write_begin("r");
    writer.write_end("r", 100, 200, "{}", record);
}

TEST_F(marker_write_test, timemory_disabled_no_annotations)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_record(1, 10, 20);

    EXPECT_CALL(mock, push_timemory(_)).Times(0);
    EXPECT_CALL(mock, pop_timemory(_)).Times(0);
    EXPECT_CALL(mock, push_perfetto_ts(_, 100, _, IsEmpty()));
    EXPECT_CALL(mock, pop_perfetto_ts(_, 200, IsEmpty()));
    EXPECT_CALL(mock, store_region(_));

    const mock_marker_writer writer(true, false, false);
    writer.write_begin("r");
    writer.write_end("r", 100, 200, "{}", record);
}

TEST_F(marker_write_test, sequential_writes_propagate_independent_data)
{
    auto& mock = *mock_marker_policy::api;

    std::vector<region_sample> captured;
    ON_CALL(mock, store_region(_)).WillByDefault([&](const region_sample& s) {
        captured.push_back(s);
    });

    auto record1 = make_record(1, 100, 200);
    auto record2 = make_record(2, 300, 400);

    const mock_marker_writer writer(false, false, false);
    writer.write_end("First", 1000, 2000, "a=1", record1);
    writer.write_end("Second", 3000, 4000, "b=2", record2);

    ASSERT_THAT(captured, SizeIs(2));

    EXPECT_EQ(captured[0].name, "First");
    EXPECT_EQ(captured[0].thread_id, 1u);
    EXPECT_EQ(captured[0].start_timestamp, 1000u);
    EXPECT_EQ(captured[0].end_timestamp, 2000u);
    EXPECT_EQ(captured[0].correlation_id_internal, 100u);
    EXPECT_EQ(captured[0].correlation_id_ancestor, 200u);
    EXPECT_EQ(captured[0].args_str, "a=1");

    EXPECT_EQ(captured[1].name, "Second");
    EXPECT_EQ(captured[1].thread_id, 2u);
    EXPECT_EQ(captured[1].start_timestamp, 3000u);
    EXPECT_EQ(captured[1].end_timestamp, 4000u);
    EXPECT_EQ(captured[1].correlation_id_internal, 300u);
    EXPECT_EQ(captured[1].correlation_id_ancestor, 400u);
    EXPECT_EQ(captured[1].args_str, "b=2");
}

TEST_F(marker_write_test, write_end_with_empty_args)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_record(1, 10, 20);

    EXPECT_CALL(mock, store_region(Field(&region_sample::args_str, "")));

    const mock_marker_writer writer(false, false, false);
    writer.write_end("R", 100, 200, "", record);
}
