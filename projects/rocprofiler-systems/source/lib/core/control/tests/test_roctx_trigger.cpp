// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/session.hpp"
#include "core/control/triggers/roctx.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace
{

using rocprofsys::control::action;
using rocprofsys::control::scope;
using rocprofsys::control::session;
using rocprofsys::control::triggers::roctx;

std::unique_ptr<roctx>
make_roctx_trigger(const std::shared_ptr<session>& sess, const std::string& regions)
{
    return std::make_unique<roctx>(sess, regions);
}

}  // namespace

// ============================================================================
// roctx trigger construction tests
// ============================================================================

class roctx_trigger_test : public ::testing::Test
{
protected:
    std::shared_ptr<session> m_session = std::make_shared<session>();
};

TEST_F(roctx_trigger_test, constructor_registers_trigger_on_supplied_session)
{
    const auto trigger = make_roctx_trigger(m_session, "TestRegion");

    EXPECT_TRUE(trigger->filter_active());
    // A filter is configured and no region is open, so the trigger registers
    // as paused and holds the session down from construction.
    EXPECT_FALSE(trigger->should_write_markers());
    EXPECT_FALSE(m_session->is_active());
}

TEST_F(roctx_trigger_test, constructor_without_region_filter)
{
    const auto trigger = make_roctx_trigger(m_session, "");

    EXPECT_FALSE(trigger->filter_active());
}

TEST_F(roctx_trigger_test, constructor_with_region_filter)
{
    const auto trigger = make_roctx_trigger(m_session, "Region 1");

    EXPECT_TRUE(trigger->filter_active());
}

TEST_F(roctx_trigger_test, should_write_no_filter)
{
    const auto trigger = make_roctx_trigger(m_session, "");

    EXPECT_TRUE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_test, should_write_with_filter_not_in_region)
{
    const auto trigger = make_roctx_trigger(m_session, "Region 1");

    EXPECT_FALSE(trigger->should_write_markers());
}

// ============================================================================
// Integration tests: events driven through the roctx trigger
//
// Each test creates a trigger, registers callback counters as a subscriber on
// its session, and simulates events by calling the trigger's on_range_* /
// on_pause / on_resume methods. Assertions verify should_write_markers()
// returns the correct value at each point and that start/stop callbacks fire
// on subscriber-state transitions.
// ============================================================================

class roctx_trigger_control_test : public ::testing::Test
{
protected:
    int start_count = 0;
    int stop_count  = 0;

    std::shared_ptr<session> m_session = std::make_shared<session>();

    void subscribe_counters()
    {
        m_session->subscribe({ .on_pause  = [this]() { stop_count++; },
                               .on_resume = [this]() { start_count++; },
                               .name      = "test_counters",
                               .scopes    = { scope::global } });
    }

    /// Create a trigger, then subscribe callback counters on its session. The
    /// subscriber therefore misses the trigger's initial transition.
    std::unique_ptr<roctx> make_trigger(const std::string& regions)
    {
        auto trigger = make_roctx_trigger(m_session, regions);
        subscribe_counters();
        return trigger;
    }

    /// Subscribes before constructing the trigger, the way library.cpp does:
    /// registering a trigger broadcasts its initial transition immediately, so
    /// only a subscriber attached beforehand observes it. make_trigger() above
    /// uses the reverse order and therefore misses that first pause.
    std::unique_ptr<roctx> make_trigger_production_order(const std::string& regions)
    {
        subscribe_counters();
        return make_roctx_trigger(m_session, regions);
    }

    static constexpr std::uint64_t k_unknown_range_id = 999;
};

// ---------------------------------------------------------------------------
// Pause / Resume (no region filter)
// ---------------------------------------------------------------------------

// Scenario (no region filter):
//   CodeA            => profiled
//   roctx_pause      => stop callback fires; should_write becomes false
//   CodeB            => NOT profiled (paused)
//   roctx_resume     => start callback fires; should_write becomes true
//   CodeC            => profiled
//   CodeD            => profiled
//
// Without a region filter there is no region to be outside of, so markers are
// written whenever the trigger is not paused. An explicit pause still
// suppresses them - compute_should_write() short-circuits on the paused flag
// before the filter is ever consulted.
TEST_F(roctx_trigger_control_test, pause_resume_no_filter)
{
    auto trigger = make_trigger("");

    EXPECT_FALSE(trigger->filter_active());
    EXPECT_TRUE(trigger->should_write_markers());

    // Pause: stop callback fires and marker writes are suppressed globally.
    trigger->on_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());

    // Resume: start callback fires
    trigger->on_resume();
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_control_test,
       no_filter_range_started_while_paused_suppresses_markers)
{
    auto trigger = make_trigger("");

    trigger->on_pause();
    EXPECT_FALSE(trigger->should_write_markers());

    // Opening a range must not lift the pause when no region filter is set.
    trigger->on_range_start(1, "Region1");
    EXPECT_FALSE(trigger->should_write_markers());

    trigger->on_resume();
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(1);
    EXPECT_TRUE(trigger->should_write_markers());
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
TEST_F(roctx_trigger_control_test, selective_region_normal)
{
    auto trigger = make_trigger("Region 1");

    EXPECT_TRUE(trigger->filter_active());

    // Code-Block A: outside target region
    EXPECT_FALSE(trigger->should_write_markers());

    // Region-Start "Region 1"
    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());  // B

    // Region-Start "Region 2" (not a target)
    trigger->on_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);                     // no new callback
    EXPECT_TRUE(trigger->should_write_markers());  // C (Region 1 still active)

    // Region-Stop "Region 2" (not tracked)
    trigger->on_range_stop(2);
    EXPECT_TRUE(trigger->should_write_markers());  // D

    // Region-Stop "Region 1"
    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());

    // Region-Start "Region 3" (not a target)
    trigger->on_range_start(3, "Region 3");
    EXPECT_FALSE(trigger->should_write_markers());  // E: not profiled

    // Region-Stop "Region 3"
    trigger->on_range_stop(3);
    EXPECT_FALSE(trigger->should_write_markers());

    // Region-Start "Region 1" again (new range id)
    trigger->on_range_start(4, "Region 1");
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(trigger->should_write_markers());  // F

    // Region-Stop "Region 1"
    trigger->on_range_stop(4);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(trigger->should_write_markers());  // G: not profiled
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
TEST_F(roctx_trigger_control_test, selective_region_pause_resume_inside)
{
    auto trigger = make_trigger("Region 1");

    // CodeZ: outside region
    EXPECT_FALSE(trigger->should_write_markers());

    // Push Region1
    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());  // CodeA

    // roctx_pause
    trigger->on_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());  // CodeB: not profiled

    // roctx_resume (paused is true, inside region => succeeds)
    trigger->on_resume();
    EXPECT_EQ(start_count, 2);
    EXPECT_TRUE(trigger->should_write_markers());  // CodeC

    // Pop Region1
    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(trigger->should_write_markers());  // CodeD
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
TEST_F(roctx_trigger_control_test, selective_region_pause_outside_resume_inside)
{
    auto trigger = make_trigger("Region 1");

    // roctx_pause outside region: ignored (region filter active, no active ranges)
    trigger->on_pause();
    EXPECT_EQ(stop_count, 0);

    // CodeZ: outside region
    EXPECT_FALSE(trigger->should_write_markers());

    // Push Region1 (pause was ignored, so not paused)
    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());  // CodeA

    // CodeB: still profiled
    EXPECT_TRUE(trigger->should_write_markers());

    // roctx_resume: not paused => ignored
    trigger->on_resume();
    EXPECT_EQ(start_count, 1);  // no new callback

    // CodeC: still profiled
    EXPECT_TRUE(trigger->should_write_markers());

    // Pop Region1
    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());  // CodeD
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
TEST_F(roctx_trigger_control_test, selective_region_pause_then_region_ends)
{
    auto trigger = make_trigger("Region 1");

    // Push Region1
    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());  // CodeA

    // roctx_pause
    trigger->on_pause();
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());  // CodeC: not profiled

    // Pop Region1: region ends while paused.
    // Trigger sees user_paused=true => logs warning, resets paused to false.
    // Stop callbacks NOT fired (already fired by pause).
    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 1);                       // no double-stop
    EXPECT_FALSE(trigger->should_write_markers());  // CodeD: outside region

    // roctx_resume: paused was reset to false by range_stop,
    // also outside region => ignored
    trigger->on_resume();
    EXPECT_EQ(start_count, 1);  // no new callback
    EXPECT_FALSE(trigger->should_write_markers());
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

TEST_F(roctx_trigger_control_test, double_pause_is_ignored)
{
    auto trigger = make_trigger("");

    trigger->on_pause();
    EXPECT_EQ(stop_count, 1);

    // Second pause is ignored (already paused)
    trigger->on_pause();
    EXPECT_EQ(stop_count, 1);

    // Still paused after the ignored second pause, so marker writes remain suppressed.
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_control_test, resume_without_pause_is_ignored)
{
    auto trigger = make_trigger("");

    // Resume without prior pause
    trigger->on_resume();
    EXPECT_EQ(start_count, 0);
    EXPECT_TRUE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_control_test, nested_target_regions)
{
    auto trigger = make_trigger("Region 1");

    EXPECT_FALSE(trigger->should_write_markers());

    // First instance
    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());

    // Nested second instance (same region name, different range id)
    trigger->on_range_start(2, "Region 1");
    EXPECT_EQ(start_count, 1);  // already active, no extra callback
    EXPECT_TRUE(trigger->should_write_markers());

    // Stop first - still have second
    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // not yet empty
    EXPECT_TRUE(trigger->should_write_markers());

    // Stop second - now empty
    trigger->on_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_control_test, multiple_target_regions)
{
    auto trigger = make_trigger("Region 1,Region 2");

    EXPECT_TRUE(trigger->filter_active());
    EXPECT_FALSE(trigger->should_write_markers());

    trigger->on_range_start(1, "Region 1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_start(2, "Region 2");
    EXPECT_EQ(start_count, 1);  // already active
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(1);
    EXPECT_EQ(stop_count, 0);  // Region 2 still active
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(2);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());
}

// session::shutdown() clears subscribers + actions only; trigger state (e.g.
// the range filter) persists, since the trigger's creator owns it. With no
// actions recorded, the resolved state defaults back to active.
//
// Starts from a *paused* session so the active flag has to change - asserting
// only the trigger would pass even if shutdown() did nothing, since shutdown
// never touches trigger state.
TEST_F(roctx_trigger_control_test, shutdown_reactivates_session_and_drops_subscribers)
{
    auto trigger = make_trigger("Region 1");

    // Filter set with no region open: the trigger holds the session paused.
    EXPECT_FALSE(m_session->is_active());
    EXPECT_FALSE(trigger->should_write_markers());

    m_session->shutdown();

    EXPECT_TRUE(m_session->is_active());

    // Trigger state is owned by the trigger and survives shutdown.
    EXPECT_TRUE(trigger->filter_active());

    // Subscribers were dropped, so later transitions reach no one.
    const int start_before = start_count;
    const int stop_before  = stop_count;
    trigger->on_range_start(1, "Region 1");
    trigger->on_range_stop(1);
    EXPECT_EQ(start_count, start_before);
    EXPECT_EQ(stop_count, stop_before);
}

TEST_F(roctx_trigger_control_test, subscriber_added_before_trigger_sees_initial_pause)
{
    auto trigger = make_trigger_production_order("Region 1");

    // Filter set with no region open: the trigger registers as paused, and a
    // subscriber attached beforehand is notified of that transition.
    EXPECT_EQ(stop_count, 1);
    EXPECT_EQ(start_count, 0);
    EXPECT_FALSE(m_session->is_active());
    EXPECT_FALSE(trigger->should_write_markers());

    trigger->on_range_start(1, "Region 1");

    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());
}

// Finalization shuts the control session down before destroying the roctx
// trigger. Without that order, unregistering the still-paused trigger resumes
// the scope and calls every subscriber's on_resume mid-teardown.
TEST_F(roctx_trigger_control_test, trigger_destroyed_after_shutdown_sends_no_resume)
{
    auto trigger = make_trigger_production_order("Region 1");
    EXPECT_EQ(stop_count, 1);

    m_session->shutdown();

    const int start_before = start_count;
    trigger.reset();
    EXPECT_EQ(start_count, start_before);
}

// Matched pair for the test above: destroying the trigger really does resume
// the scope and notify when the session is still live, so the zero asserted
// there is meaningful rather than vacuous.
TEST_F(roctx_trigger_control_test, trigger_destroyed_without_shutdown_sends_resume)
{
    auto trigger = make_trigger_production_order("Region 1");
    EXPECT_EQ(stop_count, 1);
    EXPECT_EQ(start_count, 0);

    trigger.reset();

    EXPECT_EQ(start_count, 1);
}

TEST_F(roctx_trigger_control_test, stop_unknown_range_is_noop)
{
    auto trigger = make_trigger("Region 1");

    trigger->on_range_stop(k_unknown_range_id);
    EXPECT_EQ(stop_count, 0);
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_trigger_control_test, start_with_null_message_is_ignored)
{
    auto trigger = make_trigger("Region 1");

    trigger->on_range_start(1, nullptr);
    EXPECT_EQ(start_count, 0);
    EXPECT_FALSE(trigger->should_write_markers());
}

// ============================================================================
// roctxRangePush / roctxRangePop region-filter and pause/resume tests
//
// These tests verify the behavioral contract that roctxRangePush/roctxRangePop
// must deliver: the roctx trigger's region-filter and pause/resume logic must
// engage for push/pop the same way it does for roctxRangeStartA/roctxRangeStop.
//
// They drive the trigger directly, using synthetic range IDs from the
// UINT64_MAX-downward space that callback/marker.hpp reserves for push/pop.
// ============================================================================

class roctx_push_pop_region_test : public roctx_trigger_control_test
{
protected:
    // Synthetic IDs mirror the push-range counter: starts at UINT64_MAX and
    // decrements on each roctxRangePush to avoid colliding with SDK-allocated
    // roctxRangeStart IDs (which count upward from small values).
    static constexpr std::uint64_t k_push_id = std::numeric_limits<std::uint64_t>::max();
};

TEST_F(roctx_push_pop_region_test, push_pop_with_subscriber_added_before_trigger)
{
    auto trigger = make_trigger_production_order("Region1");

    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());

    trigger->on_range_start(k_push_id, "Region1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(k_push_id);
    EXPECT_EQ(stop_count, 2);
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_push_pop_region_test, push_matching_region_resumes_session)
{
    auto trigger = make_trigger("Region1");

    EXPECT_FALSE(trigger->should_write_markers());

    trigger->on_range_start(k_push_id, "Region1");

    EXPECT_TRUE(trigger->should_write_markers());
}

TEST_F(roctx_push_pop_region_test, pop_matching_region_pauses_session)
{
    auto trigger = make_trigger("Region1");

    trigger->on_range_start(k_push_id, "Region1");
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(k_push_id);
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_push_pop_region_test, push_non_matching_region_does_not_activate)
{
    auto trigger = make_trigger("Region1");

    trigger->on_range_start(k_push_id, "OtherRegion");

    EXPECT_FALSE(trigger->should_write_markers());
    EXPECT_EQ(start_count, 0);
}

TEST_F(roctx_push_pop_region_test, resume_callback_fires_on_first_push)
{
    auto trigger = make_trigger("Region1");

    EXPECT_EQ(start_count, 0);
    trigger->on_range_start(k_push_id, "Region1");
    EXPECT_EQ(start_count, 1);
}

TEST_F(roctx_push_pop_region_test, pause_callback_fires_on_last_pop)
{
    auto trigger = make_trigger("Region1");

    trigger->on_range_start(k_push_id, "Region1");
    EXPECT_EQ(stop_count, 0);

    trigger->on_range_stop(k_push_id);
    EXPECT_EQ(stop_count, 1);
}

// Nested pushes of the same region use distinct synthetic IDs (UINT64_MAX,
// UINT64_MAX-1, ...). The trigger resumes on the first push and pauses
// only when the last pop removes the final active ID.
TEST_F(roctx_push_pop_region_test, nested_push_pop_same_region)
{
    auto trigger = make_trigger("Region1");

    // First push: activates
    trigger->on_range_start(k_push_id, "Region1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());

    // Second push: already active — no extra resume callback
    trigger->on_range_start(k_push_id - 1, "Region1");
    EXPECT_EQ(start_count, 1);
    EXPECT_TRUE(trigger->should_write_markers());

    // First pop: removes second ID — first is still active
    trigger->on_range_stop(k_push_id - 1);
    EXPECT_EQ(stop_count, 0);
    EXPECT_TRUE(trigger->should_write_markers());

    // Second pop: removes last ID — pause fires
    trigger->on_range_stop(k_push_id);
    EXPECT_EQ(stop_count, 1);
    EXPECT_FALSE(trigger->should_write_markers());
}

TEST_F(roctx_push_pop_region_test, push_pop_no_filter_always_active)
{
    auto trigger = make_trigger("");

    EXPECT_FALSE(trigger->filter_active());
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_start(k_push_id, "AnyRegion");
    EXPECT_TRUE(trigger->should_write_markers());

    trigger->on_range_stop(k_push_id);
    EXPECT_TRUE(trigger->should_write_markers());
}

// ============================================================================
// Marker write gating: the roctx trigger's own decision combined with the votes
// of every other trigger on the session (is_active_without excludes the roctx
// trigger's own vote so it is not double-counted).
// ============================================================================

class roctx_session_gating_test : public ::testing::Test
{
protected:
    // Independent trigger used to pause the session without touching the
    // roctx trigger itself - reproduces "some other trigger (e.g. the
    // time_window trigger) has paused the global scope".
    class other_trigger
    {
    public:
        static constexpr std::string_view k_trigger_name = "other";

        explicit other_trigger(session& sess)
        : m_session{ sess }
        {
            m_session.register_trigger(k_trigger_name, action::trace);
        }

        ~other_trigger() { m_session.unregister_trigger(k_trigger_name); }

        other_trigger(const other_trigger&)            = delete;
        other_trigger& operator=(const other_trigger&) = delete;
        other_trigger(other_trigger&&)                 = delete;
        other_trigger& operator=(other_trigger&&)      = delete;

        void set_action(action act) const { m_session.set_action(k_trigger_name, act); }

    private:
        session& m_session;
    };

    std::shared_ptr<session> m_session = std::make_shared<session>();
};

TEST_F(roctx_session_gating_test, trigger_and_session_allow_writes_when_fully_active)
{
    const auto trigger = make_roctx_trigger(m_session, "");

    EXPECT_TRUE(trigger->should_write_markers());
    EXPECT_TRUE(m_session->is_active_without(roctx::k_trigger_name));
}

TEST_F(roctx_session_gating_test, session_reports_inactive_when_paused_by_other_trigger)
{
    const auto trigger = make_roctx_trigger(m_session, "");

    const other_trigger other{ *m_session };
    other.set_action(action::pause);

    ASSERT_TRUE(trigger->should_write_markers())
        << "the roctx trigger itself has no filter and is not paused - only the "
           "unrelated 'other' trigger is pausing the session";

    EXPECT_FALSE(m_session->is_active_without(roctx::k_trigger_name))
        << "is_active_without() must respect other triggers' votes";
}

TEST_F(roctx_session_gating_test, own_pause_is_excluded_from_is_active_without)
{
    const auto trigger = make_roctx_trigger(m_session, "Region1");

    ASSERT_FALSE(trigger->should_write_markers())
        << "a region filter is configured but no matching region is active yet";

    EXPECT_FALSE(m_session->is_active())
        << "the trigger's own pause holds the session down";
    EXPECT_TRUE(m_session->is_active_without(roctx::k_trigger_name))
        << "the trigger's own vote must not be counted against itself";
}
