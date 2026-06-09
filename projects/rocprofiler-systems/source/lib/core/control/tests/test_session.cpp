// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/clocks/manual.hpp"
#include "core/control/session.hpp"
#include "core/control/subscriber.hpp"
#include "core/control/trigger.hpp"
#include "core/control/triggers/time_window.hpp"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
using rocprofsys::control::scope;
using rocprofsys::control::session;
using rocprofsys::control::subscriber;
using rocprofsys::control::trigger;
using rocprofsys::control::vote;

class mock_trigger : public trigger
{
public:
    mock_trigger(std::string_view name, scope event_scope, vote initial) noexcept
    : m_name{ name }
    , m_scope{ event_scope }
    , m_initial{ initial }
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return m_name; }
    [[nodiscard]] vote  initial_vote() const noexcept override { return m_initial; }
    [[nodiscard]] scope event_scope() const noexcept override { return m_scope; }

private:
    std::string_view m_name;
    scope            m_scope;
    vote             m_initial;
};

struct call_log
{
    std::mutex               mu;
    std::vector<std::string> events;

    void record(std::string ev)
    {
        std::scoped_lock lk{ mu };
        events.push_back(std::move(ev));
    }

    std::vector<std::string> snapshot()
    {
        std::scoped_lock lk{ mu };
        return events;
    }
};

subscriber
make_logged_subscriber(call_log& log, std::string name,
                       std::vector<scope> scopes = { scope::global })
{
    subscriber s{};
    s.name      = name;
    s.scopes    = std::move(scopes);
    s.on_pause  = [&log, name]() { log.record(name + ":pause"); };
    s.on_resume = [&log, name]() { log.record(name + ":resume"); };
    return s;
}

template <typename Clock>
bool
wait_with_advance(call_log& log, Clock& clk, std::size_t expected_size,
                  rocprofsys::control::clock_duration step)
{
    using namespace std::chrono_literals;
    for(int i = 0; i < 200; ++i)
    {
        if(log.snapshot().size() >= expected_size) return true;
        clk.advance(step);
        std::this_thread::sleep_for(2ms);
    }
    return false;
}
}  // namespace

class session_test : public ::testing::Test
{
protected:
    session  s{};
    call_log log{};
};

TEST_F(session_test, single_scope_subscriber_paused_at_attach_fires_once)
{
    mock_trigger t{ "global_paused", scope::global, vote::paused };
    s.subscribe(make_logged_subscriber(log, "sub"));
    s.attach(t);
    s.force_initial_pause();

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "sub:pause");
    EXPECT_FALSE(s.is_active(scope::global));
}

TEST_F(session_test, single_scope_subscriber_active_at_attach_no_fire)
{
    mock_trigger t{ "global_active", scope::global, vote::active };
    s.subscribe(make_logged_subscriber(log, "sub"));
    s.attach(t);
    s.force_initial_pause();

    EXPECT_TRUE(log.snapshot().empty());
    EXPECT_TRUE(s.is_active(scope::global));
}

TEST_F(session_test, multi_scope_subscriber_pauses_once_when_both_scopes_paused)
{
    mock_trigger tg{ "global_paused", scope::global, vote::paused };
    mock_trigger ts{ "samp_paused", scope::sampling_only, vote::paused };
    s.subscribe(
        make_logged_subscriber(log, "sub", { scope::global, scope::sampling_only }));
    s.attach(tg);
    s.attach(ts);
    s.force_initial_pause();

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "sub:pause");
}

TEST_F(session_test, multi_scope_subscriber_does_not_resume_while_other_scope_paused)
{
    mock_trigger tg{ "global_paused", scope::global, vote::paused };
    mock_trigger ts{ "samp_paused", scope::sampling_only, vote::paused };
    s.subscribe(
        make_logged_subscriber(log, "sub", { scope::global, scope::sampling_only }));
    s.attach(tg);
    s.attach(ts);
    s.force_initial_pause();

    s.publish(ts, vote::active);
    {
        const auto events = log.snapshot();
        ASSERT_EQ(events.size(), 1u);
        EXPECT_EQ(events[0], "sub:pause");
    }
    EXPECT_FALSE(s.is_active(scope::global));
    EXPECT_TRUE(s.is_active(scope::sampling_only));

    s.publish(tg, vote::active);
    {
        const auto events = log.snapshot();
        ASSERT_EQ(events.size(), 2u);
        EXPECT_EQ(events[1], "sub:resume");
    }
    EXPECT_TRUE(s.is_active(scope::global));
}

TEST_F(session_test, force_initial_pause_broadcasts_after_subscribe_after_attach)
{
    // Production parallel: roctx_client attaches its trigger before the
    // library's _dtor lambda subscribes the rocm/sampling/process_sampler/etc.
    // subscribers. force_initial_pause must still deliver on_pause to those
    // subscribers even though their seeded sub.paused already matches the
    // resolved state (otherwise PMC samples leak across the paused window).
    mock_trigger t{ "global_paused", scope::global, vote::paused };
    s.attach(t);
    s.subscribe(make_logged_subscriber(log, "sub"));
    s.force_initial_pause();

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "sub:pause");
}

TEST_F(session_test, registration_order_preserved_on_pause)
{
    s.subscribe(make_logged_subscriber(log, "a"));
    s.subscribe(make_logged_subscriber(log, "b"));
    s.subscribe(make_logged_subscriber(log, "c"));

    mock_trigger t{ "global_active", scope::global, vote::active };
    s.attach(t);
    s.publish(t, vote::paused);

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0], "a:pause");
    EXPECT_EQ(events[1], "b:pause");
    EXPECT_EQ(events[2], "c:pause");
}

TEST_F(session_test, is_active_excluding_ignores_named_trigger)
{
    mock_trigger t1{ "trig_a", scope::global, vote::paused };
    mock_trigger t2{ "trig_b", scope::global, vote::active };
    s.attach(t1);
    s.attach(t2);

    EXPECT_FALSE(s.is_active(scope::global));
    EXPECT_TRUE(s.is_active_excluding("trig_a", scope::global));
    EXPECT_FALSE(s.is_active_excluding("trig_b", scope::global));
}

TEST_F(session_test, has_trigger_reports_registered_trigger_by_name_and_scope)
{
    mock_trigger global_t{ "time_window", scope::global, vote::paused };
    mock_trigger sampling_t{ "time_window", scope::sampling_only, vote::active };

    EXPECT_FALSE(s.has_trigger("time_window", scope::global));
    s.attach(global_t);

    EXPECT_TRUE(s.has_trigger("time_window", scope::global));
    EXPECT_FALSE(s.has_trigger("time_window", scope::sampling_only));

    s.attach(sampling_t);
    EXPECT_TRUE(s.has_trigger("time_window", scope::sampling_only));
}

TEST_F(session_test, shutdown_resets_state)
{
    mock_trigger t{ "global_paused", scope::global, vote::paused };
    s.subscribe(make_logged_subscriber(log, "sub_pre"));
    s.attach(t);
    s.force_initial_pause();
    EXPECT_EQ(log.snapshot().size(), 1u);  // sub_pre paused

    s.shutdown();

    // Re-subscribe a fresh subscriber after shutdown. Even though the previous
    // subscriber was in paused state, shutdown clears the subscribers list, and
    // m_active is reset. So the new subscriber attaches into an active session,
    // and force_initial_pause must NOT fire on it (no trigger paused).
    call_log log2{};
    s.subscribe(make_logged_subscriber(log2, "sub_post"));
    s.force_initial_pause();
    EXPECT_TRUE(log2.snapshot().empty());
    EXPECT_TRUE(s.is_active(scope::global));
}

TEST_F(session_test, terminal_pause_does_not_double_fire)
{
    s.subscribe(make_logged_subscriber(log, "sub"));
    mock_trigger t{ "trig", scope::global, vote::active };
    s.attach(t);

    s.publish(t, vote::paused);
    s.publish(t, vote::paused);  // idempotent re-publish

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "sub:pause");
}

TEST_F(session_test, time_window_with_manual_clock_drives_full_cycle)
{
    using namespace std::chrono_literals;
    using clock_t          = rocprofsys::control::clocks::manual;
    using time_window_t    = rocprofsys::control::triggers::time_window<clock_t>;
    using clock_duration_t = rocprofsys::control::clock_duration;

    clock_t clk{};
    s.subscribe(make_logged_subscriber(log, "sub"));

    constexpr auto delay = clock_duration_t{ 100'000'000 };  // 100 ms virtual
    constexpr auto dur   = clock_duration_t{ 200'000'000 };  // 200 ms virtual
    time_window_t  tw{ s, clk, time_window_t::config{ delay, dur } };
    s.attach(tw);
    s.force_initial_pause();

    // Initial vote is paused (delay > 0); subscriber should observe one pause.
    ASSERT_EQ(log.snapshot().size(), 1u);
    EXPECT_EQ(log.snapshot()[0], "sub:pause");
    EXPECT_FALSE(s.is_active(scope::global));

    tw.start();

    // The worker captures its t0 from clk.now() asynchronously after start().
    // We do not know exactly when, so advance virtual time in chunks while
    // polling. Each advance() wakes any sleep_until that has captured a
    // now-satisfied deadline.
    ASSERT_TRUE(wait_with_advance(log, clk, 2u, delay))
        << "worker did not publish active";
    EXPECT_EQ(log.snapshot()[1], "sub:resume");
    EXPECT_TRUE(s.is_active(scope::global));

    ASSERT_TRUE(wait_with_advance(log, clk, 3u, dur))
        << "worker did not publish terminal pause";
    EXPECT_EQ(log.snapshot()[2], "sub:pause");
    EXPECT_FALSE(s.is_active(scope::global));

    // Worker thread is joined by ~time_window; no explicit stop needed.
}

TEST_F(session_test, time_window_repeats_and_advances_multiple_periods)
{
    using clock_t          = rocprofsys::control::clocks::manual;
    using time_window_t    = rocprofsys::control::triggers::time_window<clock_t>;
    using clock_duration_t = rocprofsys::control::clock_duration;

    clock_t clk{};
    s.subscribe(make_logged_subscriber(log, "sub"));

    constexpr auto delay_a = clock_duration_t{ 100'000'000 };
    constexpr auto dur_a   = clock_duration_t{ 200'000'000 };
    constexpr auto delay_b = clock_duration_t{ 50'000'000 };
    constexpr auto dur_b   = clock_duration_t{ 100'000'000 };

    time_window_t tw{ s, clk,
                      time_window_t::schedule_type{
                          time_window_t::config{ delay_a, dur_a, 2 },
                          time_window_t::config{ delay_b, dur_b, 1 },
                      } };
    s.attach(tw);
    s.force_initial_pause();

    ASSERT_EQ(log.snapshot().size(), 1u);
    EXPECT_EQ(log.snapshot()[0], "sub:pause");

    tw.start();

    ASSERT_TRUE(wait_with_advance(log, clk, 2u, delay_a));
    ASSERT_TRUE(wait_with_advance(log, clk, 3u, dur_a));
    ASSERT_TRUE(wait_with_advance(log, clk, 4u, delay_a));
    ASSERT_TRUE(wait_with_advance(log, clk, 5u, dur_a));
    ASSERT_TRUE(wait_with_advance(log, clk, 6u, delay_b));
    ASSERT_TRUE(wait_with_advance(log, clk, 7u, dur_b));

    const auto events = log.snapshot();
    ASSERT_EQ(events.size(), 7u);
    EXPECT_EQ(events[0], "sub:pause");
    EXPECT_EQ(events[1], "sub:resume");
    EXPECT_EQ(events[2], "sub:pause");
    EXPECT_EQ(events[3], "sub:resume");
    EXPECT_EQ(events[4], "sub:pause");
    EXPECT_EQ(events[5], "sub:resume");
    EXPECT_EQ(events[6], "sub:pause");
    EXPECT_FALSE(s.is_active(scope::global));
}

TEST_F(session_test, subscriber_added_after_delay_elapsed_observes_active_window)
{
    using clock_t          = rocprofsys::control::clocks::manual;
    using time_window_t    = rocprofsys::control::triggers::time_window<clock_t>;
    using clock_duration_t = rocprofsys::control::clock_duration;

    clock_t clk{};

    constexpr auto delay = clock_duration_t{ 100'000'000 };
    constexpr auto dur   = clock_duration_t{ 200'000'000 };
    time_window_t  tw{ s, clk, time_window_t::config{ delay, dur } };
    s.attach(tw);
    tw.start();

    call_log early_log{};
    s.subscribe(make_logged_subscriber(early_log, "early"));
    s.force_initial_pause();
    ASSERT_EQ(early_log.snapshot().size(), 1u);

    ASSERT_TRUE(wait_with_advance(early_log, clk, 2u, delay));
    EXPECT_TRUE(s.is_active(scope::global));

    call_log late_log{};
    s.subscribe(make_logged_subscriber(late_log, "late"));
    s.force_initial_pause();
    EXPECT_TRUE(late_log.snapshot().empty())
        << "late subscriber should not be paused after delay elapsed";

    ASSERT_TRUE(wait_with_advance(early_log, clk, 3u, dur));
    const auto late_events = late_log.snapshot();
    ASSERT_EQ(late_events.size(), 1u);
    EXPECT_EQ(late_events[0], "late:pause");
}
