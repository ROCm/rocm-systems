// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/session.hpp"
#include "core/control/subscriber.hpp"
#include "core/control/trigger.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
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

TEST_F(session_test, force_initial_pause_and_publish_equivalence)
{
    // Path A: attach paused triggers, then force_initial_pause.
    call_log     log_a{};
    session      sess_a{};
    mock_trigger tag{ "trig", scope::global, vote::paused };
    sess_a.subscribe(make_logged_subscriber(log_a, "sub"));
    sess_a.attach(tag);
    sess_a.force_initial_pause();

    // Path B: attach active triggers, then publish paused via publish().
    call_log     log_b{};
    session      sess_b{};
    mock_trigger tbg{ "trig", scope::global, vote::active };
    sess_b.subscribe(make_logged_subscriber(log_b, "sub"));
    sess_b.attach(tbg);
    sess_b.publish(tbg, vote::paused);

    EXPECT_EQ(log_a.snapshot(), log_b.snapshot());
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
