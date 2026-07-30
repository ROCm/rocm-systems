// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/session.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{
using rocprofsys::control::action;
using rocprofsys::control::scope;
using rocprofsys::control::session;
using rocprofsys::control::subscriber;

// Minimal trigger stub: registers under a fixed name and scope, and exposes
// set_action() so a test can drive a later action change.
class mock_trigger
{
public:
    mock_trigger(session& sess, std::string name, scope event_scope, action initial)
    : m_session{ sess }
    , m_name{ std::move(name) }
    {
        m_session.register_trigger(m_name, initial, event_scope);
    }

    ~mock_trigger() { m_session.unregister_trigger(m_name); }

    mock_trigger(const mock_trigger&)            = delete;
    mock_trigger& operator=(const mock_trigger&) = delete;
    mock_trigger(mock_trigger&&)                 = delete;
    mock_trigger& operator=(mock_trigger&&)      = delete;

    void set_action(action a) const { m_session.set_action(m_name, a); }

private:
    session&    m_session;
    std::string m_name;
};

class session_scope_test : public ::testing::Test
{
protected:
    session s{};
};
}  // namespace

TEST_F(session_scope_test, subscriber_not_resumed_while_a_listened_scope_is_still_paused)
{
    // Reproduces the scenario a scoped subscriber (e.g. "sampling", which
    // listens to both global and sampling_only) must not be resumed just
    // because ONE of its scopes cleared - it must wait for ALL of them.
    mock_trigger global_trigger{ s, "global_trigger", scope::global, action::pause };
    mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling_only,
                                   action::pause };

    int resume_count = 0;
    int pause_count  = 0;
    s.subscribe({ [&pause_count]() { ++pause_count; },
                  [&resume_count]() { ++resume_count; },
                  "scoped_sub",
                  { scope::global, scope::sampling_only } });

    // Both scopes paused initially - resuming just the global scope must not
    // fire the subscriber's on_resume, since sampling_only is still paused.
    global_trigger.set_action(action::trace);
    EXPECT_EQ(resume_count, 0);

    // Once the remaining scope (sampling_only) also clears, the subscriber
    // should resume exactly once.
    sampling_trigger.set_action(action::trace);
    EXPECT_EQ(resume_count, 1);
}

TEST_F(session_scope_test, subscriber_paused_immediately_when_any_listened_scope_pauses)
{
    mock_trigger global_trigger{ s, "global_trigger", scope::global, action::trace };
    mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling_only,
                                   action::trace };

    int pause_count = 0;
    s.subscribe({ [&pause_count]() { ++pause_count; },
                  []() {},
                  "scoped_sub",
                  { scope::global, scope::sampling_only } });

    sampling_trigger.set_action(action::pause);
    EXPECT_EQ(pause_count, 1) << "any listened scope pausing must pause the subscriber";
}

TEST_F(session_scope_test, single_scope_subscriber_unaffected_by_other_scope)
{
    // A subscriber listening only to scope::global (the default) must be
    // unaffected by a scope::sampling_only trigger's transitions.
    mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling_only,
                                   action::trace };

    int pause_count  = 0;
    int resume_count = 0;
    s.subscribe({ [&pause_count]() { ++pause_count; },
                  [&resume_count]() { ++resume_count; }, "global_only_sub" });

    sampling_trigger.set_action(action::pause);
    sampling_trigger.set_action(action::trace);

    EXPECT_EQ(pause_count, 0);
    EXPECT_EQ(resume_count, 0);
}

TEST_F(session_scope_test, is_active_is_tracked_independently_per_scope)
{
    mock_trigger global_trigger{ s, "global_trigger", scope::global, action::trace };
    mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling_only,
                                   action::trace };

    sampling_trigger.set_action(action::pause);

    EXPECT_TRUE(s.is_active(scope::global));
    EXPECT_FALSE(s.is_active(scope::sampling_only));
}
