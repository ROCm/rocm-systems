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
    , m_scope{ event_scope }
    {
        m_session.register_trigger(m_name, initial, m_scope);
    }

    ~mock_trigger() { m_session.unregister_trigger(m_name, m_scope); }

    mock_trigger(const mock_trigger&)            = delete;
    mock_trigger& operator=(const mock_trigger&) = delete;
    mock_trigger(mock_trigger&&)                 = delete;
    mock_trigger& operator=(mock_trigger&&)      = delete;

    void set_action(action a) const { m_session.set_action(m_name, a, m_scope); }

private:
    session&    m_session;
    std::string m_name;
    scope       m_scope;
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
    // listens to both global and sampling) must not be resumed just
    // because ONE of its scopes cleared - it must wait for ALL of them.
    const mock_trigger global_trigger{ s, "global_trigger", scope::global,
                                       action::pause };
    const mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling,
                                         action::pause };

    int resume_count = 0;
    int pause_count  = 0;
    s.subscribe({ [&pause_count]() { ++pause_count; },
                  [&resume_count]() { ++resume_count; },
                  "scoped_sub",
                  { scope::global, scope::sampling } });

    // Both scopes paused initially - resuming just the global scope must not
    // fire the subscriber's on_resume, since sampling is still paused.
    global_trigger.set_action(action::trace);
    EXPECT_EQ(resume_count, 0);

    // Once the remaining scope (sampling) also clears, the subscriber
    // should resume exactly once.
    sampling_trigger.set_action(action::trace);
    EXPECT_EQ(resume_count, 1);
}

TEST_F(session_scope_test, subscriber_paused_immediately_when_any_listened_scope_pauses)
{
    const mock_trigger global_trigger{ s, "global_trigger", scope::global,
                                       action::trace };
    const mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling,
                                         action::trace };

    int pause_count = 0;
    s.subscribe({ [&pause_count]() { ++pause_count; },
                  []() {},
                  "scoped_sub",
                  { scope::global, scope::sampling } });

    sampling_trigger.set_action(action::pause);
    EXPECT_EQ(pause_count, 1) << "any listened scope pausing must pause the subscriber";
}

TEST_F(session_scope_test, single_scope_subscriber_unaffected_by_other_scope)
{
    // A subscriber listening only to scope::global (the default) must be
    // unaffected by a scope::sampling trigger's transitions.
    const mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling,
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
    const mock_trigger global_trigger{ s, "global_trigger", scope::global,
                                       action::trace };
    const mock_trigger sampling_trigger{ s, "sampling_trigger", scope::sampling,
                                         action::trace };

    sampling_trigger.set_action(action::pause);

    EXPECT_TRUE(s.is_active(scope::global));
    EXPECT_FALSE(s.is_active(scope::sampling));
}

TEST_F(session_scope_test,
       same_named_triggers_in_different_scopes_do_not_clobber_each_other)
{
    // The global TRACE_DELAY window and the sampling-scoped SAMPLING_DURATION
    // window are both time_window instances, so both report the same trigger
    // name. Storing them under that name alone lets the second registration
    // overwrite the first.
    const mock_trigger global_window{ s, "time_window", scope::global, action::pause };
    const mock_trigger sampling_window{ s, "time_window", scope::sampling,
                                        action::pause };

    EXPECT_FALSE(s.is_active(scope::global));
    EXPECT_FALSE(s.is_active(scope::sampling));

    sampling_window.set_action(action::trace);
    EXPECT_TRUE(s.is_active(scope::sampling));

    // Force a fresh resolve of the global scope without touching
    // global_window's own setter: if its entry had been overwritten, the
    // global scope would now wrongly resolve active.
    {
        const mock_trigger probe{ s, "probe", scope::global, action::trace };
    }
    EXPECT_FALSE(s.is_active(scope::global))
        << "global_window's pause must survive the sampling window's registration";
}

TEST_F(session_scope_test, unregister_removes_only_the_matching_scope)
{
    const mock_trigger sampling_window{ s, "time_window", scope::sampling,
                                        action::pause };
    {
        const mock_trigger global_window{ s, "time_window", scope::global,
                                          action::pause };
        EXPECT_FALSE(s.is_active(scope::global));
    }

    EXPECT_TRUE(s.is_active(scope::global))
        << "the global window's entry should be gone once it unregisters";

    // Force a fresh resolve of sampling: a shared entry erased by the
    // global window's destructor would leave nothing to keep this scope paused.
    {
        const mock_trigger probe{ s, "probe", scope::sampling, action::trace };
    }
    EXPECT_FALSE(s.is_active(scope::sampling))
        << "unregistering the global window must not remove the same-named sampling one";
}
