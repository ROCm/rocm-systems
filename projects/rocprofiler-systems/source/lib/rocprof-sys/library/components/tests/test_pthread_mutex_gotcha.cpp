// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/pthread_mutex_gotcha.hpp"

#include <algorithm>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

struct GMockPthreadMutexGotcha
{
    MOCK_METHOD(void, configure, (std::string func_name));
    MOCK_METHOD(void, disable, ());
    MOCK_METHOD(void, set_ready, (bool) );
    MOCK_METHOD(std::function<void()>&, get_initializer, ());
};

namespace test_globals
{
std::unique_ptr<GMockPthreadMutexGotcha> g_gotcha_mock;
bool                                     g_settings_enabled = true;
bool                                     g_use_causal       = false;
bool                                     g_trace_locks      = true;
bool                                     g_trace_rwlocks    = true;
bool                                     g_trace_spin_locks = true;
bool                                     g_trace_barriers   = true;
bool                                     g_trace_join       = true;
bool                                     g_inactive_state   = true;
bool                                     g_is_disabled      = false;
int                                      g_callee_calls     = 0;
int                                      g_callee_retval    = 0;
int                                      g_audit_incoming   = 0;
int                                      g_audit_outgoing   = 0;
}  // namespace test_globals

namespace
{
int
mock_mutex_fn(pthread_mutex_t*)
{
    ++test_globals::g_callee_calls;
    return test_globals::g_callee_retval;
}
int
mock_spinlock_fn(pthread_spinlock_t*)
{
    ++test_globals::g_callee_calls;
    return test_globals::g_callee_retval;
}
int
mock_rwlock_fn(pthread_rwlock_t*)
{
    ++test_globals::g_callee_calls;
    return test_globals::g_callee_retval;
}
int
mock_barrier_fn(pthread_barrier_t*)
{
    ++test_globals::g_callee_calls;
    return test_globals::g_callee_retval;
}
int
mock_join_fn(pthread_t, void**)
{
    ++test_globals::g_callee_calls;
    return test_globals::g_callee_retval;
}
}  // namespace

struct MockedPthreadMutexGotcha
{
    template <size_t N, typename Ret, typename... Args>
    static void configure(std::string name)
    {
        test_globals::g_gotcha_mock->configure(std::move(name));
    }
    static void disable() { test_globals::g_gotcha_mock->disable(); }
    static void set_ready(bool v) { test_globals::g_gotcha_mock->set_ready(v); }
    static std::function<void()>& get_initializer()
    {
        return test_globals::g_gotcha_mock->get_initializer();
    }
};

struct MockedPthreadMutexPolicy
{
    struct gotcha_data_t
    {
        std::string tool_id;
    };

    using gotcha_t = MockedPthreadMutexGotcha;

    template <typename... Args>
    static void audit_incoming(std::string_view, Args&&...)
    {
        ++test_globals::g_audit_incoming;
    }

    static void audit_outgoing(std::string_view, int)
    {
        ++test_globals::g_audit_outgoing;
    }
    static bool      settings_enabled() { return test_globals::g_settings_enabled; }
    static bool      get_use_causal() { return test_globals::g_use_causal; }
    static bool      get_trace_locks() { return test_globals::g_trace_locks; }
    static bool      get_trace_rwlocks() { return test_globals::g_trace_rwlocks; }
    static bool      get_trace_spin_locks() { return test_globals::g_trace_spin_locks; }
    static bool      get_trace_barriers() { return test_globals::g_trace_barriers; }
    static bool      get_trace_join() { return test_globals::g_trace_join; }
    static bool      inactive_state() { return test_globals::g_inactive_state; }
    static bool      is_disabled_check() { return test_globals::g_is_disabled; }
    static uintptr_t get_thread_id() { return 0; }
};

using sut = rocprofsys::component::pthread_mutex_gotcha<MockedPthreadMutexPolicy>;

class pthread_mutex_gotcha_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_globals::g_gotcha_mock      = std::make_unique<GMockPthreadMutexGotcha>();
        test_globals::g_settings_enabled = true;
        test_globals::g_use_causal       = false;
        test_globals::g_trace_locks      = true;
        test_globals::g_trace_rwlocks    = true;
        test_globals::g_trace_spin_locks = true;
        test_globals::g_trace_barriers   = true;
        test_globals::g_trace_join       = true;
        test_globals::g_inactive_state   = true;
        test_globals::g_is_disabled      = false;
        test_globals::g_callee_calls     = 0;
        test_globals::g_callee_retval    = 0;
        test_globals::g_audit_incoming   = 0;
        test_globals::g_audit_outgoing   = 0;
    }

    void TearDown() override { test_globals::g_gotcha_mock.reset(); }

    static MockedPthreadMutexPolicy::gotcha_data_t make_data(std::string tool_id)
    {
        return MockedPthreadMutexPolicy::gotcha_data_t{ std::move(tool_id) };
    }
};

TEST_F(pthread_mutex_gotcha_test, test_label_returns_correct_string)
{
    EXPECT_EQ(sut::label(), "pthread_mutex_gotcha");
}

TEST_F(pthread_mutex_gotcha_test, test_gotcha_capacity_is_thirteen)
{
    EXPECT_EQ(sut::gotcha_capacity, 13u);
}

TEST_F(pthread_mutex_gotcha_test, test_shutdown_calls_disable)
{
    EXPECT_CALL(*test_globals::g_gotcha_mock, disable()).Times(1);

    EXPECT_NO_THROW(sut::shutdown());
}

TEST_F(pthread_mutex_gotcha_test, test_configure_sets_initializer)
{
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));

    EXPECT_NO_THROW(sut::configure());
    EXPECT_TRUE(initializer);
}

TEST_F(pthread_mutex_gotcha_test, test_configure_registers_all_thirteen_functions)
{
    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(13)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    ASSERT_EQ(configured_names.size(), 13u);

    const std::set<std::string> expected_names = {
        "pthread_mutex_lock",
        "pthread_mutex_unlock",
        "pthread_mutex_trylock",
        "pthread_rwlock_rdlock",
        "pthread_rwlock_wrlock",
        "pthread_rwlock_tryrdlock",
        "pthread_rwlock_trywrlock",
        "pthread_rwlock_unlock",
        "pthread_barrier_wait",
        "pthread_spin_lock",
        "pthread_spin_trylock",
        "pthread_spin_unlock",
        "pthread_join",
    };

    const std::set<std::string> configured_set(configured_names.begin(),
                                               configured_names.end());
    EXPECT_EQ(configured_set, expected_names);
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_locks_when_disabled)
{
    test_globals::g_trace_locks = false;

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(10)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    const std::set<std::string> lock_names = { "pthread_mutex_lock",
                                               "pthread_mutex_unlock",
                                               "pthread_mutex_trylock" };
    for(const auto& name : lock_names)
    {
        EXPECT_EQ(configured_names.end(),
                  std::find(configured_names.begin(), configured_names.end(), name))
            << "unexpected configuration of: " << name;
    }
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_rwlocks_when_disabled)
{
    test_globals::g_trace_rwlocks = false;

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(8)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    const std::set<std::string> rwlock_names = {
        "pthread_rwlock_rdlock", "pthread_rwlock_wrlock", "pthread_rwlock_tryrdlock",
        "pthread_rwlock_trywrlock", "pthread_rwlock_unlock"
    };
    for(const auto& name : rwlock_names)
    {
        EXPECT_EQ(configured_names.end(),
                  std::find(configured_names.begin(), configured_names.end(), name))
            << "unexpected configuration of: " << name;
    }
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_spinlocks_when_disabled)
{
    test_globals::g_trace_spin_locks = false;

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(10)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    const std::set<std::string> spin_names = { "pthread_spin_lock",
                                               "pthread_spin_trylock",
                                               "pthread_spin_unlock" };
    for(const auto& name : spin_names)
    {
        EXPECT_EQ(configured_names.end(),
                  std::find(configured_names.begin(), configured_names.end(), name))
            << "unexpected configuration of: " << name;
    }
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_barrier_when_disabled)
{
    test_globals::g_trace_barriers = false;

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(12)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_EQ(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(),
                        "pthread_barrier_wait"))
        << "unexpected configuration of: pthread_barrier_wait";
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_join_when_disabled)
{
    test_globals::g_trace_join = false;

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_))
        .Times(12)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_EQ(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "pthread_join"))
        << "unexpected configuration of: pthread_join";
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_all_when_settings_disabled)
{
    test_globals::g_settings_enabled = false;

    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_)).Times(0);

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();
}

TEST_F(pthread_mutex_gotcha_test, test_configure_skips_all_when_causal_enabled)
{
    test_globals::g_use_causal = true;

    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_gotcha_mock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_gotcha_mock, configure(::testing::_)).Times(0);

    sut::configure();
    ASSERT_TRUE(initializer);
    initializer();
}

// ── operator() tests ───────────────────────────────────────────────────────

TEST_F(pthread_mutex_gotcha_test, test_operator_mutex_passthrough_when_inactive)
{
    auto data = make_data("pthread_mutex_lock");
    sut  g(data);

    pthread_mutex_t mtx           = PTHREAD_MUTEX_INITIALIZER;
    test_globals::g_callee_retval = 42;

    EXPECT_EQ(g(mock_mutex_fn, &mtx), 42);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_spinlock_passthrough_when_inactive)
{
    auto data = make_data("pthread_spin_lock");
    sut  g(data);

    pthread_spinlock_t lock{};
    test_globals::g_callee_retval = 7;

    EXPECT_EQ(g(mock_spinlock_fn, &lock), 7);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_rwlock_passthrough_when_inactive)
{
    auto data = make_data("pthread_rwlock_rdlock");
    sut  g(data);

    pthread_rwlock_t rwlock       = PTHREAD_RWLOCK_INITIALIZER;
    test_globals::g_callee_retval = 3;

    EXPECT_EQ(g(mock_rwlock_fn, &rwlock), 3);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_barrier_passthrough_when_inactive)
{
    auto data = make_data("pthread_barrier_wait");
    sut  g(data);

    pthread_barrier_t barrier{};
    test_globals::g_callee_retval = 0;

    EXPECT_EQ(g(mock_barrier_fn, &barrier), 0);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_join_passthrough_when_inactive)
{
    auto data = make_data("pthread_join");
    sut  g(data);

    test_globals::g_callee_retval = 0;

    EXPECT_EQ(g(mock_join_fn, pthread_self(), nullptr), 0);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_calls_callee_when_disabled)
{
    test_globals::g_inactive_state = false;
    test_globals::g_is_disabled    = true;
    test_globals::g_callee_retval  = 99;

    auto data = make_data("pthread_mutex_lock");
    sut  g(data);

    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

    EXPECT_EQ(g(mock_mutex_fn, &mtx), 99);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_EQ(test_globals::g_audit_incoming, 0);
    EXPECT_EQ(test_globals::g_audit_outgoing, 0);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_null_callee_returns_einval_when_disabled)
{
    test_globals::g_inactive_state = false;
    test_globals::g_is_disabled    = true;

    auto data = make_data("pthread_mutex_lock");
    sut  g(data);

    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

    EXPECT_EQ(g(static_cast<int (*)(pthread_mutex_t*)>(nullptr), &mtx), EINVAL);
    EXPECT_EQ(test_globals::g_callee_calls, 0);
}

TEST_F(pthread_mutex_gotcha_test, test_operator_audit_path_calls_callee_and_audits)
{
    test_globals::g_inactive_state = false;
    test_globals::g_is_disabled    = false;
    test_globals::g_callee_retval  = 55;

    auto data = make_data("pthread_mutex_lock");
    sut  g(data);

    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

    EXPECT_EQ(g(mock_mutex_fn, &mtx), 55);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_EQ(test_globals::g_audit_incoming, 1);
    EXPECT_EQ(test_globals::g_audit_outgoing, 1);
}

TEST_F(pthread_mutex_gotcha_test, test_pause_disables_audit_path)
{
    test_globals::g_inactive_state = false;
    test_globals::g_is_disabled    = false;
    test_globals::g_callee_retval  = 7;

    sut::pause();

    auto            data = make_data("pthread_mutex_lock");
    sut             g(data);
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

    EXPECT_EQ(g(mock_mutex_fn, &mtx), 7);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_EQ(test_globals::g_audit_incoming, 0);
    EXPECT_EQ(test_globals::g_audit_outgoing, 0);

    sut::resume();
}

TEST_F(pthread_mutex_gotcha_test, test_resume_reenables_audit_path)
{
    test_globals::g_inactive_state = false;
    test_globals::g_is_disabled    = false;
    test_globals::g_callee_retval  = 3;

    sut::pause();
    sut::resume();

    auto            data = make_data("pthread_mutex_lock");
    sut             g(data);
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

    EXPECT_EQ(g(mock_mutex_fn, &mtx), 3);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_EQ(test_globals::g_audit_incoming, 1);
    EXPECT_EQ(test_globals::g_audit_outgoing, 1);
}
