// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/state.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <thread>

namespace
{
using process_state_value = rocprofsys::process_lifecycle_state;
using rocprofsys::thread_state;
using thread_state_value = rocprofsys::thread_state::State;

// basic_process_state<Policy> calls Policy::get_debug_init() as a static call, so
// per the policy-based-DI GMock pattern we use a thin static wrapper that forwards
// to a global GMock instance (GMock mocks are non-copyable and cannot be stored as a
// template value member).
struct gmock_config_policy
{
    MOCK_METHOD(bool, get_debug_init, ());
};

std::unique_ptr<::testing::StrictMock<gmock_config_policy>> g_config_policy_mock;

struct mock_config_policy
{
    static bool get_debug_init() { return g_config_policy_mock->get_debug_init(); }
};

using test_process_state = rocprofsys::basic_process_state<mock_config_policy>;

// Second, independent policy type used only to prove that distinct Policy
// instantiations of basic_process_state do not share storage.
struct gmock_config_policy_b
{
    MOCK_METHOD(bool, get_debug_init, ());
};

std::unique_ptr<::testing::StrictMock<gmock_config_policy_b>> g_config_policy_mock_b;

struct mock_config_policy_b
{
    static bool get_debug_init() { return g_config_policy_mock_b->get_debug_init(); }
};

using test_process_state_b = rocprofsys::basic_process_state<mock_config_policy_b>;

class ProcessStateTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_config_policy_mock =
            std::make_unique<::testing::StrictMock<gmock_config_policy>>();
        EXPECT_CALL(*g_config_policy_mock, get_debug_init())
            .Times(::testing::AnyNumber())
            .WillRepeatedly(::testing::Return(false));

        test_process_state::reset();
    }

    void TearDown() override
    {
        test_process_state::reset();
        g_config_policy_mock.reset();
    }
};

class ThreadStateTest : public ::testing::Test
{
protected:
    void TearDown() override { thread_state::set(thread_state_value::Enabled); }
};
}  // namespace

TEST_F(ProcessStateTest, default_state_is_pre_init)
{
    EXPECT_EQ(test_process_state::get(), process_state_value::PreInit);
}

TEST_F(ProcessStateTest, set_advances_state_and_returns_previous)
{
    auto _prior = test_process_state::set(process_state_value::Init);
    EXPECT_EQ(_prior, process_state_value::PreInit);
    EXPECT_EQ(test_process_state::get(), process_state_value::Init);
}

TEST_F(ProcessStateTest, set_to_lesser_value_throws_and_state_is_unchanged)
{
    test_process_state::set(process_state_value::Active);

    EXPECT_THROW(test_process_state::set(process_state_value::Init), std::runtime_error);
    EXPECT_EQ(test_process_state::get(), process_state_value::Active);
}

TEST_F(ProcessStateTest, set_invokes_policy_get_debug_init)
{
    // Times(AtLeast(1)), not Times(1): once this expectation is set, it becomes the
    // matching expectation for every subsequent get_debug_init() call in this test
    // (GMock resolves overlapping expectations by matching the most-recently-set one
    // first, and keeps using it -- it does not fall back to an older expectation
    // once this one is "used up"). TearDown()'s reset() call triggers one more call
    // after this test body's, so an exact Times(1) would over-saturate.
    EXPECT_CALL(*g_config_policy_mock, get_debug_init())
        .Times(::testing::AtLeast(1))
        .WillRepeatedly(::testing::Return(false));

    test_process_state::set(process_state_value::Active);
}

TEST_F(ProcessStateTest, reset_bypasses_validation_and_returns_to_pre_init)
{
    test_process_state::set(process_state_value::Finalized);

    auto _prior = test_process_state::reset();
    EXPECT_EQ(_prior, process_state_value::Finalized);
    EXPECT_EQ(test_process_state::get(), process_state_value::PreInit);
}

TEST_F(ProcessStateTest, reset_invokes_policy_get_debug_init)
{
    test_process_state::set(process_state_value::Active);

    // See the comment in set_invokes_policy_get_debug_init for why this is
    // AtLeast(1) rather than an exact Times(1): TearDown()'s own reset() call
    // triggers one more matching call after this test body's.
    EXPECT_CALL(*g_config_policy_mock, get_debug_init())
        .Times(::testing::AtLeast(1))
        .WillRepeatedly(::testing::Return(false));

    test_process_state::reset();
}

TEST(ProcessStateStorageTest, distinct_policy_types_have_independent_storage)
{
    g_config_policy_mock = std::make_unique<::testing::StrictMock<gmock_config_policy>>();
    g_config_policy_mock_b =
        std::make_unique<::testing::StrictMock<gmock_config_policy_b>>();
    EXPECT_CALL(*g_config_policy_mock, get_debug_init())
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*g_config_policy_mock_b, get_debug_init())
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Return(false));

    test_process_state::reset();
    test_process_state_b::reset();

    test_process_state::set(process_state_value::Active);

    EXPECT_EQ(test_process_state::get(), process_state_value::Active);
    EXPECT_EQ(test_process_state_b::get(), process_state_value::PreInit)
        << "basic_process_state<PolicyB> must not share storage with "
           "basic_process_state<PolicyA>";

    test_process_state::reset();
    test_process_state_b::reset();
    g_config_policy_mock.reset();
    g_config_policy_mock_b.reset();
}

TEST_F(ThreadStateTest, default_state_is_enabled)
{
    EXPECT_EQ(thread_state::get(), thread_state_value::Enabled);
}

TEST_F(ThreadStateTest, set_returns_previous_value)
{
    auto _prior = thread_state::set(thread_state_value::Internal);
    EXPECT_EQ(_prior, thread_state_value::Enabled);
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal);
}

TEST_F(ThreadStateTest, push_pop_round_trip_restores_prior_state)
{
    thread_state::push(thread_state_value::Internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal);

    thread_state::pop();
    EXPECT_EQ(thread_state::get(), thread_state_value::Enabled);
}

TEST_F(ThreadStateTest, pop_on_empty_history_is_a_no_op)
{
    thread_state::set(thread_state_value::Internal);

    auto _v = thread_state::pop();
    EXPECT_EQ(_v, thread_state_value::Internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal);
}

TEST_F(ThreadStateTest, push_and_pop_are_no_ops_once_completed)
{
    thread_state::set(thread_state_value::Completed);

    auto _pushed = thread_state::push(thread_state_value::Internal);
    EXPECT_EQ(_pushed, thread_state_value::Completed);
    EXPECT_EQ(thread_state::get(), thread_state_value::Completed);

    auto _popped = thread_state::pop();
    EXPECT_EQ(_popped, thread_state_value::Completed);
}

TEST_F(ThreadStateTest, scoped_guard_restores_state_on_destruction)
{
    {
        auto _guard = thread_state::scoped(thread_state_value::Internal);
        EXPECT_EQ(thread_state::get(), thread_state_value::Internal);
    }
    EXPECT_EQ(thread_state::get(), thread_state_value::Enabled);
}

TEST_F(ThreadStateTest, nested_scoped_guards_restore_in_lifo_order)
{
    auto _outer = thread_state::scoped(thread_state_value::Internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal);
    {
        auto _inner = thread_state::scoped(thread_state_value::Enabled);
        EXPECT_EQ(thread_state::get(), thread_state_value::Enabled);
    }
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal);
}

TEST_F(ThreadStateTest, state_is_thread_local)
{
    thread_state::set(thread_state_value::Internal);

    auto        _other_state = thread_state_value::Internal;
    std::thread _other([&_other_state]() { _other_state = thread_state::get(); });
    _other.join();

    EXPECT_EQ(_other_state, thread_state_value::Enabled)
        << "a freshly spawned thread must start at Enabled regardless of the "
           "spawning thread's state";
    EXPECT_EQ(thread_state::get(), thread_state_value::Internal)
        << "the spawning thread's state must be unaffected by the child thread";
}

TEST(StateFormatterTest, process_lifecycle_state)
{
    EXPECT_EQ(fmt::format("{}", process_state_value::PreInit), "PreInit");
    EXPECT_EQ(fmt::format("{}", process_state_value::Init), "Init");
    EXPECT_EQ(fmt::format("{}", process_state_value::Active), "Active");
    EXPECT_EQ(fmt::format("{}", process_state_value::Finalized), "Finalized");
    EXPECT_EQ(fmt::format("{}", process_state_value::Disabled), "Disabled");
}

TEST(StateFormatterTest, thread_lifecycle_state)
{
    EXPECT_EQ(fmt::format("{}", thread_state_value::Enabled), "Enabled");
    EXPECT_EQ(fmt::format("{}", thread_state_value::Internal), "Internal");
    EXPECT_EQ(fmt::format("{}", thread_state_value::Completed), "Completed");
    EXPECT_EQ(fmt::format("{}", thread_state_value::Disabled), "Disabled");
}

TEST(StateFormatterTest, process_mode)
{
    using rocprofsys::process_mode;
    EXPECT_EQ(fmt::format("{}", process_mode::Trace), "Trace");
    EXPECT_EQ(fmt::format("{}", process_mode::Sampling), "Sampling");
    EXPECT_EQ(fmt::format("{}", process_mode::Causal), "Causal");
    EXPECT_EQ(fmt::format("{}", process_mode::Coverage), "Coverage");
}

TEST(StateFormatterTest, process_causal_mode)
{
    using rocprofsys::process_causal_mode;
    EXPECT_EQ(fmt::format("{}", process_causal_mode::Line), "Line");
    EXPECT_EQ(fmt::format("{}", process_causal_mode::Function), "Function");
}
