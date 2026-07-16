// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/state.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>

namespace
{
using process_state       = rocprofsys::state::process;
using thread_state        = rocprofsys::state::thread;
using process_state_value = process_state::State;
using thread_state_value  = thread_state::State;

class ProcessStateTest : public ::testing::Test
{
protected:
    void SetUp() override { process_state::reset(); }

    void TearDown() override { process_state::reset(); }
};

class ThreadStateTest : public ::testing::Test
{
protected:
    void TearDown() override { thread_state::set(thread_state_value::Enabled); }
};
}  // namespace

TEST_F(ProcessStateTest, default_state_is_pre_init)
{
    EXPECT_EQ(process_state::get(), process_state_value::PreInit);
}

TEST_F(ProcessStateTest, set_advances_state_and_returns_previous)
{
    auto _prior = process_state::set(process_state_value::Init);
    EXPECT_EQ(_prior, process_state_value::PreInit);
    EXPECT_EQ(process_state::get(), process_state_value::Init);
}

TEST_F(ProcessStateTest, set_to_lesser_value_throws_and_state_is_unchanged)
{
    process_state::set(process_state_value::Active);

    EXPECT_THROW(process_state::set(process_state_value::Init), std::runtime_error);
    EXPECT_EQ(process_state::get(), process_state_value::Active);
}

TEST_F(ProcessStateTest, reset_bypasses_validation_and_returns_to_pre_init)
{
    process_state::set(process_state_value::Finalized);

    auto _prior = process_state::reset();
    EXPECT_EQ(_prior, process_state_value::Finalized);
    EXPECT_EQ(process_state::get(), process_state_value::PreInit);
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
    using rocprofsys::mode::process;
    EXPECT_EQ(fmt::format("{}", process::Trace), "Trace");
    EXPECT_EQ(fmt::format("{}", process::Sampling), "Sampling");
    EXPECT_EQ(fmt::format("{}", process::Causal), "Causal");
    EXPECT_EQ(fmt::format("{}", process::Coverage), "Coverage");
}

TEST(StateFormatterTest, process_causal_mode)
{
    using rocprofsys::mode::process_causal;
    EXPECT_EQ(fmt::format("{}", process_causal::Line), "Line");
    EXPECT_EQ(fmt::format("{}", process_causal::Function), "Function");
}
