// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/state.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <stdexcept>
#include <thread>
#include <vector>

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
    void TearDown() override { thread_state::set(thread_state_value::enabled); }
};
}  // namespace

TEST_F(ProcessStateTest, default_state_is_pre_init)
{
    EXPECT_EQ(process_state::get(), process_state_value::pre_init);
}

TEST_F(ProcessStateTest, set_advances_state_and_returns_previous)
{
    auto const _prior = process_state::set(process_state_value::init);
    EXPECT_EQ(_prior, process_state_value::pre_init);
    EXPECT_EQ(process_state::get(), process_state_value::init);
}

TEST_F(ProcessStateTest, set_to_lesser_value_throws_and_state_is_unchanged)
{
    process_state::set(process_state_value::active);

    EXPECT_THROW(process_state::set(process_state_value::init), std::runtime_error);
    EXPECT_EQ(process_state::get(), process_state_value::active);
}

TEST_F(ProcessStateTest, reset_bypasses_validation_and_returns_to_pre_init)
{
    process_state::set(process_state_value::finalized);

    auto const _prior = process_state::reset();
    EXPECT_EQ(_prior, process_state_value::finalized);
    EXPECT_EQ(process_state::get(), process_state_value::pre_init);
}

TEST_F(ProcessStateTest, racing_setters_never_lower_the_state)
{
    constexpr size_t k_iterations = 20'000;
    constexpr size_t k_parties    = 3;

    std::barrier        sync{ k_parties };
    std::atomic<size_t> rejected{ 0 };

    auto setter = [&](process_state_value value) {
        for(size_t i = 0; i < k_iterations; ++i)
        {
            sync.arrive_and_wait();
            try
            {
                process_state::set(value);
            } catch(const std::runtime_error&)
            {
                rejected.fetch_add(1);
            }
            sync.arrive_and_wait();
        }
    };

    std::thread activator(setter, process_state_value::active);
    std::thread finalizer(setter, process_state_value::finalized);

    size_t lowered = 0;
    for(size_t i = 0; i < k_iterations; ++i)
    {
        process_state::reset();
        sync.arrive_and_wait();
        sync.arrive_and_wait();
        if(process_state::get() != process_state_value::finalized)
        {
            ++lowered;
        }
    }
    activator.join();
    finalizer.join();

    EXPECT_EQ(lowered, 0U)
        << "set(finalized) always runs, so the final state must be "
           "finalized; active means set(active) lowered it (rejected sets: "
        << rejected.load() << ")";
}

TEST_F(ThreadStateTest, default_state_is_enabled)
{
    EXPECT_EQ(thread_state::get(), thread_state_value::enabled);
}

TEST_F(ThreadStateTest, set_returns_previous_value)
{
    auto const _prior = thread_state::set(thread_state_value::internal);
    EXPECT_EQ(_prior, thread_state_value::enabled);
    EXPECT_EQ(thread_state::get(), thread_state_value::internal);
}

TEST_F(ThreadStateTest, push_pop_round_trip_restores_prior_state)
{
    thread_state::push(thread_state_value::internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::internal);

    thread_state::pop();
    EXPECT_EQ(thread_state::get(), thread_state_value::enabled);
}

TEST_F(ThreadStateTest, pop_on_empty_history_is_a_no_op)
{
    thread_state::set(thread_state_value::internal);

    auto const _v = thread_state::pop();
    EXPECT_EQ(_v, thread_state_value::internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::internal);
}

TEST_F(ThreadStateTest, push_and_pop_are_no_ops_once_completed)
{
    thread_state::set(thread_state_value::completed);

    auto const _pushed = thread_state::push(thread_state_value::internal);
    EXPECT_EQ(_pushed, thread_state_value::completed);
    EXPECT_EQ(thread_state::get(), thread_state_value::completed);

    auto const _popped = thread_state::pop();
    EXPECT_EQ(_popped, thread_state_value::completed);
}

TEST_F(ThreadStateTest, nesting_beyond_reserved_depth_unwinds_in_lifo_order)
{
    constexpr size_t k_depth = 64;

    std::vector<thread_state_value> observed_on_unwind;

    std::thread worker([&]() {
        std::vector<thread_state_value> expected_on_unwind;
        auto                            current = thread_state::get();
        for(size_t i = 0; i < k_depth; ++i)
        {
            expected_on_unwind.push_back(current);
            current =
                (i % 2 == 0) ? thread_state_value::internal : thread_state_value::enabled;
            thread_state::push(current);
        }
        for(size_t i = 0; i < k_depth; ++i)
        {
            observed_on_unwind.push_back(thread_state::pop());
        }

        std::ranges::reverse(expected_on_unwind);
        EXPECT_EQ(observed_on_unwind, expected_on_unwind);
    });
    worker.join();

    ASSERT_EQ(observed_on_unwind.size(), k_depth);
    EXPECT_EQ(observed_on_unwind.back(), thread_state_value::enabled);
}

TEST_F(ThreadStateTest, scoped_guard_restores_state_on_destruction)
{
    {
        auto const _guard = thread_state::scoped(thread_state_value::internal);
        EXPECT_EQ(thread_state::get(), thread_state_value::internal);
    }
    EXPECT_EQ(thread_state::get(), thread_state_value::enabled);
}

TEST_F(ThreadStateTest, nested_scoped_guards_restore_in_lifo_order)
{
    auto const _outer = thread_state::scoped(thread_state_value::internal);
    EXPECT_EQ(thread_state::get(), thread_state_value::internal);
    {
        auto const _inner = thread_state::scoped(thread_state_value::enabled);
        EXPECT_EQ(thread_state::get(), thread_state_value::enabled);
    }
    EXPECT_EQ(thread_state::get(), thread_state_value::internal);
}

TEST_F(ThreadStateTest, state_is_thread_local)
{
    thread_state::set(thread_state_value::internal);

    auto        _other_state = thread_state_value::internal;
    std::thread _other([&_other_state]() { _other_state = thread_state::get(); });
    _other.join();

    EXPECT_EQ(_other_state, thread_state_value::enabled)
        << "a freshly spawned thread must start at Enabled regardless of the "
           "spawning thread's state";
    EXPECT_EQ(thread_state::get(), thread_state_value::internal)
        << "the spawning thread's state must be unaffected by the child thread";
}

TEST(StateFormatterTest, process_lifecycle_state)
{
    EXPECT_EQ(fmt::format("{}", process_state_value::pre_init), "PreInit");
    EXPECT_EQ(fmt::format("{}", process_state_value::init), "Init");
    EXPECT_EQ(fmt::format("{}", process_state_value::active), "Active");
    EXPECT_EQ(fmt::format("{}", process_state_value::finalized), "Finalized");
}

TEST(StateFormatterTest, thread_lifecycle_state)
{
    EXPECT_EQ(fmt::format("{}", thread_state_value::enabled), "Enabled");
    EXPECT_EQ(fmt::format("{}", thread_state_value::internal), "Internal");
    EXPECT_EQ(fmt::format("{}", thread_state_value::completed), "Completed");
    EXPECT_EQ(fmt::format("{}", thread_state_value::disabled), "Disabled");
}

TEST(StateFormatterTest, process_mode)
{
    using rocprofsys::mode::process;
    EXPECT_EQ(fmt::format("{}", process::trace), "Trace");
    EXPECT_EQ(fmt::format("{}", process::sampling), "Sampling");
    EXPECT_EQ(fmt::format("{}", process::causal), "Causal");
    EXPECT_EQ(fmt::format("{}", process::coverage), "Coverage");
}

TEST(StateFormatterTest, process_causal_mode)
{
    using rocprofsys::mode::process_causal;
    EXPECT_EQ(fmt::format("{}", process_causal::line), "Line");
    EXPECT_EQ(fmt::format("{}", process_causal::function), "Function");
}
