// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/state_machine.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace
{
enum class TestState
{
    IDLE,
    INITIALIZING,
    READY,
    RUNNING,
    PAUSED,
    STOPPED
};

struct TestTransition
{
    TestState                             from_state;
    TestState                             to_state;
    std::function<rocprofiler_status_t()> transition_func;
};

class StateMachineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        transition_count = 0;
        error_transition = false;
    }

    std::atomic<int>  transition_count{0};
    std::atomic<bool> error_transition{false};
};

TEST_F(StateMachineTest, BasicTransitions)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE,
                                                TestState::INITIALIZING,
                                                [this]() {
                                                    transition_count++;
                                                    return ROCPROFILER_STATUS_SUCCESS;
                                                }},
                                               {TestState::INITIALIZING,
                                                TestState::READY,
                                                [this]() {
                                                    transition_count++;
                                                    return ROCPROFILER_STATUS_SUCCESS;
                                                }},
                                               {TestState::READY,
                                                TestState::RUNNING,
                                                [this]() {
                                                    transition_count++;
                                                    return ROCPROFILER_STATUS_SUCCESS;
                                                }},
                                               {TestState::RUNNING, TestState::STOPPED, [this]() {
                                                    transition_count++;
                                                    return ROCPROFILER_STATUS_SUCCESS;
                                                }}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);

    auto status = sm.transition_to(TestState::INITIALIZING);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::INITIALIZING);
    EXPECT_EQ(transition_count, 1);

    status = sm.transition_to(TestState::READY);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::READY);
    EXPECT_EQ(transition_count, 2);

    status = sm.transition_to(TestState::RUNNING);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::RUNNING);
    EXPECT_EQ(transition_count, 3);

    status = sm.transition_to(TestState::STOPPED);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::STOPPED);
    EXPECT_EQ(transition_count, 4);
}

TEST_F(StateMachineTest, InvalidTransition)
{
    std::vector<TestTransition> transitions = {
        {TestState::IDLE, TestState::INITIALIZING, nullptr},
        {TestState::INITIALIZING, TestState::READY, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);

    auto status = sm.transition_to(TestState::READY);
    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);
}

TEST_F(StateMachineTest, TransitionFunctionError)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE,
                                                TestState::INITIALIZING,
                                                [this]() {
                                                    error_transition = true;
                                                    return ROCPROFILER_STATUS_ERROR;
                                                }},
                                               {TestState::IDLE, TestState::READY, [this]() {
                                                    transition_count++;
                                                    return ROCPROFILER_STATUS_SUCCESS;
                                                }}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);

    auto status = sm.transition_to(TestState::INITIALIZING);
    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR);
    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);
    EXPECT_TRUE(error_transition);

    status = sm.transition_to(TestState::READY);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::READY);
    EXPECT_EQ(transition_count, 1);
}

TEST_F(StateMachineTest, IsValidTransition)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE, TestState::INITIALIZING, nullptr},
                                               {TestState::INITIALIZING, TestState::READY, nullptr},
                                               {TestState::READY, TestState::RUNNING, nullptr},
                                               {TestState::RUNNING, TestState::PAUSED, nullptr},
                                               {TestState::PAUSED, TestState::RUNNING, nullptr},
                                               {TestState::RUNNING, TestState::STOPPED, nullptr},
                                               {TestState::PAUSED, TestState::STOPPED, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    EXPECT_TRUE(sm.is_valid_transition(TestState::INITIALIZING));
    EXPECT_FALSE(sm.is_valid_transition(TestState::READY));
    EXPECT_FALSE(sm.is_valid_transition(TestState::RUNNING));
    EXPECT_FALSE(sm.is_valid_transition(TestState::STOPPED));

    sm.transition_to(TestState::INITIALIZING);
    EXPECT_TRUE(sm.is_valid_transition(TestState::READY));
    EXPECT_FALSE(sm.is_valid_transition(TestState::RUNNING));

    sm.transition_to(TestState::READY);
    sm.transition_to(TestState::RUNNING);
    EXPECT_TRUE(sm.is_valid_transition(TestState::PAUSED));
    EXPECT_TRUE(sm.is_valid_transition(TestState::STOPPED));
    EXPECT_FALSE(sm.is_valid_transition(TestState::IDLE));

    sm.transition_to(TestState::PAUSED);
    EXPECT_TRUE(sm.is_valid_transition(TestState::RUNNING));
    EXPECT_TRUE(sm.is_valid_transition(TestState::STOPPED));
}

TEST_F(StateMachineTest, ComplexWorkflow)
{
    int init_count   = 0;
    int run_count    = 0;
    int pause_count  = 0;
    int resume_count = 0;
    int stop_count   = 0;

    std::vector<TestTransition> transitions = {
        {TestState::IDLE,
         TestState::INITIALIZING,
         [&init_count]() {
             init_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::INITIALIZING, TestState::READY, []() { return ROCPROFILER_STATUS_SUCCESS; }},
        {TestState::READY,
         TestState::RUNNING,
         [&run_count]() {
             run_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::RUNNING,
         TestState::PAUSED,
         [&pause_count]() {
             pause_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::PAUSED,
         TestState::RUNNING,
         [&resume_count]() {
             resume_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::RUNNING,
         TestState::STOPPED,
         [&stop_count]() {
             stop_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::PAUSED,
         TestState::STOPPED,
         [&stop_count]() {
             stop_count++;
             return ROCPROFILER_STATUS_SUCCESS;
         }},
        {TestState::READY, TestState::IDLE, []() { return ROCPROFILER_STATUS_SUCCESS; }},
        {TestState::STOPPED, TestState::IDLE, []() { return ROCPROFILER_STATUS_SUCCESS; }}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    sm.transition_to(TestState::INITIALIZING);
    EXPECT_EQ(init_count, 1);

    sm.transition_to(TestState::READY);
    sm.transition_to(TestState::RUNNING);
    EXPECT_EQ(run_count, 1);

    sm.transition_to(TestState::PAUSED);
    EXPECT_EQ(pause_count, 1);

    sm.transition_to(TestState::RUNNING);
    EXPECT_EQ(resume_count, 1);

    sm.transition_to(TestState::PAUSED);
    EXPECT_EQ(pause_count, 2);

    sm.transition_to(TestState::STOPPED);
    EXPECT_EQ(stop_count, 1);

    sm.transition_to(TestState::IDLE);
    EXPECT_EQ(sm.get_current_state(), TestState::IDLE);
}

TEST_F(StateMachineTest, NullTransitionFunction)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE, TestState::READY, nullptr},
                                               {TestState::READY, TestState::RUNNING, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    auto status = sm.transition_to(TestState::READY);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::READY);

    status = sm.transition_to(TestState::RUNNING);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(sm.get_current_state(), TestState::RUNNING);
}

TEST_F(StateMachineTest, IntegerStates)
{
    struct IntTransition
    {
        int                                   from_state;
        int                                   to_state;
        std::function<rocprofiler_status_t()> transition_func;
    };

    std::vector<IntTransition> transitions = {{0, 1, []() { return ROCPROFILER_STATUS_SUCCESS; }},
                                              {1, 2, []() { return ROCPROFILER_STATUS_SUCCESS; }},
                                              {2, 3, []() { return ROCPROFILER_STATUS_SUCCESS; }},
                                              {3, 0, []() { return ROCPROFILER_STATUS_SUCCESS; }}};

    rocprofiler::common::state_machine<int, IntTransition> sm(0, transitions);

    EXPECT_EQ(sm.get_current_state(), 0);

    sm.transition_to(1);
    EXPECT_EQ(sm.get_current_state(), 1);

    sm.transition_to(2);
    EXPECT_EQ(sm.get_current_state(), 2);

    sm.transition_to(3);
    EXPECT_EQ(sm.get_current_state(), 3);

    sm.transition_to(0);
    EXPECT_EQ(sm.get_current_state(), 0);
}

TEST_F(StateMachineTest, ThreadSafety)
{
    std::atomic<int> successful_transitions{0};
    std::atomic<int> failed_transitions{0};

    std::vector<TestTransition> transitions = {{TestState::IDLE, TestState::INITIALIZING, nullptr},
                                               {TestState::INITIALIZING, TestState::READY, nullptr},
                                               {TestState::READY, TestState::RUNNING, nullptr},
                                               {TestState::RUNNING, TestState::PAUSED, nullptr},
                                               {TestState::PAUSED, TestState::RUNNING, nullptr},
                                               {TestState::RUNNING, TestState::STOPPED, nullptr},
                                               {TestState::PAUSED, TestState::STOPPED, nullptr},
                                               {TestState::STOPPED, TestState::IDLE, nullptr},
                                               {TestState::READY, TestState::IDLE, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    // First transition to READY state
    sm.transition_to(TestState::INITIALIZING);
    sm.transition_to(TestState::READY);

    std::vector<std::thread> threads;

    // Create threads that try to transition to RUNNING
    for(int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&sm, &successful_transitions, &failed_transitions]() {
            auto status = sm.transition_to(TestState::RUNNING);
            if(status == ROCPROFILER_STATUS_SUCCESS)
                successful_transitions++;
            else
                failed_transitions++;
        });
    }

    for(auto& t : threads)
        t.join();

    // Only one thread should succeed
    EXPECT_EQ(successful_transitions, 1);
    EXPECT_EQ(failed_transitions, 9);
    EXPECT_EQ(sm.get_current_state(), TestState::RUNNING);
}

TEST_F(StateMachineTest, ConcurrentReads)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE, TestState::READY, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    std::atomic<int>         read_count{0};
    std::vector<std::thread> threads;

    // Create multiple threads reading the state concurrently
    for(int i = 0; i < 20; ++i)
    {
        threads.emplace_back([&sm, &read_count]() {
            for(int j = 0; j < 100; ++j)
            {
                auto state = sm.get_current_state();
                if(state == TestState::IDLE || state == TestState::READY) read_count++;
            }
        });
    }

    // Transition state while reads are happening
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sm.transition_to(TestState::READY);

    for(auto& t : threads)
        t.join();

    EXPECT_EQ(read_count, 2000);
    EXPECT_EQ(sm.get_current_state(), TestState::READY);
}

TEST_F(StateMachineTest, ConcurrentValidationAndTransition)
{
    std::vector<TestTransition> transitions = {{TestState::IDLE, TestState::INITIALIZING, nullptr},
                                               {TestState::INITIALIZING, TestState::READY, nullptr},
                                               {TestState::READY, TestState::RUNNING, nullptr},
                                               {TestState::RUNNING, TestState::PAUSED, nullptr},
                                               {TestState::PAUSED, TestState::RUNNING, nullptr}};

    rocprofiler::common::state_machine<TestState, TestTransition> sm(TestState::IDLE, transitions);

    std::atomic<int> validation_count{0};
    std::atomic<int> transition_count{0};

    // Thread that constantly validates transitions
    std::thread validator([&sm, &validation_count]() {
        for(int i = 0; i < 1000; ++i)
        {
            sm.is_valid_transition(TestState::INITIALIZING);
            sm.is_valid_transition(TestState::READY);
            sm.is_valid_transition(TestState::RUNNING);
            sm.is_valid_transition(TestState::PAUSED);
            validation_count += 4;
        }
    });

    // Thread that performs transitions
    std::thread transitioner([&sm, &transition_count]() {
        if(sm.transition_to(TestState::INITIALIZING) == ROCPROFILER_STATUS_SUCCESS)
            transition_count++;
        if(sm.transition_to(TestState::READY) == ROCPROFILER_STATUS_SUCCESS) transition_count++;
        if(sm.transition_to(TestState::RUNNING) == ROCPROFILER_STATUS_SUCCESS) transition_count++;
        if(sm.transition_to(TestState::PAUSED) == ROCPROFILER_STATUS_SUCCESS) transition_count++;
        if(sm.transition_to(TestState::RUNNING) == ROCPROFILER_STATUS_SUCCESS) transition_count++;
    });

    validator.join();
    transitioner.join();

    EXPECT_EQ(validation_count, 4000);
    EXPECT_EQ(transition_count, 5);
    EXPECT_EQ(sm.get_current_state(), TestState::RUNNING);
}

}  // namespace