// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/c/profiler_hub.h"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace profiler_hub;
using namespace std::chrono_literals;

struct task_state
{
    std::atomic<bool> started{ false };
    std::atomic<bool> finished{ false };
    std::atomic<bool> release{ false };
};

void
quick_task(void* data)
{
    static_cast<task_state*>(data)->finished = true;
}

void
blocking_task(void* data)
{
    auto* state    = static_cast<task_state*>(data);
    state->started = true;
    while(!state->release)
    {
        std::this_thread::sleep_for(1ms);
    }
    state->finished = true;
}

void
wait_until(const std::atomic<bool>& flag)
{
    while(!flag)
    {
        std::this_thread::sleep_for(1ms);
    }
}

class c_api_future_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = test::temp_trace_path("c_api_future_test");
        test::write_node_only(m_db_path);
    }

    void TearDown() override
    {
        if(m_ctx != nullptr) ph_ctx_free(m_ctx);
        std::filesystem::remove(m_db_path);
    }

    void create_context()
    {
        ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str()), PH_RESULT_SUCCESS);
    }

    // The context pool has max(1, hardware threads / 2) workers; keeping every one of
    // them busy makes the next submitted task wait in the queue.
    void occupy_all_workers()
    {
        const size_t workers =
            std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
        m_blockers.reserve(workers);
        for(size_t i = 0; i < workers; ++i)
        {
            auto&       state  = *m_blockers.emplace_back(std::make_unique<task_state>());
            ph_future_t future = nullptr;
            ASSERT_EQ(ph_future_get(m_ctx, &future, blocking_task, &state),
                      PH_RESULT_SUCCESS);
            m_blocker_futures.push_back(future);
        }
        for(const auto& state : m_blockers)
        {
            wait_until(state->started);
        }
    }

    void release_blockers()
    {
        for(const auto& state : m_blockers)
        {
            state->release = true;
        }
    }

    [[nodiscard]] bool blockers_finished() const
    {
        return std::all_of(m_blockers.begin(), m_blockers.end(), [](const auto& state) {
            return state->finished.load();
        });
    }

    std::vector<std::unique_ptr<task_state>> m_blockers;
    std::vector<ph_future_t>                 m_blocker_futures;

    std::string m_db_path;
    ph_ctx_t    m_ctx{ nullptr };
};

TEST_F(c_api_future_test, a_submitted_task_runs_and_the_wait_returns)
{
    create_context();
    task_state  state;
    ph_future_t future = nullptr;

    ASSERT_EQ(ph_future_get(m_ctx, &future, quick_task, &state), PH_RESULT_SUCCESS);
    ASSERT_NE(future, nullptr);
    EXPECT_EQ(ph_future_wait(m_ctx, future), PH_RESULT_SUCCESS);

    EXPECT_TRUE(state.finished);
    EXPECT_EQ(ph_future_free(m_ctx, future), PH_RESULT_SUCCESS);
}

TEST_F(c_api_future_test, a_null_task_clears_the_future_and_is_rejected)
{
    create_context();
    ph_future_t future = reinterpret_cast<ph_future_t>(0x1);

    EXPECT_EQ(ph_future_get(m_ctx, &future, nullptr, nullptr),
              PH_RESULT_INVALID_ARGUMENT);

    EXPECT_EQ(future, nullptr);
}

TEST_F(c_api_future_test, the_arguments_of_the_future_functions_are_checked)
{
    create_context();
    task_state  state;
    ph_future_t future = nullptr;

    EXPECT_EQ(ph_future_get(nullptr, &future, quick_task, &state),
              PH_RESULT_INVALID_CONTEXT);
    EXPECT_EQ(ph_future_get(m_ctx, nullptr, quick_task, &state),
              PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_wait(nullptr, future), PH_RESULT_INVALID_CONTEXT);
    EXPECT_EQ(ph_future_wait(m_ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_cancel(m_ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_free(m_ctx, nullptr), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, a_handle_the_context_never_issued_is_rejected)
{
    create_context();
    const auto foreign = reinterpret_cast<ph_future_t>(0x10);

    EXPECT_EQ(ph_future_wait(m_ctx, foreign), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_cancel(m_ctx, foreign), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_free(m_ctx, foreign), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, freeing_a_future_twice_reports_the_second_call)
{
    create_context();
    task_state  state;
    ph_future_t future = nullptr;
    ASSERT_EQ(ph_future_get(m_ctx, &future, quick_task, &state), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_future_wait(m_ctx, future), PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_future_free(m_ctx, future), PH_RESULT_SUCCESS);
    EXPECT_EQ(ph_future_free(m_ctx, future), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, concurrent_frees_of_one_future_succeed_exactly_once)
{
    create_context();
    task_state  state;
    ph_future_t future = nullptr;
    ASSERT_EQ(ph_future_get(m_ctx, &future, quick_task, &state), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_future_wait(m_ctx, future), PH_RESULT_SUCCESS);

    std::atomic<int>         successes{ 0 };
    std::vector<std::thread> threads;
    for(int i = 0; i < 6; ++i)
    {
        threads.emplace_back([&] {
            if(ph_future_free(m_ctx, future) == PH_RESULT_SUCCESS) ++successes;
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(successes, 1);
}

TEST_F(c_api_future_test, cancelling_a_pending_task_prevents_it_from_running)
{
    create_context();
    occupy_all_workers();
    task_state  skipped;
    ph_future_t pending = nullptr;
    ASSERT_EQ(ph_future_get(m_ctx, &pending, quick_task, &skipped), PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_future_cancel(m_ctx, pending), PH_RESULT_SUCCESS);
    release_blockers();
    for(const auto future : m_blocker_futures)
    {
        EXPECT_EQ(ph_future_wait(m_ctx, future), PH_RESULT_SUCCESS);
    }
    EXPECT_EQ(ph_future_wait(m_ctx, pending), PH_RESULT_SUCCESS);

    EXPECT_FALSE(skipped.finished);
}

TEST_F(c_api_future_test,
       freeing_the_context_waits_for_running_tasks_and_drops_pending_ones)
{
    create_context();
    occupy_all_workers();
    task_state  skipped;
    ph_future_t pending = nullptr;
    ASSERT_EQ(ph_future_get(m_ctx, &pending, quick_task, &skipped), PH_RESULT_SUCCESS);

    std::thread releaser{ [&] {
        std::this_thread::sleep_for(100ms);
        release_blockers();
    } };
    EXPECT_EQ(ph_ctx_free(m_ctx), PH_RESULT_SUCCESS);
    m_ctx = nullptr;
    releaser.join();

    EXPECT_TRUE(blockers_finished());
    EXPECT_FALSE(skipped.finished);
}

TEST_F(c_api_future_test, freeing_the_context_with_unfreed_futures_succeeds)
{
    create_context();
    task_state  state;
    ph_future_t future = nullptr;
    ASSERT_EQ(ph_future_get(m_ctx, &future, quick_task, &state), PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_ctx_free(m_ctx), PH_RESULT_SUCCESS);
    m_ctx = nullptr;
}

TEST_F(c_api_future_test,
       a_task_that_uses_the_context_while_it_is_freed_does_not_deadlock)
{
    create_context();
    struct reentrant
    {
        ph_ctx_t          ctx;
        std::atomic<bool> started{ false };
        std::atomic<int>  result{ -1 };
    } state{ m_ctx };

    ph_future_t future = nullptr;
    ASSERT_EQ(ph_future_get(
                  m_ctx,
                  &future,
                  [](void* data) {
                      auto* self    = static_cast<reentrant*>(data);
                      self->started = true;
                      std::this_thread::sleep_for(100ms);
                      ph_future_t inner = nullptr;
                      self->result      = static_cast<int>(
                          ph_future_get(self->ctx, &inner, [](void*) {}, nullptr));
                  },
                  &state),
              PH_RESULT_SUCCESS);
    wait_until(state.started);

    auto freed = std::async(std::launch::async, [&] { return ph_ctx_free(m_ctx); });
    ASSERT_EQ(freed.wait_for(5s), std::future_status::ready);
    m_ctx = nullptr;

    EXPECT_EQ(freed.get(), PH_RESULT_SUCCESS);
    EXPECT_EQ(state.result, static_cast<int>(PH_RESULT_INVALID_CONTEXT));
}

}  // namespace
