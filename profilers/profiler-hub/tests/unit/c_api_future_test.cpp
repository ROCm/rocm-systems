// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ph_future.hpp"
#include "profiler-hub/c/profiler_hub.h"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace std::chrono_literals;

struct callback_log
{
    std::atomic<int>                progress_calls{ 0 };
    std::atomic<double>             last_progress{ -1.0 };
    std::atomic<int>                finished_calls{ 0 };
    std::atomic<ph_future_status_t> status{ PH_FUTURE_ERROR };
    std::atomic<ph_result_t>        result{ PH_RESULT_INTERNAL_ERROR };
    std::atomic<ph_future_t>        finished_future{ nullptr };
    std::atomic<ph_future_t>        progress_future{ nullptr };
};

callback_log g_log;

void
on_progress(ph_future_t future, double value)
{
    g_log.progress_future = future;
    g_log.last_progress   = value;
    ++g_log.progress_calls;
}

void
on_finished(ph_future_t future, ph_future_status_t status, ph_result_t result)
{
    g_log.finished_future = future;
    g_log.status          = status;
    g_log.result          = result;
    ++g_log.finished_calls;
}

class c_api_future_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_log.progress_calls  = 0;
        g_log.last_progress   = -1.0;
        g_log.finished_calls  = 0;
        g_log.status          = PH_FUTURE_ERROR;
        g_log.result          = PH_RESULT_INTERNAL_ERROR;
        g_log.finished_future = nullptr;
        g_log.progress_future = nullptr;
    }

    void TearDown() override
    {
        if(m_future != nullptr) std::ignore = ph_future_free(m_future);
    }

    void create(ph_progress_fn progress = on_progress,
                ph_finished_fn finished = on_finished)
    {
        ASSERT_EQ(ph_future_create(progress, finished, &m_future), PH_RESULT_SUCCESS);
    }

    ph_future_t m_future{ nullptr };
};

TEST_F(c_api_future_test, a_created_future_is_a_handle_the_caller_owns)
{
    create();

    EXPECT_NE(m_future, nullptr);
}

TEST_F(c_api_future_test, a_null_out_parameter_is_an_invalid_argument)
{
    EXPECT_EQ(ph_future_create(on_progress, on_finished, nullptr),
              PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, a_future_may_have_no_callbacks)
{
    create(nullptr, nullptr);
    ASSERT_TRUE(m_future->try_attach());

    m_future->report_progress(0.5);
    m_future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);
}

TEST_F(c_api_future_test, a_future_serves_only_one_operation)
{
    create();

    EXPECT_TRUE(m_future->try_attach());
    EXPECT_FALSE(m_future->try_attach());
}

TEST_F(c_api_future_test, wait_and_cancel_need_an_operation)
{
    create();

    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_cancel(m_future), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, the_functions_reject_a_null_future)
{
    ph_result_t result{};

    EXPECT_EQ(ph_future_wait(nullptr), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_cancel(nullptr), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_result(nullptr, &result), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_free(nullptr), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_future_test, finishing_calls_on_finished_once_with_the_outcome)
{
    create();
    ASSERT_TRUE(m_future->try_attach());

    m_future->finish(PH_FUTURE_ERROR, PH_RESULT_INTERNAL_ERROR);
    m_future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);

    EXPECT_EQ(g_log.finished_calls, 1);
    EXPECT_EQ(g_log.status, PH_FUTURE_ERROR);
    EXPECT_EQ(g_log.result, PH_RESULT_INTERNAL_ERROR);
    EXPECT_EQ(g_log.finished_future, m_future);
}

TEST_F(c_api_future_test, the_result_is_available_once_the_operation_ended)
{
    create();
    ASSERT_TRUE(m_future->try_attach());
    ph_result_t result = PH_RESULT_SUCCESS;

    EXPECT_EQ(ph_future_result(m_future, &result), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_result(m_future, nullptr), PH_RESULT_INVALID_ARGUMENT);

    m_future->finish(PH_FUTURE_FINISHED, PH_RESULT_INTERNAL_ERROR);

    ASSERT_EQ(ph_future_result(m_future, &result), PH_RESULT_SUCCESS);
    EXPECT_EQ(result, PH_RESULT_INTERNAL_ERROR);
}

TEST_F(c_api_future_test, progress_is_reported_clamped_until_the_operation_ends)
{
    create();
    ASSERT_TRUE(m_future->try_attach());

    m_future->report_progress(0.25);
    EXPECT_DOUBLE_EQ(g_log.last_progress, 0.25);
    m_future->report_progress(7.0);
    EXPECT_DOUBLE_EQ(g_log.last_progress, 1.0);
    m_future->report_progress(-3.0);
    EXPECT_DOUBLE_EQ(g_log.last_progress, 0.0);
    EXPECT_EQ(g_log.progress_future, m_future);

    m_future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);
    m_future->report_progress(0.5);

    EXPECT_EQ(g_log.progress_calls, 3);
}

TEST_F(c_api_future_test, wait_returns_after_the_finished_callback_has_returned)
{
    create();
    ASSERT_TRUE(m_future->try_attach());

    std::thread worker{ [future = m_future] {
        std::this_thread::sleep_for(20ms);
        future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);
    } };

    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);
    EXPECT_EQ(g_log.finished_calls, 1);
    worker.join();
}

TEST_F(c_api_future_test, cancel_requests_a_stop_the_operation_can_observe)
{
    create();
    ASSERT_TRUE(m_future->try_attach());
    const auto token = m_future->stop_token();
    ASSERT_FALSE(token.stop_requested());

    EXPECT_EQ(ph_future_cancel(m_future), PH_RESULT_SUCCESS);

    EXPECT_TRUE(token.stop_requested());
}

TEST_F(c_api_future_test, cancelling_a_finished_operation_is_harmless)
{
    create();
    ASSERT_TRUE(m_future->try_attach());
    m_future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);

    EXPECT_EQ(ph_future_cancel(m_future), PH_RESULT_SUCCESS);
    EXPECT_EQ(g_log.status, PH_FUTURE_FINISHED);
}

TEST_F(c_api_future_test, freeing_twice_reports_the_second_call)
{
    create();
    const auto keep_alive = m_future->shared_from_this();

    EXPECT_EQ(ph_future_free(m_future), PH_RESULT_SUCCESS);
    EXPECT_EQ(ph_future_free(m_future), PH_RESULT_INVALID_ARGUMENT);

    m_future = nullptr;
}

TEST_F(c_api_future_test, an_operation_keeps_the_future_alive_after_it_was_freed)
{
    create();
    ASSERT_TRUE(m_future->try_attach());
    const auto operation_reference = m_future->shared_from_this();

    ASSERT_EQ(ph_future_free(m_future), PH_RESULT_SUCCESS);
    const auto future = m_future;
    m_future          = nullptr;

    operation_reference->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);

    EXPECT_EQ(g_log.finished_future, future);
    EXPECT_EQ(g_log.finished_calls, 1);
}

TEST_F(c_api_future_test, concurrent_finishes_call_on_finished_exactly_once)
{
    create();
    ASSERT_TRUE(m_future->try_attach());

    std::vector<std::thread> threads;
    for(int i = 0; i < 8; ++i)
    {
        threads.emplace_back([future = m_future] {
            future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(g_log.finished_calls, 1);
}

class c_api_async_ctx_test : public c_api_future_test
{
protected:
    void SetUp() override
    {
        c_api_future_test::SetUp();
        m_db_path = profiler_hub::test::temp_trace_path("c_api_async_ctx_test");
        profiler_hub::test::write_two_thread_tracks(m_db_path);
    }

    void TearDown() override
    {
        if(m_ctx != nullptr) ph_ctx_free(m_ctx);
        c_api_future_test::TearDown();
        std::filesystem::remove(m_db_path);
    }

    std::string m_db_path;
    ph_ctx_t    m_ctx{ nullptr };
};

TEST_F(c_api_async_ctx_test, the_context_is_read_in_the_background_and_reports_its_end)
{
    create();

    ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future), PH_RESULT_SUCCESS);
    ASSERT_NE(m_ctx, nullptr);
    ASSERT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);

    ph_result_t result = PH_RESULT_INTERNAL_ERROR;
    ASSERT_EQ(ph_future_result(m_future, &result), PH_RESULT_SUCCESS);
    EXPECT_EQ(result, PH_RESULT_SUCCESS);
    EXPECT_EQ(g_log.finished_calls, 1);
    EXPECT_EQ(g_log.status, PH_FUTURE_FINISHED);
    EXPECT_DOUBLE_EQ(g_log.last_progress, 1.0);
    EXPECT_EQ(g_log.progress_calls, 2);

    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &tracks, nullptr), PH_RESULT_SUCCESS);
    EXPECT_EQ(tracks.list_size, 2U);
}

TEST_F(c_api_async_ctx_test, a_data_call_waits_for_the_read_to_end)
{
    create();
    ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future), PH_RESULT_SUCCESS);

    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &tracks, nullptr), PH_RESULT_SUCCESS);

    EXPECT_EQ(tracks.list_size, 2U);
    EXPECT_EQ(tracks.tracks[0].nesting_depth, 1U);
}

TEST_F(c_api_async_ctx_test, a_future_that_serves_an_operation_is_rejected)
{
    create();
    ASSERT_TRUE(m_future->try_attach());

    EXPECT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future),
              PH_RESULT_INVALID_ARGUMENT);

    EXPECT_EQ(m_ctx, nullptr);
}

TEST_F(c_api_async_ctx_test, a_missing_trace_fails_at_once_and_leaves_the_future_unused)
{
    create();

    EXPECT_EQ(ph_ctx_create(&m_ctx, (m_db_path + ".missing").c_str(), m_future),
              PH_RESULT_CONTEXT_ALLOCATION_FAILED);

    EXPECT_EQ(m_ctx, nullptr);
    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(g_log.finished_calls, 0);
}

TEST_F(c_api_async_ctx_test,
       cancelling_the_read_ends_it_cancelled_and_the_data_calls_follow)
{
    static std::atomic<bool> release{ false };
    static std::atomic<bool> in_progress{ false };
    release     = false;
    in_progress = false;

    create(
        [](ph_future_t, double) {
            in_progress = true;
            while(!release)
            {
                std::this_thread::sleep_for(1ms);
            }
        },
        on_finished);
    ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future), PH_RESULT_SUCCESS);

    while(!in_progress)
    {
        std::this_thread::sleep_for(1ms);
    }
    ASSERT_EQ(ph_future_cancel(m_future), PH_RESULT_SUCCESS);
    release = true;
    ASSERT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);

    EXPECT_EQ(g_log.status, PH_FUTURE_CANCELLED);
    EXPECT_EQ(g_log.result, PH_RESULT_CANCELLED);
    ph_track_list_t tracks{};
    EXPECT_EQ(ph_get_track_list(m_ctx, &tracks, nullptr), PH_RESULT_CANCELLED);
}

TEST_F(c_api_async_ctx_test, freeing_the_context_ends_the_future_exactly_once)
{
    create();
    ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future), PH_RESULT_SUCCESS);

    ASSERT_EQ(ph_ctx_free(m_ctx), PH_RESULT_SUCCESS);
    m_ctx = nullptr;

    EXPECT_EQ(g_log.finished_calls, 1);
    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);
}

TEST_F(c_api_async_ctx_test, a_future_can_be_freed_while_the_read_runs)
{
    create();
    ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), m_future), PH_RESULT_SUCCESS);

    ASSERT_EQ(ph_future_free(m_future), PH_RESULT_SUCCESS);
    m_future = nullptr;

    ph_track_list_t tracks{};
    EXPECT_EQ(ph_get_track_list(m_ctx, &tracks, nullptr), PH_RESULT_SUCCESS);

    ASSERT_EQ(ph_ctx_free(m_ctx), PH_RESULT_SUCCESS);
    m_ctx = nullptr;
    EXPECT_EQ(g_log.finished_calls, 1);
}

struct worker_block
{
    std::atomic<bool> release{ false };
    std::atomic<int>  blocked{ 0 };
};

worker_block g_block;

void
blocking_finished(ph_future_t, ph_future_status_t, ph_result_t)
{
    ++g_block.blocked;
    while(!g_block.release)
    {
        std::this_thread::sleep_for(1ms);
    }
}

class c_api_async_call_test : public c_api_future_test
{
protected:
    void SetUp() override
    {
        c_api_future_test::SetUp();
        g_block.release = false;
        g_block.blocked = 0;
        m_db_path       = profiler_hub::test::temp_trace_path("c_api_async_call_test");
        profiler_hub::test::write_two_thread_tracks(m_db_path);
        ASSERT_EQ(ph_ctx_create(&m_ctx, m_db_path.c_str(), nullptr), PH_RESULT_SUCCESS);
    }

    void TearDown() override
    {
        g_block.release = true;
        for(auto* blocker : m_blockers)
        {
            std::ignore = ph_future_wait(blocker);
            std::ignore = ph_future_free(blocker);
        }
        if(m_ctx != nullptr) ph_ctx_free(m_ctx);
        c_api_future_test::TearDown();
        std::filesystem::remove(m_db_path);
    }

    // The context pool has max(1, hardware threads / 2) workers; keeping every one of
    // them busy makes the next submitted operation wait in the queue.
    void occupy_every_worker()
    {
        const int workers =
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency() / 2));
        for(int i = 0; i < workers; ++i)
        {
            ph_future_t blocker = nullptr;
            ASSERT_EQ(ph_future_create(nullptr, blocking_finished, &blocker),
                      PH_RESULT_SUCCESS);
            m_blockers.push_back(blocker);
            ASSERT_EQ(ph_get_track_list(m_ctx, &m_blocker_tracks, blocker),
                      PH_RESULT_SUCCESS);
        }
        while(g_block.blocked < workers)
        {
            std::this_thread::sleep_for(1ms);
        }
    }

    std::string              m_db_path;
    ph_ctx_t                 m_ctx{ nullptr };
    ph_track_list_t          m_blocker_tracks{};
    std::vector<ph_future_t> m_blockers;
};

TEST_F(c_api_async_call_test,
       a_call_with_a_future_returns_and_fills_the_out_parameter_later)
{
    create();
    ph_event_list_t events{};

    ASSERT_EQ(ph_get_track_events(m_ctx, 0, 0, 0, &events, m_future), PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);

    EXPECT_EQ(events.list_size, 2U);
    EXPECT_EQ(g_log.finished_calls, 1);
    EXPECT_EQ(g_log.status, PH_FUTURE_FINISHED);
    EXPECT_EQ(g_log.result, PH_RESULT_SUCCESS);
}

TEST_F(c_api_async_call_test, every_data_call_accepts_a_future)
{
    ph_track_list_t  tracks{};
    ph_node_t        node{};
    ph_sample_list_t samples{};
    ph_event_list_t  events{};
    ph_future_t      futures[4] = {};
    for(auto& future : futures)
    {
        ASSERT_EQ(ph_future_create(nullptr, nullptr, &future), PH_RESULT_SUCCESS);
    }

    EXPECT_EQ(ph_get_track_list(m_ctx, &tracks, futures[0]), PH_RESULT_SUCCESS);
    EXPECT_EQ(ph_get_node(m_ctx, &node, futures[1]), PH_RESULT_SUCCESS);
    EXPECT_EQ(ph_get_track_samples(m_ctx, 0, 0, 0, &samples, futures[2]),
              PH_RESULT_SUCCESS);
    EXPECT_EQ(ph_get_track_events(m_ctx, 0, 0, 0, &events, futures[3]),
              PH_RESULT_SUCCESS);

    for(auto* future : futures)
    {
        EXPECT_EQ(ph_future_wait(future), PH_RESULT_SUCCESS);
        EXPECT_EQ(ph_future_free(future), PH_RESULT_SUCCESS);
    }
    EXPECT_EQ(tracks.list_size, 2U);
    EXPECT_EQ(node.track_list.list_size, 2U);
    EXPECT_EQ(events.list_size, 2U);
}

TEST_F(c_api_async_call_test, a_failing_call_ends_the_future_with_its_result)
{
    create();
    ph_event_list_t events{};

    ASSERT_EQ(ph_get_track_events(m_ctx, 999999, 0, 0, &events, m_future),
              PH_RESULT_SUCCESS);
    ASSERT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);

    EXPECT_EQ(g_log.status, PH_FUTURE_ERROR);
    EXPECT_EQ(g_log.result, PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(events.list_size, 0U);
}

TEST_F(c_api_async_call_test, a_future_that_serves_an_operation_is_rejected)
{
    create();
    ASSERT_TRUE(m_future->try_attach());
    ph_track_list_t tracks{};

    EXPECT_EQ(ph_get_track_list(m_ctx, &tracks, m_future), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_async_call_test, invalid_arguments_are_reported_by_the_call_itself)
{
    create();

    EXPECT_EQ(ph_get_track_list(m_ctx, nullptr, m_future), PH_RESULT_INVALID_ARGUMENT);
    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_INVALID_ARGUMENT);
}

TEST_F(c_api_async_call_test, a_call_that_has_not_started_is_skipped_when_cancelled)
{
    occupy_every_worker();
    create();
    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &tracks, m_future), PH_RESULT_SUCCESS);

    ASSERT_EQ(ph_future_cancel(m_future), PH_RESULT_SUCCESS);
    g_block.release = true;
    ASSERT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);

    EXPECT_EQ(g_log.status, PH_FUTURE_CANCELLED);
    EXPECT_EQ(g_log.result, PH_RESULT_CANCELLED);
    EXPECT_EQ(tracks.list_size, 0U);
}

TEST_F(c_api_async_call_test, freeing_the_context_cancels_the_calls_still_waiting)
{
    occupy_every_worker();
    create();
    ph_track_list_t tracks{};
    ASSERT_EQ(ph_get_track_list(m_ctx, &tracks, m_future), PH_RESULT_SUCCESS);

    std::thread releaser{ [] {
        std::this_thread::sleep_for(50ms);
        g_block.release = true;
    } };
    ASSERT_EQ(ph_ctx_free(m_ctx), PH_RESULT_SUCCESS);
    m_ctx = nullptr;
    releaser.join();

    EXPECT_EQ(g_log.finished_calls, 1);
    EXPECT_EQ(g_log.status, PH_FUTURE_CANCELLED);
    EXPECT_EQ(ph_future_wait(m_future), PH_RESULT_SUCCESS);
}

}  // namespace
