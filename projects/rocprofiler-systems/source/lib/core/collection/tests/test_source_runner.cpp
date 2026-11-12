// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/clocks.hpp"
#include "core/collection/detail/source_runner.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/mock_source.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "core/collection/tests/wait_helpers.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::collection::detail
{
namespace
{
using ::testing::_;
using testing::log_capture;
using testing::mock_source;
using testing::recording_sink;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

constexpr std::string_view k_name        = "mock";
constexpr std::uint64_t    k_tick_ns     = 1000;
constexpr std::uint64_t    k_deadline_ns = 2000;
constexpr auto             k_interval    = std::chrono::milliseconds{ 1 };

class source_runner_test_interface : public ::testing::Test
{
protected:
    source_runner_test_interface()
    {
        EXPECT_CALL(*m_source, name()).WillRepeatedly(Return(k_name));
    }

    /** Builds the runner; options must be set before. */
    [[nodiscard]] source_runner& runner()
    {
        if(!m_runner)
        {
            m_runner = std::make_unique<source_runner>(std::move(m_owned), k_interval,
                                                       m_context, m_sink);
        }
        return *m_runner;
    }

    fork_gate      m_gate;
    run_context    m_context{ .gate = m_gate, .options = {}, .paused = false };
    recording_sink m_sink;
    std::unique_ptr<StrictMock<mock_source>> m_owned =
        std::make_unique<StrictMock<mock_source>>();
    StrictMock<mock_source>*       m_source = m_owned.get();
    std::unique_ptr<source_runner> m_runner;
};
}  // namespace

TEST_F(source_runner_test_interface, tick_polls_with_the_time_read_after_the_checks)
{
    std::uint64_t polled_at = 0;
    EXPECT_CALL(*m_source, poll(_, _))
        .WillOnce([&polled_at](std::uint64_t timeline_ns, sink_interface&) {
            polled_at = timeline_ns;
        });
    const auto before = clocks::timeline::now_ns();

    EXPECT_TRUE(runner().tick(k_tick_ns));

    EXPECT_GE(polled_at, before);
    EXPECT_LE(polled_at, clocks::timeline::now_ns());
}

TEST_F(source_runner_test_interface, tick_at_or_after_the_deadline_ends_without_polling)
{
    m_context.options.deadline_ns = k_deadline_ns;

    EXPECT_FALSE(runner().tick(k_deadline_ns));
    EXPECT_FALSE(runner().tick(k_deadline_ns + 1));
}

TEST_F(source_runner_test_interface, tick_before_the_deadline_polls)
{
    m_context.options.deadline_ns = k_deadline_ns;
    EXPECT_CALL(*m_source, poll(_, _));

    EXPECT_TRUE(runner().tick(k_deadline_ns - 1));
}

TEST_F(source_runner_test_interface, tick_while_paused_skips_the_poll)
{
    m_context.paused = true;

    EXPECT_TRUE(runner().tick(k_tick_ns));
}

TEST_F(source_runner_test_interface, tick_while_not_allowed_skips_the_poll)
{
    m_context.options.allowed = [] { return false; };

    EXPECT_TRUE(runner().tick(k_tick_ns));
}

TEST_F(source_runner_test_interface, tick_while_allowed_polls)
{
    m_context.options.allowed = [] { return true; };
    EXPECT_CALL(*m_source, poll(_, _));

    EXPECT_TRUE(runner().tick(k_tick_ns));
}

TEST_F(source_runner_test_interface, tick_during_a_fork_skips_the_poll)
{
    m_gate.block();
    const bool keep_going = runner().tick(k_tick_ns);
    m_gate.unblock();

    EXPECT_TRUE(keep_going);
}

TEST_F(source_runner_test_interface, throwing_allowed_check_is_reported_as_such)
{
    m_context.options.allowed = []() -> bool { throw std::runtime_error("state lost"); };

    try
    {
        static_cast<void>(runner().tick(k_tick_ns));
        ADD_FAILURE() << "tick() did not throw";
    } catch(const std::runtime_error& error)
    {
        EXPECT_THAT(error.what(), ::testing::HasSubstr("allowed() failed: state lost"));
    }
}

TEST_F(source_runner_test_interface, tick_does_not_run_thread_init)
{
    int init_calls                = 0;
    m_context.options.thread_init = [&init_calls](std::string_view) { ++init_calls; };
    EXPECT_CALL(*m_source, poll(_, _));

    EXPECT_TRUE(runner().tick(k_tick_ns));

    EXPECT_EQ(init_calls, 0);
}

TEST_F(source_runner_test_interface, thread_init_runs_once_at_thread_start_while_paused)
{
    std::promise<std::string> initialized;
    auto                      initialized_future = initialized.get_future();
    m_context.options.thread_init = [&initialized](std::string_view source) {
        initialized.set_value(std::string{ source });
    };
    m_context.paused = true;
    EXPECT_CALL(*m_source, release());

    runner().start();

    ASSERT_EQ(initialized_future.wait_for(testing::k_timeout), std::future_status::ready);
    EXPECT_EQ(initialized_future.get(), k_name);
    runner().stop();
}

TEST_F(source_runner_test_interface, thread_init_waits_for_a_fork_in_progress)
{
    std::atomic<bool>  unblocked{ false };
    std::promise<bool> initialized;
    auto               initialized_future = initialized.get_future();
    m_context.options.thread_init         = [&](std::string_view) {
        initialized.set_value(unblocked.load());
    };
    m_context.paused = true;
    EXPECT_CALL(*m_source, release());

    m_gate.block();
    runner().start();
    testing::let_other_threads_run();
    unblocked = true;
    m_gate.unblock();

    ASSERT_EQ(initialized_future.wait_for(testing::k_timeout), std::future_status::ready);
    EXPECT_TRUE(initialized_future.get());
    runner().stop();
}

TEST_F(source_runner_test_interface, throwing_thread_init_ends_polling_and_is_logged)
{
    const log_capture logs;
    m_context.options.thread_init = [](std::string_view) {
        throw std::runtime_error("no name");
    };
    EXPECT_CALL(*m_source, release());

    runner().start();
    runner().stop();

    EXPECT_EQ(logs.count("error", "thread_init() failed: no name"), 1U);
}

TEST_F(source_runner_test_interface, poll_exception_propagates_out_of_tick)
{
    EXPECT_CALL(*m_source, poll(_, _)).WillOnce(Throw(std::runtime_error("read failed")));

    EXPECT_THROW(static_cast<void>(runner().tick(k_tick_ns)), std::runtime_error);
}

TEST_F(source_runner_test_interface, poll_exception_releases_the_fork_pass)
{
    EXPECT_CALL(*m_source, poll(_, _)).WillOnce(Throw(std::runtime_error("read failed")));
    EXPECT_THROW(static_cast<void>(runner().tick(k_tick_ns)), std::runtime_error);

    // block() returns only when no pass is held
    m_gate.block();
    m_gate.unblock();
}

TEST_F(source_runner_test_interface, poll_writes_to_the_collector_sink)
{
    EXPECT_CALL(*m_source, poll(_, ::testing::Ref(m_sink)));

    EXPECT_TRUE(runner().tick(k_tick_ns));
}

TEST_F(source_runner_test_interface, stop_without_start_releases_the_source_once)
{
    EXPECT_CALL(*m_source, release()).Times(1);

    runner().stop();
}

TEST_F(source_runner_test_interface, stop_from_its_own_poll_skips_release)
{
    const log_capture  logs;
    std::promise<void> stopped;
    const auto         stopped_future = stopped.get_future();
    EXPECT_CALL(*m_source, poll(_, _))
        .WillOnce([this, &stopped](std::uint64_t, sink_interface&) {
            runner().stop();
            stopped.set_value();
        });

    runner().start();
    ASSERT_EQ(stopped_future.wait_for(testing::k_timeout), std::future_status::ready);
    // the detached thread is done with the fixture once its fork pass is released
    m_gate.block();
    m_gate.unblock();

    EXPECT_EQ(logs.count("warning", "release skipped"), 1U);
    // the detached thread may still be leaving the scheduler: leak the runner
    ::testing::Mock::AllowLeak(m_source);
    static_cast<void>(m_runner.release());
}

TEST_F(source_runner_test_interface, stop_from_another_thread_after_a_poll_releases)
{
    EXPECT_CALL(*m_source, poll(_, _));
    std::jthread{ [this] { EXPECT_TRUE(runner().tick(k_tick_ns)); } }.join();
    EXPECT_CALL(*m_source, release()).Times(1);

    runner().stop();
}

TEST_F(source_runner_test_interface, start_with_non_positive_interval_throws)
{
    m_runner = std::make_unique<source_runner>(
        std::move(m_owned), std::chrono::nanoseconds::zero(), m_context, m_sink);

    EXPECT_THROW(m_runner->start(), std::invalid_argument);
}
}  // namespace rocprofsys::collection::detail
