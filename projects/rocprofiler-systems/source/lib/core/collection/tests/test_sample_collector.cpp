// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/detail/source_runner.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sample_collector.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/mock_source.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "core/collection/tests/wait_helpers.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <latch>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::collection
{
namespace
{
using ::testing::_;
using testing::call_counter;
using testing::failing_thread_creation;
using testing::k_timeout;
using testing::latch_releaser;
using testing::let_other_threads_run;
using testing::log_capture;
using testing::mock_source;
using testing::recording_sink;
using ::testing::Ref;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;
using testing::wait_for_fork_in_progress;

constexpr auto k_fast_interval =
    std::chrono::nanoseconds{ std::chrono::milliseconds{ 1 } };
constexpr auto k_slow_interval = std::chrono::nanoseconds{ std::chrono::hours{ 1 } };
constexpr int  k_wanted_polls  = 5;

class sample_collector_test_interface : public ::testing::Test
{
protected:
    /** Stops the poll threads while the test locals their mock actions use still live
     * (an ASSERT returns before the end of the test). */
    void TearDown() override { m_collector.stop(); }

    /** Adds a strict mock source; the collector owns it. */
    StrictMock<mock_source>& add_source(
        std::string_view name, std::chrono::nanoseconds interval = k_fast_interval)
    {
        auto  owned  = std::make_unique<StrictMock<mock_source>>();
        auto& source = *owned;
        EXPECT_CALL(source, name()).WillRepeatedly(Return(name));
        m_collector.add_source(std::move(owned), interval);
        return source;
    }

    /** describe() succeeds and every poll is counted. */
    static void expect_running(StrictMock<mock_source>& source, call_counter& polls)
    {
        EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
        EXPECT_CALL(source, poll(_, _)).WillRepeatedly([&polls] { polls(); });
    }

    /** Polls before `resumed` is set are counted in `early_polls`. */
    static void expect_polls_after_resume(StrictMock<mock_source>& source,
                                          const std::atomic<bool>& resumed,
                                          std::atomic<int>&        early_polls,
                                          call_counter&            polls)
    {
        EXPECT_CALL(source, poll(_, _)).WillRepeatedly([&resumed, &early_polls, &polls] {
            if(!resumed.load())
            {
                ++early_polls;
            }
            polls();
        });
    }

    recording_sink   m_sink;
    fork_gate        m_gate;
    sample_collector m_collector{ m_sink, m_gate, detail::run_options{} };
};

void
throw_non_std_exception()
{
    std::rethrow_exception(std::make_exception_ptr(k_wanted_polls));
}
}  // namespace

TEST_F(sample_collector_test_interface, describe_gets_the_collector_sink)
{
    auto& source = add_source("source");
    EXPECT_CALL(source, describe(Ref(m_sink))).WillOnce(Return(false));

    m_collector.start();
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, add_source_without_a_source_throws)
{
    EXPECT_THROW(m_collector.add_source(nullptr, k_fast_interval), std::invalid_argument);
}

TEST_F(sample_collector_test_interface, fork_waits_for_a_running_describe)
{
    std::latch        in_describe{ 1 };
    std::latch        finish_describe{ 1 };
    std::latch        may_unblock{ 1 };
    std::atomic<bool> described{ false };
    std::atomic<bool> described_before_block_returned{ false };
    auto&             source = add_source("source");
    EXPECT_CALL(source, describe(_)).WillOnce([&](sink_interface&) {
        in_describe.count_down();
        finish_describe.wait();
        described = true;
        return false;
    });

    std::jthread starter{ [this] { m_collector.start(); } };
    in_describe.wait();
    std::jthread forker{ [&] {
        m_gate.block();
        described_before_block_returned = described.load();
        may_unblock.wait();
        m_gate.unblock();
    } };
    EXPECT_TRUE(wait_for_fork_in_progress(m_gate)) << "block() never counted the fork";
    finish_describe.count_down();
    may_unblock.count_down();
    forker.join();
    starter.join();

    EXPECT_TRUE(described_before_block_returned);
}

TEST_F(sample_collector_test_interface, fork_inside_describe_does_not_wait_for_itself)
{
    auto& source = add_source("source");
    EXPECT_CALL(source, describe(_)).WillOnce([this](sink_interface&) {
        m_gate.block();
        m_gate.unblock();
        return false;
    });

    m_collector.start();
}

TEST_F(sample_collector_test_interface,
       source_without_columns_is_never_polled_nor_released)
{
    auto& source = add_source("empty");
    EXPECT_CALL(source, describe(_)).WillOnce(Return(false));

    m_collector.start();
    let_other_threads_run();
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, throwing_describe_disables_only_that_source)
{
    call_counter polls{ k_wanted_polls };
    auto&        failing = add_source("failing");
    auto&        healthy = add_source("healthy");
    EXPECT_CALL(failing, describe(_)).WillOnce(Throw(std::runtime_error("no devices")));
    expect_running(healthy, polls);
    EXPECT_CALL(failing, release()).Times(1);
    EXPECT_CALL(healthy, release()).Times(1);

    m_collector.start();
    EXPECT_TRUE(polls.wait());
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, non_std_exception_in_describe_disables_the_source)
{
    auto& failing = add_source("failing");
    EXPECT_CALL(failing, describe(_)).WillOnce([](sink_interface&) -> bool {
        throw_non_std_exception();
        return true;
    });
    EXPECT_CALL(failing, release()).Times(1);

    m_collector.start();
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, add_source_after_stop_throws)
{
    m_collector.stop();

    EXPECT_THROW(m_collector.add_source(std::make_unique<StrictMock<mock_source>>(),
                                        k_fast_interval),
                 std::logic_error);
}

TEST_F(sample_collector_test_interface, throwing_poll_stops_only_that_source)
{
    call_counter failing_polls{ 1 };
    call_counter healthy_polls{ k_wanted_polls };
    auto&        failing = add_source("failing");
    auto&        healthy = add_source("healthy");
    EXPECT_CALL(failing, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(failing, poll(_, _)).WillOnce([&failing_polls] {
        failing_polls();
        throw std::runtime_error("read failed");
    });
    expect_running(healthy, healthy_polls);
    EXPECT_CALL(failing, release()).Times(1);
    EXPECT_CALL(healthy, release()).Times(1);

    m_collector.start();
    ASSERT_TRUE(failing_polls.wait());
    EXPECT_TRUE(healthy_polls.wait());
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, source_whose_thread_cannot_start_is_disabled)
{
    const log_capture logs;
    auto&             source = add_source("source");
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(source, release()).Times(1);
    {
        const failing_thread_creation no_threads;
        m_collector.start();
    }
    m_collector.stop();

    EXPECT_EQ(logs.count("error", "source 'source' start failed"), 1U);
}

TEST_F(sample_collector_test_interface, add_source_with_a_non_positive_interval_throws)
{
    EXPECT_THROW(m_collector.add_source(std::make_unique<StrictMock<mock_source>>(),
                                        std::chrono::nanoseconds::zero()),
                 std::invalid_argument);
}

TEST_F(sample_collector_test_interface, stop_twice_releases_once)
{
    call_counter polls{ 1 };
    auto&        source = add_source("source");
    expect_running(source, polls);
    EXPECT_CALL(source, release()).Times(1);
    m_collector.start();
    ASSERT_TRUE(polls.wait());

    m_collector.stop();
    m_collector.stop();
}

TEST_F(sample_collector_test_interface,
       throwing_release_is_called_once_and_stop_does_not_throw)
{
    call_counter polls{ 1 };
    auto&        source = add_source("source");
    expect_running(source, polls);
    EXPECT_CALL(source, release()).WillOnce(Throw(std::runtime_error("shutdown failed")));
    m_collector.start();
    ASSERT_TRUE(polls.wait());

    EXPECT_NO_THROW(m_collector.stop());
    EXPECT_NO_THROW(m_collector.stop());
}

TEST_F(sample_collector_test_interface, stop_before_start_neither_describes_nor_releases)
{
    static_cast<void>(add_source("source"));

    m_collector.stop();
}

TEST_F(sample_collector_test_interface,
       start_only_describes_sources_added_since_the_last_start)
{
    call_counter first_polls{ 1 };
    call_counter second_polls{ 1 };
    auto&        first = add_source("first", k_slow_interval);
    expect_running(first, first_polls);
    m_collector.start();
    auto& second = add_source("second", k_slow_interval);
    expect_running(second, second_polls);
    EXPECT_CALL(first, release()).Times(1);
    EXPECT_CALL(second, release()).Times(1);

    m_collector.start();

    EXPECT_TRUE(first_polls.wait());
    EXPECT_TRUE(second_polls.wait());
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, sources_are_polled_at_their_own_interval)
{
    call_counter fast_polls{ k_wanted_polls };
    call_counter slow_polls{ 1 };
    auto&        fast = add_source("fast", k_fast_interval);
    auto&        slow = add_source("slow", k_slow_interval);
    expect_running(fast, fast_polls);
    expect_running(slow, slow_polls);
    EXPECT_CALL(fast, release());
    EXPECT_CALL(slow, release());

    m_collector.start();
    ASSERT_TRUE(fast_polls.wait());
    ASSERT_TRUE(slow_polls.wait());
    m_collector.stop();

    EXPECT_EQ(slow_polls.count(), 1);
}

TEST_F(sample_collector_test_interface, pause_and_resume_reach_running_sources)
{
    call_counter      before{ k_wanted_polls };
    call_counter      after{ k_wanted_polls };
    std::atomic<bool> resumed{ false };
    auto&             source = add_source("source");
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(source, poll(_, _)).WillRepeatedly([&] {
        (resumed.load() ? after : before)();
    });
    EXPECT_CALL(source, release());
    m_collector.start();
    ASSERT_TRUE(before.wait());

    m_collector.pause();
    // a poll may already be running when pause() returns
    const int at_pause = before.count() + 1;
    let_other_threads_run();
    EXPECT_LE(before.count(), at_pause);

    resumed = true;
    m_collector.resume();
    EXPECT_TRUE(after.wait());
    m_collector.stop();
}

TEST_F(sample_collector_test_interface, source_added_while_paused_starts_paused)
{
    call_counter      polls{ 1 };
    std::atomic<bool> resumed{ false };
    std::atomic<int>  early_polls{ 0 };
    m_collector.pause();
    auto& source = add_source("late");
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    expect_polls_after_resume(source, resumed, early_polls, polls);
    EXPECT_CALL(source, release());

    m_collector.start();
    let_other_threads_run();
    resumed = true;
    m_collector.resume();

    EXPECT_TRUE(polls.wait());
    m_collector.stop();
    EXPECT_EQ(early_polls.load(), 0);
}

TEST_F(sample_collector_test_interface,
       pause_during_describe_returns_and_the_source_starts_paused)
{
    call_counter       polls{ 1 };
    std::atomic<bool>  resumed{ false };
    std::atomic<int>   early_polls{ 0 };
    std::promise<void> describing;
    std::latch         finish_describe{ 1 };
    auto&              source = add_source("slow_init");
    EXPECT_CALL(source, describe(_)).WillOnce([&](sink_interface&) {
        describing.set_value();
        finish_describe.wait();
        return true;
    });
    expect_polls_after_resume(source, resumed, early_polls, polls);
    EXPECT_CALL(source, release());

    std::jthread starter{ [this] { m_collector.start(); } };
    {
        const latch_releaser release_describe{ finish_describe };
        ASSERT_EQ(describing.get_future().wait_for(k_timeout), std::future_status::ready);
        m_collector.pause();
    }
    starter.join();
    let_other_threads_run();
    resumed = true;
    m_collector.resume();

    EXPECT_TRUE(polls.wait());
    m_collector.stop();
    EXPECT_EQ(early_polls.load(), 0);
}

TEST_F(sample_collector_test_interface, gap_is_written_once_per_pause_at_resume)
{
    m_collector.pause();
    m_collector.pause();
    EXPECT_TRUE(m_sink.gaps().empty());
    m_collector.resume();
    m_collector.resume();

    const auto gaps = m_sink.gaps();
    ASSERT_EQ(gaps.size(), 1U);
    EXPECT_LE(gaps[0].begin_ns, gaps[0].end_ns);
}

TEST_F(sample_collector_test_interface, each_pause_writes_its_own_gap)
{
    m_collector.pause();
    m_collector.resume();
    m_collector.pause();
    m_collector.resume();

    const auto gaps = m_sink.gaps();
    ASSERT_EQ(gaps.size(), 2U);
    EXPECT_LE(gaps[0].end_ns, gaps[1].begin_ns);
    EXPECT_LE(gaps[1].begin_ns, gaps[1].end_ns);
}

TEST_F(sample_collector_test_interface, stop_while_paused_writes_the_gap_once)
{
    m_collector.pause();

    m_collector.stop();
    m_collector.stop();

    const auto gaps = m_sink.gaps();
    ASSERT_EQ(gaps.size(), 1U);
    EXPECT_LE(gaps[0].begin_ns, gaps[0].end_ns);
}

TEST_F(sample_collector_test_interface, resume_without_pause_writes_no_gap)
{
    m_collector.resume();
    m_collector.stop();

    EXPECT_TRUE(m_sink.gaps().empty());
}

TEST(sample_collector, destructor_stops_and_releases_running_sources)
{
    recording_sink sink;
    fork_gate      gate;
    call_counter   polls{ 1 };
    auto           owned  = std::make_unique<StrictMock<mock_source>>();
    auto&          source = *owned;
    EXPECT_CALL(source, name()).WillRepeatedly(Return("source"));
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(source, poll(_, _)).WillRepeatedly([&polls] { polls(); });
    EXPECT_CALL(source, release()).Times(1);
    {
        sample_collector collector{ sink, gate, detail::run_options{} };
        collector.add_source(std::move(owned), k_fast_interval);
        collector.start();
        ASSERT_TRUE(polls.wait());
    }
}

TEST(sample_collector, past_the_deadline_sources_are_not_polled)
{
    recording_sink sink;
    fork_gate      gate;
    auto           owned  = std::make_unique<StrictMock<mock_source>>();
    auto&          source = *owned;
    EXPECT_CALL(source, name()).WillRepeatedly(Return("source"));
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(source, release()).Times(1);
    detail::run_options options;
    options.deadline_ns = 0;
    sample_collector collector{ sink, gate, std::move(options) };
    collector.add_source(std::move(owned), k_fast_interval);

    collector.start();
    let_other_threads_run();
    collector.stop();
}

TEST(sample_collector, thread_init_gets_the_source_name_on_the_poll_thread)
{
    constexpr std::string_view k_name = "named";
    recording_sink             sink;
    fork_gate                  gate;
    call_counter               polls{ 1 };
    std::promise<std::string>  initialized;
    auto                       initialized_future = initialized.get_future();
    std::atomic<bool>          on_poll_thread{ false };
    auto                       owned  = std::make_unique<StrictMock<mock_source>>();
    auto&                      source = *owned;
    EXPECT_CALL(source, name()).WillRepeatedly(Return(k_name));
    EXPECT_CALL(source, describe(_)).WillOnce(Return(true));
    EXPECT_CALL(source, poll(_, _)).WillRepeatedly([&polls] { polls(); });
    EXPECT_CALL(source, release()).Times(1);
    const auto          test_thread = std::this_thread::get_id();
    detail::run_options options;
    options.thread_init = [&](std::string_view name) {
        on_poll_thread = std::this_thread::get_id() != test_thread;
        initialized.set_value(std::string{ name });
    };
    sample_collector collector{ sink, gate, std::move(options) };
    collector.add_source(std::move(owned), k_fast_interval);

    collector.start();
    ASSERT_TRUE(polls.wait());
    collector.stop();

    ASSERT_EQ(initialized_future.wait_for(k_timeout), std::future_status::ready);
    EXPECT_EQ(initialized_future.get(), k_name);
    EXPECT_TRUE(on_poll_thread.load());
}
}  // namespace rocprofsys::collection
