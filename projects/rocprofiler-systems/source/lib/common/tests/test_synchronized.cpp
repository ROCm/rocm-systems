// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/synchronized.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

using namespace rocprofsys::common;

namespace
{
struct mock_scoped_guard_t
{};

struct gmock_thread_state_policy_t
{
    MOCK_METHOD(mock_scoped_guard_t, scoped, (int) );
};

std::unique_ptr<::testing::NiceMock<gmock_thread_state_policy_t>>
    g_mock_thread_state_policy;

struct mock_thread_state_policy_t
{
    using State = int;

    static constexpr State Internal = 1;

    static mock_scoped_guard_t scoped(State state_to_set)
    {
        return g_mock_thread_state_policy->scoped(state_to_set);
    }
};

template <typename T, bool IsMappedTypeV = false>
using traced_synchronized = synchronized<T, IsMappedTypeV, mock_thread_state_policy_t>;

// The gmock policy above can only observe that scoped() was *called*. The
// invariant that actually prevents the self-deadlock is that the guard is
// already live when the lock is acquired, which requires observing the guard's
// lifetime. mock_scoped_guard_t is empty, so it cannot; this policy can.
std::atomic<int> g_guard_depth{ 0 };

struct raii_scoped_guard_t
{
    raii_scoped_guard_t() { g_guard_depth.fetch_add(1); }
    ~raii_scoped_guard_t() { g_guard_depth.fetch_sub(1); }

    raii_scoped_guard_t(const raii_scoped_guard_t&)            = delete;
    raii_scoped_guard_t& operator=(const raii_scoped_guard_t&) = delete;
    raii_scoped_guard_t(raii_scoped_guard_t&&)                 = delete;
    raii_scoped_guard_t& operator=(raii_scoped_guard_t&&)      = delete;
};

struct raii_thread_state_policy_t
{
    using State = int;

    static constexpr State Internal = 1;

    static raii_scoped_guard_t scoped(State) { return raii_scoped_guard_t{}; }
};

template <typename T, bool IsMappedTypeV = false>
using guarded_synchronized = synchronized<T, IsMappedTypeV, raii_thread_state_policy_t>;
}  // namespace

class synchronized_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_thread_state_policy =
            std::make_unique<::testing::NiceMock<gmock_thread_state_policy_t>>();
        ON_CALL(*g_mock_thread_state_policy, scoped(::testing::_))
            .WillByDefault(::testing::Return(mock_scoped_guard_t{}));
    }

    void TearDown() override { g_mock_thread_state_policy.reset(); }
};

TEST_F(synchronized_test, rlock_reads_value)
{
    const traced_synchronized<int> data{ 42 };

    int seen = 0;
    data.rlock([&seen](const int& value) { seen = value; });

    EXPECT_EQ(seen, 42);
}

TEST_F(synchronized_test, wlock_mutates_value)
{
    traced_synchronized<int> data{ 0 };

    data.wlock([](int& value) { value = 7; });

    int seen = 0;
    data.rlock([&seen](const int& value) { seen = value; });
    EXPECT_EQ(seen, 7);
}

TEST_F(synchronized_test, ulock_skips_write_when_read_succeeds)
{
    traced_synchronized<std::string> data{ std::string{ "cached" } };

    bool       wrote = false;
    const bool found =
        data.ulock([](const std::string& value) { return value == "cached"; },
                   [&wrote](std::string& value) {
                       wrote = true;
                       value = "written";
                       return true;
                   });

    EXPECT_TRUE(found);
    EXPECT_FALSE(wrote);
}

TEST_F(synchronized_test, ulock_writes_when_read_fails)
{
    traced_synchronized<std::string> data{ std::string{ "stale" } };

    const bool found =
        data.ulock([](const std::string& value) { return value == "cached"; },
                   [](std::string& value) {
                       value = "written";
                       return true;
                   });

    EXPECT_TRUE(found);

    std::string seen;
    data.rlock([&seen](const std::string& value) { seen = value; });
    EXPECT_EQ(seen, "written");
}

using synchronized_thread_state_test = synchronized_test;

TEST_F(synchronized_thread_state_test, rlock_scopes_thread_state_to_internal)
{
    EXPECT_CALL(*g_mock_thread_state_policy, scoped(mock_thread_state_policy_t::Internal))
        .Times(1);

    const traced_synchronized<int> data{ 1 };
    data.rlock([](const int& /*value*/) {});
}

TEST_F(synchronized_thread_state_test, wlock_scopes_thread_state_to_internal)
{
    EXPECT_CALL(*g_mock_thread_state_policy, scoped(mock_thread_state_policy_t::Internal))
        .Times(1);

    traced_synchronized<int> data{ 1 };
    data.wlock([](int& /*value*/) {});
}

TEST_F(synchronized_thread_state_test, ulock_scopes_thread_state_to_internal_once)
{
    EXPECT_CALL(*g_mock_thread_state_policy, scoped(mock_thread_state_policy_t::Internal))
        .Times(1);

    traced_synchronized<int> data{ 1 };
    data.ulock([](const int& /*value*/) { return false; },
               [](int& /*value*/) { return true; });
}

// The tests above verify that the guard is taken. These verify *when* — the
// guard must be constructed before the lock is acquired, or an interposed
// pthread_rwlock_rdlock re-enters rocprof-sys on the same thread and deadlocks.
class synchronized_ordering_test : public ::testing::Test
{
protected:
    void SetUp() override { g_guard_depth.store(0); }
    void TearDown() override
    {
        EXPECT_EQ(g_guard_depth.load(), 0) << "a scoped guard leaked";
    }
};

TEST_F(synchronized_ordering_test, rlock_holds_guard_while_blocked_on_acquisition)
{
    guarded_synchronized<int> data{ 0 };

    std::atomic<bool> writer_holds_lock{ false };
    std::atomic<bool> reader_entered_callback{ false };
    std::atomic<bool> release_writer{ false };

    auto writer = std::thread{ [&]() {
        data.wlock([&](int& /*value*/) {
            writer_holds_lock.store(true);
            while(!release_writer.load())
                std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
        });
    } };

    while(!writer_holds_lock.load())
        std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });

    // The writer's own guard is live, so the reader's would be the second.
    const int baseline = g_guard_depth.load();

    auto reader = std::thread{ [&]() {
        data.rlock([&](const int& /*value*/) { reader_entered_callback.store(true); });
    } };

    // The reader cannot reach its callback while the writer holds the lock, so
    // any increment observed here necessarily happened before the reader
    // blocked on acquisition. If the guard were taken after the lock, the
    // reader would park in std::shared_lock and never reach it at all -- the
    // deadline below is a failure budget, not a race window.
    const auto deadline   = std::chrono::steady_clock::now() + std::chrono::seconds{ 5 };
    bool       guard_seen = false;
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(g_guard_depth.load() > baseline)
        {
            guard_seen = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
    }

    EXPECT_TRUE(guard_seen) << "rlock blocked on the mutex without an active thread "
                               "state guard; the guard must be constructed before the "
                               "lock is acquired";
    EXPECT_FALSE(reader_entered_callback.load())
        << "the reader was not actually blocked, so this test proved nothing";

    release_writer.store(true);
    writer.join();
    reader.join();
}

TEST_F(synchronized_ordering_test, rlock_guard_spans_the_callback)
{
    const guarded_synchronized<int> data{ 0 };

    data.rlock([](const int& /*value*/) { EXPECT_EQ(g_guard_depth.load(), 1); });

    EXPECT_EQ(g_guard_depth.load(), 0);
}

TEST_F(synchronized_ordering_test, wlock_guard_spans_the_callback)
{
    guarded_synchronized<int> data{ 0 };

    data.wlock([](int& /*value*/) { EXPECT_EQ(g_guard_depth.load(), 1); });

    EXPECT_EQ(g_guard_depth.load(), 0);
}

TEST_F(synchronized_ordering_test, ulock_holds_one_guard_across_both_phases)
{
    guarded_synchronized<int> data{ 0 };

    bool read_ran  = false;
    bool write_ran = false;

    data.ulock(
        [&read_ran](const int& /*value*/) {
            read_ran = true;
            EXPECT_EQ(g_guard_depth.load(), 1);
            return false;
        },
        [&write_ran](int& /*value*/) {
            write_ran = true;
            // Still 1, not 2: the read phase's guard must be reused, not
            // nested, and must not have been released between the phases.
            EXPECT_EQ(g_guard_depth.load(), 1);
            return true;
        });

    EXPECT_TRUE(read_ran);
    EXPECT_TRUE(write_ran);
    EXPECT_EQ(g_guard_depth.load(), 0);
}
