// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/synchronized.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

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
using traced_synchronized = synchronized<T, mock_thread_state_policy_t, IsMappedTypeV>;
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
