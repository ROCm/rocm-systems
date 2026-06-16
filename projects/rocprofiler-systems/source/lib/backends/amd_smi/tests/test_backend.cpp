// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/amd_smi/backend.hpp"
#include "mock_backend.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetArrayArgument;
using ::testing::StrictMock;

namespace rocprofsys::backends::amd_smi
{

using MockApi   = StrictMock<testing::gmock_backend_api>;
using factory_t = backend_factory<testing::mock_backend>;
using sut_t     = factory_t::backend_t;

constexpr testing::mock_status_t k_ok  = testing::mock_backend::STATUS_SUCCESS;
constexpr testing::mock_status_t k_err = 1;

// ── Fixture ───────────────────────────────────────────────────────────────────
//
// All tests operate on the session (handle-less) backend.

class BackendTest : public ::testing::Test
{
protected:
    void SetUp() override { testing::g_mock_backend = std::make_unique<MockApi>(); }
    void TearDown() override { testing::g_mock_backend.reset(); }

    sut_t m_session;
};

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, initialize_calls_backend_init)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_ok));
    m_session.initialize();
}

TEST_F(BackendTest, initialize_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_err));
    EXPECT_THROW(m_session.initialize(), std::runtime_error);
}

TEST_F(BackendTest, initialize_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_err));
    EXPECT_THROW(
        {
            try
            {
                m_session.initialize();
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_init"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, shutdown_calls_backend_shutdown)
{
    EXPECT_CALL(*testing::g_mock_backend, shutdown()).WillOnce(Return(k_ok));
    m_session.shutdown();
}

TEST_F(BackendTest, shutdown_is_noexcept)
{
    EXPECT_CALL(*testing::g_mock_backend, shutdown()).WillOnce(Return(k_err));
    EXPECT_NO_THROW(m_session.shutdown());
}

TEST_F(BackendTest, get_lib_version_returns_version_fields)
{
    const testing::mock_version_t raw{
        .major = 26, .minor = 3, .release = 0, .build = "26.3.0"
    };
    EXPECT_CALL(*testing::g_mock_backend, get_version(NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(raw), Return(k_ok)));

    auto ver = m_session.get_lib_version();
    EXPECT_EQ(ver.major, 26U);
    EXPECT_EQ(ver.minor, 3U);
    EXPECT_EQ(ver.release, 0U);
}

TEST_F(BackendTest, get_lib_version_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_version(_)).WillOnce(Return(k_err));
    EXPECT_THROW(static_cast<void>(m_session.get_lib_version()), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU handle enumeration
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, enumerate_gpu_handles_returns_empty_when_no_sockets)
{
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(0U), Return(k_ok)));

    EXPECT_TRUE(m_session.enumerate_gpu_handles().empty());
}

TEST_F(BackendTest, enumerate_gpu_handles_throws_on_socket_count_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(Return(k_err));
    EXPECT_THROW(static_cast<void>(m_session.enumerate_gpu_handles()),
                 std::runtime_error);
}

TEST_F(BackendTest, enumerate_gpu_handles_throws_on_socket_data_error)
{
    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(Return(k_err));

    EXPECT_THROW(static_cast<void>(m_session.enumerate_gpu_handles()),
                 std::runtime_error);
}

TEST_F(BackendTest, enumerate_gpu_handles_skips_socket_with_no_processors)
{
    constexpr std::uint64_t k_socket = 10;

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(0U), Return(k_ok)));

    EXPECT_TRUE(m_session.enumerate_gpu_handles().empty());
}

TEST_F(BackendTest, enumerate_gpu_handles_returns_handles_from_single_socket)
{
    constexpr std::uint64_t k_socket  = 42;
    constexpr std::uint64_t k_procs[] = { 100, 101 };

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), SetArrayArgument<2>(k_procs, k_procs + 2),
                        Return(k_ok)));

    auto handles = m_session.enumerate_gpu_handles();
    ASSERT_EQ(handles.size(), 2U);
    EXPECT_EQ(handles[0], k_procs[0]);
    EXPECT_EQ(handles[1], k_procs[1]);
}

TEST_F(BackendTest, enumerate_gpu_handles_aggregates_across_two_sockets)
{
    constexpr std::uint64_t k_sockets[] = { 10, 20 };
    constexpr std::uint64_t k_proc_a    = 100;
    constexpr std::uint64_t k_proc_b    = 200;

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(2U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(2U),
                        SetArrayArgument<1>(k_sockets, k_sockets + 2), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[0], NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[0], NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U),
                        SetArrayArgument<2>(&k_proc_a, &k_proc_a + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[1], NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[1], NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U),
                        SetArrayArgument<2>(&k_proc_b, &k_proc_b + 1), Return(k_ok)));

    auto handles = m_session.enumerate_gpu_handles();
    ASSERT_EQ(handles.size(), 2U);
    EXPECT_EQ(handles[0], k_proc_a);
    EXPECT_EQ(handles[1], k_proc_b);
}

TEST_F(BackendTest, enumerate_gpu_handles_throws_on_processor_data_error)
{
    constexpr std::uint64_t k_socket = 10;

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), NotNull()))
        .WillOnce(Return(k_err));

    EXPECT_THROW(static_cast<void>(m_session.enumerate_gpu_handles()),
                 std::runtime_error);
}

}  // namespace rocprofsys::backends::amd_smi
