// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/amd_smi/backend.hpp"
#include "mock_wrapper.hpp"

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

constexpr testing::mock_status_t k_ok     = testing::mock_backend::STATUS_SUCCESS;
constexpr testing::mock_status_t k_err    = 1;
constexpr std::uint64_t          k_handle = 0xABCDABCDULL;

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

// ─────────────────────────────────────────────────────────────────────────────
// Per-device GPU forwarding
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_gpu_asic_info_calls_backend_method)
{
    const testing::mock_asic_info_t raw{ .market_name = "Instinct MI300X",
                                         .vendor_name = "AMD" };
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    testing::mock_asic_info_t out{};
    m_session.get_gpu_asic_info(k_handle, &out);
    EXPECT_STREQ(out.market_name, "Instinct MI300X");
}

TEST_F(BackendTest, get_gpu_asic_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_asic_info_t out{};
    EXPECT_THROW(m_session.get_gpu_asic_info(k_handle, &out), std::runtime_error);
}

TEST_F(BackendTest, get_gpu_asic_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_asic_info_t out{};
    EXPECT_THROW(
        {
            try
            {
                m_session.get_gpu_asic_info(k_handle, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_asic_info"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, get_metrics_info_calls_backend_method)
{
    testing::mock_gpu_metrics_t raw{};
    raw.current_socket_power = 200;
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const auto out = m_session.get_metrics_info(k_handle);
    EXPECT_EQ(out.current_socket_power, 200U);
}

TEST_F(BackendTest, get_metrics_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    EXPECT_THROW(static_cast<void>(m_session.get_metrics_info(k_handle)),
                 std::runtime_error);
}

TEST_F(BackendTest, get_metrics_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(m_session.get_metrics_info(k_handle));
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_metrics_info"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, get_memory_usage_calls_backend_method)
{
    constexpr std::uint64_t k_bytes = 4ULL * 1024 * 1024 * 1024;
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<2>(k_bytes), Return(k_ok)));

    std::uint64_t out = 0;
    m_session.get_memory_usage(k_handle, 0, &out);
    EXPECT_EQ(out, k_bytes);
}

TEST_F(BackendTest, get_memory_usage_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    std::uint64_t out = 0;
    EXPECT_THROW(m_session.get_memory_usage(k_handle, 0, &out), std::runtime_error);
}

TEST_F(BackendTest, get_memory_usage_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    std::uint64_t out = 0;
    EXPECT_THROW(
        {
            try
            {
                m_session.get_memory_usage(k_handle, 0, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_memory_usage"));
                throw;
            }
        },
        std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature forwarding
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_temp_metric_returns_temperature_value)
{
    constexpr std::int64_t k_temp = 85000;
    EXPECT_CALL(*testing::g_mock_backend, get_temp_metric(k_handle, _, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(k_temp), Return(k_ok)));

    EXPECT_EQ(m_session.get_temp_metric(k_handle, sut_t::TEMPERATURE_TYPE_HOTSPOT,
                                        sut_t::TEMP_CURRENT),
              k_temp);
}

TEST_F(BackendTest, get_temp_metric_passes_sensor_type_and_metric)
{
    constexpr std::int64_t k_temp = 72000;
    EXPECT_CALL(*testing::g_mock_backend,
                get_temp_metric(k_handle, testing::mock_backend::TEMPERATURE_TYPE_EDGE,
                                testing::mock_backend::TEMP_CURRENT, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(k_temp), Return(k_ok)));

    EXPECT_EQ(m_session.get_temp_metric(k_handle, sut_t::TEMPERATURE_TYPE_EDGE,
                                        sut_t::TEMP_CURRENT),
              k_temp);
}

TEST_F(BackendTest, get_temp_metric_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_temp_metric(k_handle, _, _, _))
        .WillOnce(Return(k_err));

    EXPECT_THROW(static_cast<void>(m_session.get_temp_metric(
                     k_handle, sut_t::TEMPERATURE_TYPE_HOTSPOT, sut_t::TEMP_CURRENT)),
                 std::runtime_error);
}

TEST_F(BackendTest, get_temp_metric_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_temp_metric(k_handle, _, _, _))
        .WillOnce(Return(k_err));

    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(m_session.get_temp_metric(
                    k_handle, sut_t::TEMPERATURE_TYPE_HOTSPOT, sut_t::TEMP_CURRENT));
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_temp_metric"));
                throw;
            }
        },
        std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// SDMA forwarding
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, probe_sdma_support_returns_true_on_success)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(0U), Return(k_ok)));
    EXPECT_TRUE(m_session.probe_sdma_support(k_handle));
}

TEST_F(BackendTest, probe_sdma_support_returns_false_on_failure)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(Return(k_err));
    EXPECT_FALSE(m_session.probe_sdma_support(k_handle));
}

TEST_F(BackendTest, get_gpu_process_list_returns_process_vector)
{
    const testing::mock_proc_info_t procs[] = { { .sdma_usage = 500 } };
    InSequence                      seq;
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), SetArrayArgument<2>(procs, procs + 1),
                        Return(k_ok)));

    auto result = m_session.get_gpu_process_list(k_handle);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].sdma_usage, 500U);
}

TEST_F(BackendTest, get_gpu_process_list_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(Return(k_err));
    EXPECT_THROW(static_cast<void>(m_session.get_gpu_process_list(k_handle)),
                 std::runtime_error);
}

TEST_F(BackendTest, get_gpu_process_list_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend,
                get_gpu_process_list(k_handle, NotNull(), IsNull()))
        .WillOnce(Return(k_err));
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(m_session.get_gpu_process_list(k_handle));
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_process_list"));
                throw;
            }
        },
        std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// NIC handle enumeration + per-device NIC forwarding
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, enumerate_nic_handles_returns_empty_when_no_sockets)
{
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(0U), Return(k_ok)));
    EXPECT_TRUE(m_session.enumerate_nic_handles().empty());
}

TEST_F(BackendTest, enumerate_nic_handles_returns_handles_from_single_socket)
{
    constexpr std::uint64_t k_socket = 10;
    constexpr std::uint64_t k_nics[] = { 300, 301 };

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles_by_type(k_socket, _, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(2U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles_by_type(k_socket, _, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(2U), SetArrayArgument<2>(k_nics, k_nics + 2),
                        Return(k_ok)));

    auto handles = m_session.enumerate_nic_handles();
    ASSERT_EQ(handles.size(), 2U);
    EXPECT_EQ(handles[0], k_nics[0]);
    EXPECT_EQ(handles[1], k_nics[1]);
}

TEST_F(BackendTest, enumerate_nic_handles_skips_socket_with_no_nics)
{
    constexpr std::uint64_t k_socket = 10;

    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles_by_type(k_socket, _, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(0U), Return(k_ok)));
    EXPECT_TRUE(m_session.enumerate_nic_handles().empty());
}

TEST_F(BackendTest, get_nic_asic_info_calls_backend_method)
{
    const testing::mock_nic_asic_info_t raw{ .product_name = "CX7",
                                             .vendor_name  = "Mellanox" };
    EXPECT_CALL(*testing::g_mock_backend, get_nic_asic_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    testing::mock_nic_asic_info_t out{};
    m_session.get_nic_asic_info(k_handle, &out);
    EXPECT_STREQ(out.product_name, "CX7");
}

TEST_F(BackendTest, get_nic_asic_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_asic_info_t out{};
    EXPECT_THROW(m_session.get_nic_asic_info(k_handle, &out), std::runtime_error);
}

TEST_F(BackendTest, get_nic_asic_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_asic_info_t out{};
    EXPECT_THROW(
        {
            try
            {
                m_session.get_nic_asic_info(k_handle, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_nic_asic_info"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, get_nic_port_info_calls_backend_method)
{
    testing::mock_nic_port_info_t raw{};
    raw.num_ports       = 1;
    raw.ports[0].netdev = "mlx5_0";
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    testing::mock_nic_port_info_t out{};
    m_session.get_nic_port_info(k_handle, &out);
    EXPECT_EQ(out.num_ports, 1U);
}

TEST_F(BackendTest, get_nic_port_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_port_info_t out{};
    EXPECT_THROW(m_session.get_nic_port_info(k_handle, &out), std::runtime_error);
}

TEST_F(BackendTest, get_nic_port_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_port_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_port_info_t out{};
    EXPECT_THROW(
        {
            try
            {
                m_session.get_nic_port_info(k_handle, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_nic_port_info"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, get_nic_rdma_dev_info_calls_backend_method)
{
    testing::mock_nic_rdma_devices_info_t raw{};
    raw.num_rdma_dev                    = 1;
    raw.rdma_dev_info[0].num_rdma_ports = 4;
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    testing::mock_nic_rdma_devices_info_t out{};
    m_session.get_nic_rdma_dev_info(k_handle, &out);
    EXPECT_EQ(out.num_rdma_dev, 1U);
}

TEST_F(BackendTest, get_nic_rdma_dev_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_rdma_devices_info_t out{};
    EXPECT_THROW(m_session.get_nic_rdma_dev_info(k_handle, &out), std::runtime_error);
}

TEST_F(BackendTest, get_nic_rdma_dev_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_dev_info(k_handle, _))
        .WillOnce(Return(k_err));

    testing::mock_nic_rdma_devices_info_t out{};
    EXPECT_THROW(
        {
            try
            {
                m_session.get_nic_rdma_dev_info(k_handle, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_nic_rdma_dev_info"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(BackendTest, get_nic_rdma_port_statistics_calls_backend_method)
{
    testing::mock_nic_stat_t stat{ .name = "rx_bytes", .value = 12345 };
    std::uint32_t            count = 1;
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_nic_rdma_port_statistics(k_handle, std::uint8_t{ 0 }, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<2>(1U), SetArrayArgument<3>(&stat, &stat + 1),
                        Return(k_ok)));

    testing::mock_nic_stat_t out{};
    m_session.get_nic_rdma_port_statistics(k_handle, 0, &count, &out);
    EXPECT_STREQ(out.name, "rx_bytes");
    EXPECT_EQ(out.value, 12345U);
}

TEST_F(BackendTest, get_nic_rdma_port_statistics_throws_on_backend_error)
{
    std::uint32_t            count = 0;
    testing::mock_nic_stat_t out{};
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_port_statistics(k_handle, _, _, _))
        .WillOnce(Return(k_err));
    EXPECT_THROW(m_session.get_nic_rdma_port_statistics(k_handle, 0, &count, &out),
                 std::runtime_error);
}

TEST_F(BackendTest, get_nic_rdma_port_statistics_error_message_contains_function_name)
{
    std::uint32_t            count = 0;
    testing::mock_nic_stat_t out{};
    EXPECT_CALL(*testing::g_mock_backend, get_nic_rdma_port_statistics(k_handle, _, _, _))
        .WillOnce(Return(k_err));
    EXPECT_THROW(
        {
            try
            {
                m_session.get_nic_rdma_port_statistics(k_handle, 0, &count, &out);
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_nic_rdma_port_statistics"));
                throw;
            }
        },
        std::runtime_error);
}

}  // namespace rocprofsys::backends::amd_smi
