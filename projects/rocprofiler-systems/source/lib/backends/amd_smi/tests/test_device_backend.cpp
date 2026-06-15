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

using MockApi = StrictMock<testing::gmock_backend_api>;
using sut_t   = backend<testing::mock_backend>;

constexpr testing::mock_backend::processor_handle k_handle = 0xDEAD'BEEF;
constexpr testing::mock_status_t k_ok  = testing::mock_backend::STATUS_SUCCESS;
constexpr testing::mock_status_t k_err = 1;

// ── Fixture ───────────────────────────────────────────────────────────────────

class BackendTest : public ::testing::Test
{
protected:
    void SetUp() override { testing::g_mock_backend = std::make_unique<MockApi>(); }
    void TearDown() override { testing::g_mock_backend.reset(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Static lifecycle
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, initialize_calls_backend_init)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_ok));
    sut_t::initialize();
}

TEST_F(BackendTest, initialize_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_err));
    EXPECT_THROW(sut_t::initialize(), std::runtime_error);
}

TEST_F(BackendTest, initialize_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, init()).WillOnce(Return(k_err));
    EXPECT_THROW(
        {
            try
            {
                sut_t::initialize();
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
    sut_t::shutdown();
}

TEST_F(BackendTest, shutdown_is_noexcept)
{
    // shutdown() must not throw even when backend returns error
    EXPECT_CALL(*testing::g_mock_backend, shutdown()).WillOnce(Return(k_err));
    EXPECT_NO_THROW(sut_t::shutdown());
}

TEST_F(BackendTest, get_lib_version_forwards_version_fields)
{
    const testing::mock_version_t raw{
        .major = 26, .minor = 3, .release = 0, .build = "26.3.0"
    };
    EXPECT_CALL(*testing::g_mock_backend, get_version(NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(raw), Return(k_ok)));

    auto ver = sut_t::get_lib_version();
    EXPECT_EQ(ver.major, 26U);
    EXPECT_EQ(ver.minor, 3U);
    EXPECT_EQ(ver.release, 0U);
}

TEST_F(BackendTest, get_lib_version_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_version(_)).WillOnce(Return(k_err));
    EXPECT_THROW(static_cast<void>(sut_t::get_lib_version()), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU handle enumeration
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, enumerate_gpu_handles_returns_empty_when_no_sockets)
{
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(0U), Return(k_ok)));

    auto handles = sut_t::enumerate_gpu_handles();
    EXPECT_TRUE(handles.empty());
}

TEST_F(BackendTest, enumerate_gpu_handles_throws_on_socket_count_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(Return(k_err));
    EXPECT_THROW(static_cast<void>(sut_t::enumerate_gpu_handles()), std::runtime_error);
}

TEST_F(BackendTest, enumerate_gpu_handles_throws_on_socket_data_error)
{
    InSequence seq;
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(Return(k_err));

    EXPECT_THROW(static_cast<void>(sut_t::enumerate_gpu_handles()), std::runtime_error);
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
    // Count returns 0 → socket is skipped silently, no data fetch
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(0U), Return(k_ok)));

    auto handles = sut_t::enumerate_gpu_handles();
    EXPECT_TRUE(handles.empty());
}

TEST_F(BackendTest, enumerate_gpu_handles_returns_all_handles_from_single_socket)
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

    auto handles = sut_t::enumerate_gpu_handles();
    ASSERT_EQ(handles.size(), 2U);
    EXPECT_EQ(handles[0], k_procs[0]);
    EXPECT_EQ(handles[1], k_procs[1]);
}

TEST_F(BackendTest, enumerate_gpu_handles_aggregates_handles_across_two_sockets)
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

    // Socket 0: one GPU
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[0], NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[0], NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U),
                        SetArrayArgument<2>(&k_proc_a, &k_proc_a + 1), Return(k_ok)));

    // Socket 1: one GPU
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[1], NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*testing::g_mock_backend,
                get_processor_handles(k_sockets[1], NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(1U),
                        SetArrayArgument<2>(&k_proc_b, &k_proc_b + 1), Return(k_ok)));

    auto handles = sut_t::enumerate_gpu_handles();
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

    EXPECT_THROW(static_cast<void>(sut_t::enumerate_gpu_handles()), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-device: get_memory_usage
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_memory_usage_forwards_handle_and_returns_value)
{
    EXPECT_CALL(
        *testing::g_mock_backend,
        get_memory_usage(k_handle, testing::mock_backend::MEM_TYPE_VRAM, NotNull()))
        .WillOnce(DoAll(SetArgPointee<2>(std::uint64_t{ 4096 }), Return(k_ok)));

    const sut_t dev{ k_handle };
    EXPECT_EQ(dev.get_memory_usage(), 4096U);
}

TEST_F(BackendTest, get_memory_usage_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(static_cast<void>(dev.get_memory_usage()), std::runtime_error);
}

TEST_F(BackendTest, get_memory_usage_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_memory_usage(k_handle, _, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(dev.get_memory_usage());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_memory_usage"));
                throw;
            }
        },
        std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-device: get_raw_sdma_usage / is_sdma_supported (SDMA guard off)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_raw_sdma_usage_returns_zero_when_sdma_unsupported)
{
    const sut_t dev{ k_handle };
    EXPECT_EQ(dev.get_raw_sdma_usage(), 0U);
}

TEST_F(BackendTest, is_sdma_supported_returns_false_when_sdma_unsupported)
{
    const sut_t dev{ k_handle };
    EXPECT_FALSE(dev.is_sdma_supported());
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-device: get_gpu_asic_info
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_gpu_asic_info_forwards_handle_and_converts_fields)
{
    const testing::mock_asic_info_t raw{ .market_name = "RX 9900 XT",
                                         .vendor_name = "AMD" };
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        info = dev.get_gpu_asic_info();
    EXPECT_EQ(info.product_name, "RX 9900 XT");
    EXPECT_EQ(info.vendor_name, "AMD");
}

TEST_F(BackendTest, get_gpu_asic_info_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(static_cast<void>(dev.get_gpu_asic_info()), std::runtime_error);
}

TEST_F(BackendTest, get_gpu_asic_info_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_gpu_asic_info(k_handle, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(dev.get_gpu_asic_info());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_asic_info"));
                throw;
            }
        },
        std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-device: get_gpu_metrics — conversion correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(BackendTest, get_gpu_metrics_converts_power_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.current_socket_power = 175;
    raw.average_socket_power = 120;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.current_socket_power, 175U);
    EXPECT_EQ(met.average_socket_power, 120U);
}

TEST_F(BackendTest, get_gpu_metrics_converts_temperature_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.temperature_hotspot = 82;
    raw.temperature_edge    = 60;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.hotspot_temperature, 82U);
    EXPECT_EQ(met.edge_temperature, 60U);
}

TEST_F(BackendTest, get_gpu_metrics_converts_activity_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.average_gfx_activity = 95;
    raw.average_umc_activity = 60;
    raw.average_mm_activity  = 30;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.gfx_activity, 95U);
    EXPECT_EQ(met.umc_activity, 60U);
    EXPECT_EQ(met.mm_activity, 30U);
}

TEST_F(BackendTest, get_gpu_metrics_populates_supported_clock_fields)
{
    testing::mock_gpu_metrics_t raw{};
    raw.current_gfxclk = 2400;
    raw.current_uclk   = 1000;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.gfx_clock_mhz, 2400U);
    EXPECT_EQ(met.mem_clock_mhz, 1000U);
}

TEST_F(BackendTest, get_gpu_metrics_zeroes_unsupported_clock_fields)
{
    // Default mock_gpu_metrics_t has sentinel 0xFFFF for clocks → zero in output
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(testing::mock_gpu_metrics_t{}), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.gfx_clock_mhz, 0U);
    EXPECT_EQ(met.mem_clock_mhz, 0U);
}

TEST_F(BackendTest, get_gpu_metrics_populates_supported_pcie_bandwidth)
{
    testing::mock_gpu_metrics_t raw{};
    raw.pcie_bandwidth_acc  = 12345;
    raw.pcie_bandwidth_inst = 6789;

    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(raw), Return(k_ok)));

    const sut_t dev{ k_handle };
    auto        met = dev.get_gpu_metrics();
    EXPECT_EQ(met.pcie.bandwidth.acc, 12345U);
    EXPECT_EQ(met.pcie.bandwidth.inst, 6789U);
}

TEST_F(BackendTest, get_gpu_metrics_throws_on_backend_error)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(static_cast<void>(dev.get_gpu_metrics()), std::runtime_error);
}

TEST_F(BackendTest, get_gpu_metrics_error_message_contains_function_name)
{
    EXPECT_CALL(*testing::g_mock_backend, get_metrics_info(k_handle, _))
        .WillOnce(Return(k_err));

    const sut_t dev{ k_handle };
    EXPECT_THROW(
        {
            try
            {
                static_cast<void>(dev.get_gpu_metrics());
            } catch(const std::runtime_error& ex)
            {
                EXPECT_THAT(ex.what(), HasSubstr("amdsmi_get_gpu_metrics_info"));
                throw;
            }
        },
        std::runtime_error);
}

}  // namespace rocprofsys::backends::amd_smi
