// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/device_backend.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include "mock_backend.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetArrayArgument;
using ::testing::StrictMock;

namespace rocprofsys::pmc::device_providers::amd_smi
{

namespace bknd    = ::rocprofsys::backends::amd_smi;
namespace bk_test = ::rocprofsys::backends::amd_smi::testing;

using MockApi        = StrictMock<bk_test::gmock_backend_api>;
using factory_t      = bknd::backend_factory<bk_test::mock_backend>;
using backend_t      = factory_t::backend_t;
using device_proxy_t = bknd::device_backend<backend_t>;
using provider_t     = provider<factory_t>;

constexpr bk_test::mock_status_t k_ok  = bk_test::mock_backend::STATUS_SUCCESS;
constexpr bk_test::mock_status_t k_err = 1;

// ── Stub device types ─────────────────────────────────────────────────────────
// provider::get_{gpu,nic}_devices<Device>() constructs:
//   Device::backend_type(shared_ptr<backend_t>, handle)   ← device_backend ctor
//   Device(shared_ptr<Device::backend_type>, index)
// The stub captures both to let tests inspect what was created.

struct stub_gpu_device
{
    using backend_type = device_proxy_t;

    std::shared_ptr<backend_type> proxy;
    std::size_t                   index;

    stub_gpu_device(std::shared_ptr<backend_type> b, std::size_t i)
    : proxy{ std::move(b) }
    , index{ i }
    {}
};

struct stub_nic_device
{
    using backend_type = device_proxy_t;

    std::shared_ptr<backend_type> proxy;
    std::size_t                   index;

    stub_nic_device(std::shared_ptr<backend_type> b, std::size_t i)
    : proxy{ std::move(b) }
    , index{ i }
    {}
};

// ── Fixture ───────────────────────────────────────────────────────────────────

class ProviderTest : public ::testing::Test
{
protected:
    void SetUp() override { bk_test::g_mock_backend = std::make_unique<MockApi>(); }
    void TearDown() override { bk_test::g_mock_backend.reset(); }

    // Expects the init → get_version pair every successful provider ctor triggers.
    void expect_init(std::uint32_t major = 26, std::uint32_t minor = 3,
                     std::uint32_t release = 0, const char* build = "26.3.0")
    {
        const bk_test::mock_version_t ver{
            .major = major, .minor = minor, .release = release, .build = build
        };
        EXPECT_CALL(*bk_test::g_mock_backend, init()).WillOnce(Return(k_ok));
        EXPECT_CALL(*bk_test::g_mock_backend, get_version(NotNull()))
            .WillOnce(DoAll(SetArgPointee<0>(ver), Return(k_ok)));
    }

    void expect_shutdown()
    {
        EXPECT_CALL(*bk_test::g_mock_backend, shutdown()).WillOnce(Return(k_ok));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Constructor — version storage
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, constructor_stores_version_fields)
{
    expect_init(26, 3, 1, "26.3.1");
    expect_shutdown();

    provider_t  p;
    const auto& v = p.get_version();
    EXPECT_EQ(v.numeric_representation.major, 26U);
    EXPECT_EQ(v.numeric_representation.minor, 3U);
    EXPECT_EQ(v.numeric_representation.release, 1U);
    EXPECT_EQ(v.string_representation, "26.3.1");
}

TEST_F(ProviderTest, constructor_null_build_string_stored_as_empty)
{
    expect_init(1, 0, 0, nullptr);
    expect_shutdown();

    provider_t p;
    EXPECT_TRUE(p.get_version().string_representation.empty());
}

TEST_F(ProviderTest, constructor_throws_when_initialize_fails)
{
    EXPECT_CALL(*bk_test::g_mock_backend, init()).WillOnce(Return(k_err));

    EXPECT_THROW(provider_t{}, std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shutdown
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, destructor_calls_shutdown)
{
    expect_init();
    expect_shutdown();

    {
        provider_t p;
    }
}

TEST_F(ProviderTest, explicit_shutdown_calls_backend_shutdown)
{
    expect_init();
    expect_shutdown();  // called once from p.shutdown(); destructor is then no-op

    provider_t p;
    p.shutdown();
}

TEST_F(ProviderTest, shutdown_is_idempotent)
{
    expect_init();
    expect_shutdown();  // exactly one call — second shutdown sees null m_backend_api

    provider_t p;
    p.shutdown();
    p.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Move semantics
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, move_constructor_produces_exactly_one_shutdown)
{
    expect_init();
    expect_shutdown();  // only from p2's destructor; p1's destructor is a no-op

    provider_t p1;
    provider_t p2{ std::move(p1) };
}

TEST_F(ProviderTest, move_assignment_shuts_down_overwritten_backend)
{
    // Two providers created → two init+version pairs, two shutdowns:
    // shutdown #1: from move-assignment (p2's old backend released)
    // shutdown #2: from p2's destructor (carries p1's backend)
    EXPECT_CALL(*bk_test::g_mock_backend, init()).Times(2).WillRepeatedly(Return(k_ok));
    const bk_test::mock_version_t ver{
        .major = 1, .minor = 0, .release = 0, .build = nullptr
    };
    EXPECT_CALL(*bk_test::g_mock_backend, get_version(NotNull()))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<0>(ver), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend, shutdown())
        .Times(2)
        .WillRepeatedly(Return(k_ok));

    provider_t p1;
    provider_t p2;
    p2 = std::move(p1);
}

// ─────────────────────────────────────────────────────────────────────────────
// get_gpu_devices
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, get_gpu_devices_returns_empty_when_no_sockets)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(0U), Return(k_ok)));

    provider_t p;
    EXPECT_TRUE(p.get_gpu_devices<stub_gpu_device>().empty());
}

TEST_F(ProviderTest, get_gpu_devices_assigns_sequential_indices)
{
    expect_init();
    expect_shutdown();

    // Use const locals so SetArrayArgument pointers remain valid through the test.
    const std::uint64_t k_socket  = 42;
    const std::uint64_t k_procs[] = { 100, 101 };

    InSequence seq;
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(2U), SetArrayArgument<2>(k_procs, k_procs + 2),
                        Return(k_ok)));

    provider_t p;
    auto       devices = p.get_gpu_devices<stub_gpu_device>();
    ASSERT_EQ(devices.size(), 2U);
    EXPECT_EQ(devices[0]->index, 0U);
    EXPECT_EQ(devices[1]->index, 1U);
}

TEST_F(ProviderTest, get_gpu_devices_repeated_call_re_enumerates)
{
    expect_init();
    expect_shutdown();

    constexpr std::uint64_t k_socket = 10;
    constexpr std::uint64_t k_proc   = 100;

    // Two separate calls to get_gpu_devices → two socket enumerations.
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<0>(1U),
                              SetArrayArgument<1>(&k_socket, &k_socket + 1),
                              Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), IsNull()))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<1>(1U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles(k_socket, NotNull(), NotNull()))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<1>(1U),
                              SetArrayArgument<2>(&k_proc, &k_proc + 1), Return(k_ok)));

    provider_t p;
    EXPECT_EQ(p.get_gpu_devices<stub_gpu_device>().size(), 1U);
    EXPECT_EQ(p.get_gpu_devices<stub_gpu_device>().size(), 1U);
}

// ─────────────────────────────────────────────────────────────────────────────
// get_nic_devices
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, get_nic_devices_returns_empty_when_no_sockets)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(0U), Return(k_ok)));

    provider_t p;
    EXPECT_TRUE(p.get_nic_devices<stub_nic_device>().empty());
}

TEST_F(ProviderTest, get_nic_devices_assigns_sequential_indices)
{
    expect_init();
    expect_shutdown();

    const std::uint64_t k_socket = 10;
    const std::uint64_t k_nics[] = { 300, 301, 302 };

    InSequence seq;
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), IsNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend, get_socket_handles(NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<0>(1U),
                        SetArrayArgument<1>(&k_socket, &k_socket + 1), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles_by_type(k_socket, _, IsNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(3U), Return(k_ok)));
    EXPECT_CALL(*bk_test::g_mock_backend,
                get_processor_handles_by_type(k_socket, _, NotNull(), NotNull()))
        .WillOnce(DoAll(SetArgPointee<3>(3U), SetArrayArgument<2>(k_nics, k_nics + 3),
                        Return(k_ok)));

    provider_t p;
    auto       devices = p.get_nic_devices<stub_nic_device>();
    ASSERT_EQ(devices.size(), 3U);
    EXPECT_EQ(devices[0]->index, 0U);
    EXPECT_EQ(devices[1]->index, 1U);
    EXPECT_EQ(devices[2]->index, 2U);
}

}  // namespace rocprofsys::pmc::device_providers::amd_smi
