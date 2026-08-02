// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/device_providers/amd_smi/provider.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

namespace rocprofsys::pmc::device_providers::amd_smi
{

// ── Mock session ──────────────────────────────────────────────────────────────
// Mocks the session surface the provider calls — not the underlying AMD SMI
// C API. This keeps the test isolated to provider logic only.

struct mock_version
{
    std::uint32_t major   = 0;
    std::uint32_t minor   = 0;
    std::uint32_t release = 0;
    const char*   build   = nullptr;
};

struct gmock_session
{
    MOCK_METHOD(void, initialize, ());
    MOCK_METHOD(void, shutdown, ());
    MOCK_METHOD(mock_version, get_lib_version, ());
    MOCK_METHOD(std::vector<std::uint64_t>, enumerate_gpu_handles, ());
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    MOCK_METHOD(std::vector<std::uint64_t>, enumerate_nic_handles, ());
#endif
};

inline std::unique_ptr<gmock_session> g_mock;

struct mock_session
{
    void         initialize() { g_mock->initialize(); }
    void         shutdown() { g_mock->shutdown(); }
    mock_version get_lib_version() { return g_mock->get_lib_version(); }

    std::vector<std::uint64_t> enumerate_gpu_handles()
    {
        return g_mock->enumerate_gpu_handles();
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    std::vector<std::uint64_t> enumerate_nic_handles()
    {
        return g_mock->enumerate_nic_handles();
    }
#endif
};

struct mock_factory
{
    using backend_t = mock_session;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<mock_session>();
    }
};

using provider_t = provider<mock_factory>;

// ── Stub device types ─────────────────────────────────────────────────────────
// provider::get_{gpu,nic}_devices<Device>() constructs:
//   Device::backend_type(shared_ptr<backend_t>, handle)
//   Device(shared_ptr<Device::backend_type>, index)
// Stubs capture index so tests can verify sequential assignment.

struct stub_proxy
{
    stub_proxy(std::shared_ptr<mock_session>, std::uint64_t) {}
};

struct stub_gpu_device
{
    using backend_type = stub_proxy;

    std::shared_ptr<backend_type> proxy;
    std::size_t                   index;

    stub_gpu_device(std::shared_ptr<backend_type> b, std::size_t i)
    : proxy{ std::move(b) }
    , index{ i }
    {}
};

struct stub_nic_device
{
    using backend_type = stub_proxy;

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
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_session>>(); }
    void TearDown() override { g_mock.reset(); }

    void expect_init(std::uint32_t major = 26, std::uint32_t minor = 3,
                     std::uint32_t release = 0, const char* build = "26.3.0")
    {
        EXPECT_CALL(*g_mock, initialize());
        EXPECT_CALL(*g_mock, get_lib_version())
            .WillOnce(Return(mock_version{ major, minor, release, build }));
    }

    void expect_shutdown() { EXPECT_CALL(*g_mock, shutdown()); }
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
    EXPECT_CALL(*g_mock, initialize()).WillOnce(Throw(std::runtime_error("init failed")));

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
    expect_shutdown();  // called once from p.shutdown(); destructor is then a no-op

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
    expect_shutdown();  // only from p2's destructor; p1 is moved-from

    provider_t p1;
    provider_t p2{ std::move(p1) };
}

TEST_F(ProviderTest, move_assignment_shuts_down_overwritten_backend)
{
    // Two providers → two initialize+get_lib_version pairs.
    // shutdown #1: move-assignment releases p2's old backend.
    // shutdown #2: p2's destructor releases p1's backend.
    EXPECT_CALL(*g_mock, initialize()).Times(2);
    EXPECT_CALL(*g_mock, get_lib_version())
        .Times(2)
        .WillRepeatedly(Return(mock_version{ 1, 0, 0, nullptr }));
    EXPECT_CALL(*g_mock, shutdown()).Times(2);

    provider_t p1;
    provider_t p2;
    p2 = std::move(p1);
}

// ─────────────────────────────────────────────────────────────────────────────
// get_gpu_devices
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, get_gpu_devices_returns_empty_when_no_handles)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*g_mock, enumerate_gpu_handles())
        .WillOnce(Return(std::vector<std::uint64_t>{}));

    provider_t p;
    EXPECT_TRUE(p.get_gpu_devices<stub_gpu_device>().empty());
}

TEST_F(ProviderTest, get_gpu_devices_assigns_sequential_indices)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*g_mock, enumerate_gpu_handles())
        .WillOnce(Return(std::vector<std::uint64_t>{ 100, 101 }));

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

    EXPECT_CALL(*g_mock, enumerate_gpu_handles())
        .Times(2)
        .WillRepeatedly(Return(std::vector<std::uint64_t>{ 100 }));

    provider_t p;
    EXPECT_EQ(p.get_gpu_devices<stub_gpu_device>().size(), 1U);
    EXPECT_EQ(p.get_gpu_devices<stub_gpu_device>().size(), 1U);
}

// ─────────────────────────────────────────────────────────────────────────────
// get_nic_devices
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(ProviderTest, get_nic_devices_returns_empty_when_no_handles)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*g_mock, enumerate_nic_handles())
        .WillOnce(Return(std::vector<std::uint64_t>{}));

    provider_t p;
    EXPECT_TRUE(p.get_nic_devices<stub_nic_device>().empty());
}

TEST_F(ProviderTest, get_nic_devices_assigns_sequential_indices)
{
    expect_init();
    expect_shutdown();

    EXPECT_CALL(*g_mock, enumerate_nic_handles())
        .WillOnce(Return(std::vector<std::uint64_t>{ 300, 301, 302 }));

    provider_t p;
    auto       devices = p.get_nic_devices<stub_nic_device>();
    ASSERT_EQ(devices.size(), 3U);
    EXPECT_EQ(devices[0]->index, 0U);
    EXPECT_EQ(devices[1]->index, 1U);
    EXPECT_EQ(devices[2]->index, 2U);
}

}  // namespace rocprofsys::pmc::device_providers::amd_smi
