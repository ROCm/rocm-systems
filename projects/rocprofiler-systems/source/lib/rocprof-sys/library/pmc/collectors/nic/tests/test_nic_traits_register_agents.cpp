// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "library/pmc/collectors/nic/device.hpp"
#include "library/pmc/collectors/nic/nic_traits.hpp"
#include "library/pmc/common/types.hpp"
#include "mock_nic_backend.hpp"
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using ::rocprofsys::agent;
using ::rocprofsys::agent_manager;
using ::rocprofsys::agent_type;
using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::nic_device_filter;

using namespace rocprofsys::pmc::collectors::nic;
using ::testing::AtLeast;
using ::testing::Return;
using ::testing::StrictMock;

using MockBackend = StrictMock<rocprofsys::backends::amd_smi::testing::mock_nic_backend>;
using test_device_t = device<MockBackend>;

namespace rocprofsys::pmc::collectors::nic::testing
{

constexpr size_t kHwNicDeviceId0 = 0;
constexpr size_t kHwNicDeviceId1 = 1;
constexpr size_t kHwNicDeviceId2 = 2;
constexpr size_t kHwNicDeviceId5 = 5;
constexpr size_t kHwNicDeviceId7 = 7;
constexpr size_t kHwNicDeviceId9 = 9;

struct stub_settings
{
    inline static nic_device_filter filter{};

    static void reset()
    {
        filter      = nic_device_filter{};
        filter.mode = device_selection_mode::all;
        filter.names.clear();
    }

    static nic_device_filter get_nic_device_filter() { return filter; }
};

struct stub_provider
{
    std::vector<std::shared_ptr<test_device_t>> devices;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_nic_devices()
    {
        static_assert(std::is_same_v<Device, test_device_t>,
                      "stub_provider only serves the device type under test");
        return devices;
    }
};

using traits_t = nic_traits<stub_provider, test_device_t>;

class NicTraitsRegisterAgentsTest : public ::testing::Test
{
protected:
    void SetUp() override { stub_settings::reset(); }

    static size_t nic_agent_count()
    {
        return ::rocprofsys::get_agent_manager_instance()
            .get_agents_by_type(agent_type::nic)
            .size();
    }

    static size_t nic_agent_count_with_device_id(size_t device_id)
    {
        size_t count = 0;
        for(const auto& agent_ptr :
            ::rocprofsys::get_agent_manager_instance().get_agents_by_type(
                agent_type::nic))
        {
            if(agent_ptr->device_id == device_id) ++count;
        }
        return count;
    }

    void expect_supported_nic_backend(MockBackend& backend, const std::string& port_name,
                                      const std::string& product_name = "AMD AINIC Test",
                                      const std::string& vendor_name  = "AMD")
    {
        EXPECT_CALL(backend, get_nic_asic_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(asic_info{ product_name, vendor_name }));

        EXPECT_CALL(backend, get_nic_port_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(port_info{ port_name }));

        EXPECT_CALL(backend, get_nic_rdma_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(rdma_info{ 1 }));

        EXPECT_CALL(backend, get_nic_rdma_port_statistics(0))
            .Times(AtLeast(1))
            .WillRepeatedly(
                Return(std::vector<stat_entry>{ { "rx_rdma_ucast_bytes", 0 } }));
    }

    void expect_unsupported_nic_backend(MockBackend&       backend,
                                        const std::string& port_name)
    {
        EXPECT_CALL(backend, get_nic_asic_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(asic_info{ "No RDMA NIC", "AMD" }));

        EXPECT_CALL(backend, get_nic_port_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(port_info{ port_name }));

        EXPECT_CALL(backend, get_nic_rdma_info())
            .Times(AtLeast(1))
            .WillRepeatedly(Return(rdma_info{ 0 }));
    }

    std::shared_ptr<test_device_t> make_nic_device(
        size_t hardware_index, const std::string& port_name,
        const std::string& product_name = "AMD AINIC Test",
        const std::string& vendor_name  = "AMD")
    {
        auto backend = std::make_shared<MockBackend>();
        expect_supported_nic_backend(*backend, port_name, product_name, vendor_name);
        return std::make_shared<test_device_t>(backend, hardware_index);
    }

    std::shared_ptr<test_device_t> make_unsupported_nic_device(
        size_t hardware_index, const std::string& port_name)
    {
        auto backend = std::make_shared<MockBackend>();
        expect_unsupported_nic_backend(*backend, port_name);
        return std::make_shared<test_device_t>(backend, hardware_index);
    }

    static std::shared_ptr<stub_provider> make_provider(
        std::vector<std::shared_ptr<test_device_t>> devices)
    {
        auto provider     = std::make_shared<stub_provider>();
        provider->devices = std::move(devices);
        return provider;
    }

    static std::vector<traits_t::device_entry> enumerate_nics(
        const std::shared_ptr<stub_provider>& provider)
    {
        return traits_t::enumerate_devices<stub_settings>(provider);
    }

    static std::set<size_t> entry_indices(
        const std::vector<traits_t::device_entry>& entries)
    {
        std::set<size_t> indices;
        for(const auto& entry : entries)
        {
            indices.insert(entry.device->get_index());
        }
        return indices;
    }
};

TEST_F(NicTraitsRegisterAgentsTest, register_nic_agents_lookup_by_hardware_device_id)
{
    const auto nic_before = nic_agent_count();

    auto       provider = make_provider({
        make_nic_device(kHwNicDeviceId2, "rdma2", "AINIC-2"),
        make_nic_device(kHwNicDeviceId5, "rdma5", "AINIC-5"),
    });
    const auto entries  = enumerate_nics(provider);
    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entry_indices(entries),
              (std::set<size_t>{ kHwNicDeviceId2, kHwNicDeviceId5 }));

    EXPECT_EQ(nic_agent_count(), nic_before + 2);

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();
    EXPECT_NO_THROW({
        const agent& by_id_2 = mgr.get_agent_by_id(kHwNicDeviceId2, agent_type::nic);
        EXPECT_EQ(by_id_2.device_id, static_cast<std::uint64_t>(kHwNicDeviceId2));
        EXPECT_EQ(by_id_2.name, "AINIC-2");
    });
    EXPECT_NO_THROW({
        const agent& by_id_5 = mgr.get_agent_by_id(kHwNicDeviceId5, agent_type::nic);
        EXPECT_EQ(by_id_5.device_id, static_cast<std::uint64_t>(kHwNicDeviceId5));
        EXPECT_EQ(by_id_5.name, "AINIC-5");
    });
}

TEST_F(NicTraitsRegisterAgentsTest, sampling_disabled_does_not_register_agents)
{
    const auto nic_before      = nic_agent_count();
    stub_settings::filter.mode = device_selection_mode::none;

    auto       provider = make_provider({ make_nic_device(0, "rdma0") });
    const auto entries  = enumerate_nics(provider);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(nic_agent_count(), nic_before);
}

TEST_F(NicTraitsRegisterAgentsTest, empty_discovery_registers_no_agents)
{
    const auto nic_before = nic_agent_count();

    const auto entries = enumerate_nics(make_provider({}));

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(nic_agent_count(), nic_before);
}

TEST_F(NicTraitsRegisterAgentsTest, specific_filter_registers_only_matching_port_name)
{
    const auto nic_before      = nic_agent_count();
    const auto id2_before      = nic_agent_count_with_device_id(kHwNicDeviceId2);
    const auto id5_before      = nic_agent_count_with_device_id(kHwNicDeviceId5);
    stub_settings::filter.mode = device_selection_mode::specific;
    stub_settings::filter.names.insert("rdma1");

    auto       provider = make_provider({
        make_nic_device(kHwNicDeviceId2, "rdma0"),
        make_nic_device(kHwNicDeviceId5, "rdma1", "AINIC-5"),
    });
    const auto entries  = enumerate_nics(provider);

    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entry_indices(entries), (std::set<size_t>{ kHwNicDeviceId5 }));
    EXPECT_EQ(nic_agent_count(), nic_before + 1);
    EXPECT_EQ(nic_agent_count_with_device_id(kHwNicDeviceId2), id2_before);
    EXPECT_EQ(nic_agent_count_with_device_id(kHwNicDeviceId5), id5_before + 1);

    bool found_ainic_5 = false;
    for(const auto& agent_ptr :
        ::rocprofsys::get_agent_manager_instance().get_agents_by_type(agent_type::nic))
    {
        if(agent_ptr->device_id == kHwNicDeviceId5 && agent_ptr->name == "AINIC-5")
            found_ainic_5 = true;
    }
    EXPECT_TRUE(found_ainic_5);
}

TEST_F(NicTraitsRegisterAgentsTest, unsupported_nic_skipped_under_all_mode)
{
    const auto nic_before = nic_agent_count();
    const auto id2_before = nic_agent_count_with_device_id(kHwNicDeviceId2);
    const auto id5_before = nic_agent_count_with_device_id(kHwNicDeviceId5);

    auto       provider = make_provider({
        make_nic_device(kHwNicDeviceId2, "rdma0"),
        make_unsupported_nic_device(kHwNicDeviceId5, "rdma-no-rdma"),
    });
    const auto entries  = enumerate_nics(provider);

    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entry_indices(entries), (std::set<size_t>{ kHwNicDeviceId2 }));
    EXPECT_EQ(nic_agent_count(), nic_before + 1);
    EXPECT_EQ(nic_agent_count_with_device_id(kHwNicDeviceId2), id2_before + 1);
    EXPECT_EQ(nic_agent_count_with_device_id(kHwNicDeviceId5), id5_before);
}

TEST_F(NicTraitsRegisterAgentsTest,
       device_type_index_is_sequential_per_type_device_id_is_hardware)
{
    const auto nic_before = nic_agent_count();

    auto provider = make_provider({
        make_nic_device(kHwNicDeviceId2, "rdma0"),
        make_nic_device(kHwNicDeviceId5, "rdma1"),
    });
    ASSERT_EQ(enumerate_nics(provider).size(), 2U);

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();
    // Use type_index: earlier tests may have registered the same hardware device_id.
    const agent& at_2 = mgr.get_agent_by_type_index(nic_before, agent_type::nic);
    const agent& at_5 = mgr.get_agent_by_type_index(nic_before + 1, agent_type::nic);

    EXPECT_EQ(at_2.device_id, static_cast<std::uint64_t>(kHwNicDeviceId2));
    EXPECT_EQ(at_5.device_id, static_cast<std::uint64_t>(kHwNicDeviceId5));
    EXPECT_EQ(at_2.device_type_index, nic_before);
    EXPECT_EQ(at_5.device_type_index, nic_before + 1);
}

TEST_F(NicTraitsRegisterAgentsTest, get_agent_by_type_index_matches_registered_nics)
{
    const auto nic_before = nic_agent_count();

    auto provider = make_provider({
        make_nic_device(kHwNicDeviceId7, "rdma0", "AINIC-by-id-7"),
        make_nic_device(kHwNicDeviceId9, "rdma1", "AINIC-by-id-9"),
    });
    ASSERT_EQ(enumerate_nics(provider).size(), 2U);

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();

    const agent& by_type_first = mgr.get_agent_by_type_index(nic_before, agent_type::nic);
    const agent& by_type_second =
        mgr.get_agent_by_type_index(nic_before + 1, agent_type::nic);

    EXPECT_EQ(by_type_first.device_id, static_cast<std::uint64_t>(kHwNicDeviceId7));
    EXPECT_EQ(by_type_second.device_id, static_cast<std::uint64_t>(kHwNicDeviceId9));
    EXPECT_EQ(by_type_first.name, "AINIC-by-id-7");
    EXPECT_EQ(by_type_second.name, "AINIC-by-id-9");

    EXPECT_EQ(by_type_first.device_id,
              mgr.get_agent_by_id(kHwNicDeviceId7, agent_type::nic).device_id);
    EXPECT_EQ(by_type_second.device_id,
              mgr.get_agent_by_id(kHwNicDeviceId9, agent_type::nic).device_id);

    // type_index is the per-type ordinal from insert_agent, not hardware device_id.
    EXPECT_NE(by_type_first.device_id, static_cast<std::uint64_t>(nic_before));
    EXPECT_NE(by_type_second.device_id, static_cast<std::uint64_t>(nic_before + 1));

    EXPECT_THROW(mgr.get_agent_by_type_index(nic_before + 2, agent_type::nic),
                 std::out_of_range);
}

TEST_F(NicTraitsRegisterAgentsTest, single_nic_at_index_zero_uses_hardware_device_id)
{
    const auto nic_before = nic_agent_count();

    ASSERT_EQ(enumerate_nics(make_provider({ make_nic_device(0, "rdma0") })).size(), 1U);
    EXPECT_EQ(nic_agent_count(), nic_before + 1);

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();
    const agent&   nic = mgr.get_agent_by_id(0, agent_type::nic);
    EXPECT_EQ(nic.device_id, 0U);
}

TEST_F(NicTraitsRegisterAgentsTest, provider_order_does_not_change_hardware_device_id)
{
    const auto nic_before = nic_agent_count();

    auto provider = make_provider({
        make_nic_device(kHwNicDeviceId5, "rdma1"),
        make_nic_device(kHwNicDeviceId2, "rdma0"),
    });
    ASSERT_EQ(enumerate_nics(provider).size(), 2U);

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();
    EXPECT_EQ(mgr.get_agent_by_id(kHwNicDeviceId2, agent_type::nic).device_id,
              static_cast<std::uint64_t>(kHwNicDeviceId2));
    EXPECT_EQ(mgr.get_agent_by_id(kHwNicDeviceId5, agent_type::nic).device_id,
              static_cast<std::uint64_t>(kHwNicDeviceId5));
    EXPECT_EQ(nic_agent_count(), nic_before + 2);
}

TEST_F(NicTraitsRegisterAgentsTest, agent_metadata_populated_from_device)
{
    const auto nic_before = nic_agent_count();

    enumerate_nics(make_provider(
        { make_nic_device(kHwNicDeviceId1, "rdma0", "Product-X", "Vendor-Y") }));

    agent_manager& mgr = ::rocprofsys::get_agent_manager_instance();
    const agent&   nic = mgr.get_agent_by_type_index(nic_before, agent_type::nic);

    EXPECT_EQ(nic.device_id, static_cast<std::uint64_t>(kHwNicDeviceId1));
    EXPECT_EQ(nic.name, "Product-X");
    EXPECT_EQ(nic.model_name, "Vendor-Y");
    EXPECT_EQ(nic.vendor_name, "AI NIC");
    EXPECT_EQ(nic.product_name, "AI NIC");
    EXPECT_EQ(nic.device_type_index, nic_before);
}

}  // namespace rocprofsys::pmc::collectors::nic::testing
