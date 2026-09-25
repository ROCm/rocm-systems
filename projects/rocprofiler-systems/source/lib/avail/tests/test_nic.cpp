// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/nic.hpp"
#include "mock_wrapper.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::avail
{
namespace
{

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
constexpr std::uint64_t k_nic_handle = 11;

// Names mirror the production backend contract.
// NOLINTBEGIN(readability-identifier-naming)
struct fake_nic_backend
{
    using processor_handle     = std::uint64_t;
    using gpu_metrics_t        = backends::amd_smi::testing::mock_gpu_metrics_t;
    using asic_info_t          = backends::amd_smi::testing::mock_asic_info_t;
    using memory_type_t        = std::uint32_t;
    using temperature_type_t   = std::uint32_t;
    using temperature_metric_t = std::uint32_t;
    using nic_asic_info_t      = backends::amd_smi::testing::mock_nic_asic_info_t;
    using nic_port_info_t      = backends::amd_smi::testing::mock_nic_port_info_t;
    using nic_rdma_devices_info_t =
        backends::amd_smi::testing::mock_nic_rdma_devices_info_t;
    using nic_stat_t = backends::amd_smi::testing::mock_nic_stat_t;

    static constexpr bool                 sdma_supported           = false;
    static constexpr bool                 ainic_feature_gate       = true;
    static constexpr memory_type_t        MEM_TYPE_VRAM            = 0;
    static constexpr temperature_metric_t TEMP_CURRENT             = 0;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_HOTSPOT = 1;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_EDGE    = 2;

    std::vector<processor_handle> handles;
    std::string                   bdf = "0000:e0:00.0";

    [[nodiscard]] std::vector<processor_handle> enumerate_nic_handles() const
    {
        return handles;
    }

    void                        get_gpu_asic_info(processor_handle, asic_info_t*) const {}
    [[nodiscard]] std::string   get_device_bdf(processor_handle) const { return {}; }
    [[nodiscard]] gpu_metrics_t get_metrics_info(processor_handle) const { return {}; }
    void get_memory_usage(processor_handle, memory_type_t, std::uint64_t*) const {}
    [[nodiscard]] std::int64_t get_temp_metric(processor_handle, temperature_type_t,
                                               temperature_metric_t) const
    {
        return 0;
    }

    void get_nic_asic_info(processor_handle, nic_asic_info_t* info) const
    {
        *info = nic_asic_info_t{ .product_name = "Pollara 400", .vendor_name = "AMD" };
    }

    [[nodiscard]] std::string get_nic_device_bdf(processor_handle) const { return bdf; }

    void get_nic_port_info(processor_handle, nic_port_info_t* info) const
    {
        info->num_ports = 2;
        info->ports[0]  = { .port_num = 1, .netdev = "enp1s0" };
        info->ports[1]  = { .port_num = 2, .netdev = "enp1s1" };
    }

    void get_nic_rdma_dev_info(processor_handle, nic_rdma_devices_info_t* info) const
    {
        info->num_rdma_dev                       = 1;
        info->rdma_dev_info[0].num_rdma_ports    = 2;
        info->rdma_dev_info[0].rdma_port_info[0] = { .netdev = "enp1s0", .rdma_port = 1 };
        info->rdma_dev_info[0].rdma_port_info[1] = { .netdev = "enp1s1", .rdma_port = 2 };
    }

    void get_nic_rdma_port_statistics(processor_handle, std::uint8_t port_idx,
                                      std::uint32_t* count, nic_stat_t* stats) const
    {
        if(stats == nullptr)
        {
            *count = 1;
            return;
        }
        stats[0] = port_idx == 0 ? nic_stat_t{ .name = "rx_rdma_ucast_bytes", .value = 1 }
                                 : nic_stat_t{ .name = "vendor_counter", .value = 2 };
        *count   = 1;
    }
};
// NOLINTEND(readability-identifier-naming)

TEST(avail_nic, inventory_emits_parent_ports_and_per_port_metric_capabilities)
{
    auto backend     = std::make_shared<fake_nic_backend>();
    backend->handles = { k_nic_handle };

    const auto result = inventory::nic_inventory(backend, true, true);
    ASSERT_FALSE(result.issue);
    ASSERT_EQ(result.devices.size(), 3U);
    EXPECT_EQ(result.devices[0].id, "nic-0");
    EXPECT_EQ(result.devices[0].pci_bdf, std::optional<std::string>{ "0000:e0:00.0" });
    EXPECT_EQ(result.devices[1].parent_id, std::optional<std::string>{ "nic-0" });
    EXPECT_EQ(result.devices[1].name, "enp1s0");
    EXPECT_EQ(result.devices[2].name, "enp1s1");

    ASSERT_EQ(result.metrics.size(), 2U);
    EXPECT_EQ(result.metrics[0].name, "rx_rdma_ucast_bytes");
    EXPECT_EQ(result.metrics[0].unit, "bytes");
    EXPECT_EQ(result.metrics[0].device_id, std::optional<std::string>{ "nic-0-port-1" });
    EXPECT_EQ(result.metrics[1].name, "vendor_counter");
    EXPECT_EQ(result.metrics[1].description, "AMD SMI RDMA counter");
    EXPECT_EQ(result.metrics[1].device_id, std::optional<std::string>{ "nic-0-port-2" });
}

TEST(avail_nic, metric_only_inventory_does_not_emit_device_records)
{
    auto backend     = std::make_shared<fake_nic_backend>();
    backend->handles = { k_nic_handle };

    const auto result = inventory::nic_inventory(backend, false, true);
    EXPECT_TRUE(result.devices.empty());
    EXPECT_EQ(result.metrics.size(), 2U);
}
#endif

}  // namespace
}  // namespace rocprofsys::avail
