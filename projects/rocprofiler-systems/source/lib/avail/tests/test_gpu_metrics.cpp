// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/devices.hpp"
#include "avail/gpu_metrics.hpp"
#include "avail/records.hpp"
#include "backends/amd_smi/metric_tokens.hpp"
#include "mock_wrapper.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::avail
{
namespace
{

using mock_metrics               = backends::amd_smi::testing::mock_gpu_metrics_t;
using metric_group               = backends::amd_smi::gpu::metric_group;
constexpr std::uint64_t k_handle = 7;

[[nodiscard]] mock_metrics
unsupported_metrics()
{
    constexpr auto k_u16 = std::numeric_limits<std::uint16_t>::max();
    constexpr auto k_u32 = std::numeric_limits<std::uint32_t>::max();
    constexpr auto k_u64 = std::numeric_limits<std::uint64_t>::max();

    mock_metrics result{};
    result.current_socket_power = k_u32;
    result.average_socket_power = k_u32;
    result.temperature_hotspot  = k_u16;
    result.temperature_edge     = k_u16;
    result.average_gfx_activity = k_u16;
    result.average_umc_activity = k_u16;
    result.average_mm_activity  = k_u16;
    result.current_gfxclk       = k_u16;
    result.current_uclk         = k_u16;
    result.xgmi_link_width      = k_u16;
    result.xgmi_link_speed      = k_u16;
    result.pcie_link_width      = k_u16;
    result.pcie_link_speed      = k_u16;
    result.pcie_bandwidth_acc   = k_u64;
    result.pcie_bandwidth_inst  = k_u64;
    std::ranges::fill(result.vcn_activity, k_u16);
    std::ranges::fill(result.jpeg_activity, k_u16);
    std::ranges::fill(result.xgmi_read_data_acc, k_u64);
    std::ranges::fill(result.xgmi_write_data_acc, k_u64);
    for(auto& xcp : result.xcp_stats)
    {
        std::ranges::fill(xcp.vcn_busy, k_u16);
        std::ranges::fill(xcp.jpeg_busy, k_u16);
    }
    return result;
}

// Names mirror the production backend contract.
// NOLINTBEGIN(readability-identifier-naming)
struct fake_gpu_backend
{
    using processor_handle     = std::uint64_t;
    using gpu_metrics_t        = mock_metrics;
    using asic_info_t          = backends::amd_smi::testing::mock_asic_info_t;
    using memory_type_t        = std::uint32_t;
    using temperature_type_t   = std::uint32_t;
    using temperature_metric_t = std::uint32_t;

    static constexpr bool                 sdma_supported           = false;
    static constexpr bool                 ainic_feature_gate       = false;
    static constexpr memory_type_t        MEM_TYPE_VRAM            = 0;
    static constexpr temperature_metric_t TEMP_CURRENT             = 0;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_HOTSPOT = 1;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_EDGE    = 2;

    std::vector<processor_handle> handles;
    std::string                   bdf;
    gpu_metrics_t                 metrics      = unsupported_metrics();
    bool                          fail_metrics = false;

    [[nodiscard]] std::vector<processor_handle> enumerate_gpu_handles() const
    {
        return handles;
    }

    void                      get_gpu_asic_info(processor_handle, asic_info_t*) const {}
    [[nodiscard]] std::string get_device_bdf(processor_handle) const { return bdf; }

    [[nodiscard]] gpu_metrics_t get_metrics_info(processor_handle) const
    {
        if(fail_metrics)
        {
            throw std::runtime_error{ "metrics failed" };
        }
        return metrics;
    }

    void get_memory_usage(processor_handle, memory_type_t, std::uint64_t* value) const
    {
        *value = 0;
    }

    [[nodiscard]] std::int64_t get_temp_metric(processor_handle, temperature_type_t,
                                               temperature_metric_t) const
    {
        return 0;
    }
};
// NOLINTEND(readability-identifier-naming)

[[nodiscard]] const metric_record*
find_metric(const metrics_listing_result& result, std::string_view name)
{
    const auto itr =
        std::ranges::find_if(result.records, [name](const metric_record& record) {
            return record.name == name;
        });
    return (itr == result.records.end()) ? nullptr : &*itr;
}

TEST(avail_gpu_metrics, token_table_matches_profile_vocabulary_and_help)
{
    constexpr auto k_expected = std::array{
        "busy",          "temp",      "power",     "mem_usage",
        "sdma_usage",    "gfx_clock", "mem_clock", "vcn_activity",
        "jpeg_activity", "xgmi",      "pcie",
    };
    ASSERT_EQ(backends::amd_smi::gpu::k_metric_tokens.size(), k_expected.size());
    for(std::size_t idx = 0; idx < k_expected.size(); ++idx)
    {
        EXPECT_EQ(backends::amd_smi::gpu::k_metric_tokens[idx].name, k_expected[idx]);
        EXPECT_FALSE(backends::amd_smi::gpu::k_metric_tokens[idx].description.empty());
    }
    EXPECT_EQ(backends::amd_smi::gpu::k_default_tokens, "busy,temp,power,mem_usage");
}

TEST(avail_gpu_metrics, raw_sentinels_and_valid_zero_determine_token_support)
{
    auto raw     = unsupported_metrics();
    auto support = backends::amd_smi::gpu::detect_metric_support(raw);
    EXPECT_FALSE(support[metric_group::busy]);
    EXPECT_FALSE(support[metric_group::vcn_activity]);
    EXPECT_FALSE(support[metric_group::jpeg_activity]);
    EXPECT_FALSE(support[metric_group::xgmi]);
    EXPECT_FALSE(support[metric_group::pcie]);

    raw.average_gfx_activity     = 0;
    raw.xcp_stats[1].vcn_busy[0] = 0;
    raw.jpeg_activity[2]         = 0;
    raw.xgmi_write_data_acc[3]   = 0;
    raw.pcie_bandwidth_inst      = 0;
    support                      = backends::amd_smi::gpu::detect_metric_support(raw);
    EXPECT_TRUE(support[metric_group::busy]);
    EXPECT_TRUE(support[metric_group::vcn_activity]);
    EXPECT_TRUE(support[metric_group::jpeg_activity]);
    EXPECT_TRUE(support[metric_group::xgmi]);
    EXPECT_TRUE(support[metric_group::pcie]);
}

TEST(avail_gpu_metrics, inventory_associates_tokens_with_gpu_id_by_bdf)
{
    auto backend                          = std::make_shared<fake_gpu_backend>();
    backend->handles                      = { k_handle };
    backend->bdf                          = "0000:03:00.0";
    backend->metrics.average_gfx_activity = 0;

    gpu_agent_info agent;
    agent.device.id      = "gpu-1";
    agent.device.pci_bdf = backend->bdf;

    const auto result = inventory::gpu_metrics<fake_gpu_backend>({ agent }, backend);
    ASSERT_FALSE(result.issue);
    ASSERT_EQ(result.records.size(), backends::amd_smi::gpu::k_metric_tokens.size());
    const auto* busy = find_metric(result, "busy");
    ASSERT_NE(busy, nullptr);
    EXPECT_EQ(busy->device_id, std::optional<std::string>{ "gpu-1" });
    EXPECT_EQ(busy->status, availability::available);
    EXPECT_FALSE(busy->description.empty());
}

TEST(avail_gpu_metrics, metric_query_failure_emits_unknown_tokens_and_diagnostic)
{
    auto backend          = std::make_shared<fake_gpu_backend>();
    backend->handles      = { k_handle };
    backend->bdf          = "0000:03:00.0";
    backend->fail_metrics = true;

    gpu_agent_info agent;
    agent.device.id      = "gpu-0";
    agent.device.pci_bdf = backend->bdf;

    const auto result = inventory::gpu_metrics<fake_gpu_backend>({ agent }, backend);
    ASSERT_TRUE(result.issue);
    ASSERT_EQ(result.records.size(), backends::amd_smi::gpu::k_metric_tokens.size());
    EXPECT_TRUE(std::ranges::all_of(result.records, [](const metric_record& record) {
        return record.status == availability::unknown;
    }));
}

}  // namespace
}  // namespace rocprofsys::avail
