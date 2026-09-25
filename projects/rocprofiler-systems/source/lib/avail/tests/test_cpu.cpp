// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/cpu.hpp"
#include "avail/records.hpp"

#include "backends/procfs/backend.hpp"
#include "backends/procfs/metric_tokens.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace rocprofsys::avail
{
namespace
{

constexpr float k_cpu_frequency_mhz   = 2000.0F;
constexpr auto  k_expected_cpu_tokens = std::array{
    "frequency",    "load",        "memory",   "page_rss",  "virt_mem",    "peak_rss",
    "ctx_switches", "page_faults", "cpu_time", "user_time", "kernel_time",
};
constexpr auto k_expected_cpu_masks = std::array{
    backends::procfs::cpu::k_frequency_mask,
    backends::procfs::cpu::k_load_mask,
    backends::procfs::cpu::k_memory_mask,
    backends::procfs::cpu::k_page_rss_mask,
    backends::procfs::cpu::k_virt_mem_mask,
    backends::procfs::cpu::k_peak_rss_mask,
    backends::procfs::cpu::k_context_switch_mask,
    backends::procfs::cpu::k_page_faults_mask,
    backends::procfs::cpu::k_cpu_time_mask,
    backends::procfs::cpu::k_user_time_mask,
    backends::procfs::cpu::k_kernel_time_mask,
};

struct fake_procfs_backend
{
    backends::procfs::socket_topology_t                  topology;
    std::map<std::size_t, float>                         frequencies;
    std::map<std::size_t, backends::procfs::cpu_jiffies> jiffies;
    backends::procfs::rusage_snapshot                    rusage{ .peak_rss = 1 };

    [[nodiscard]] const backends::procfs::socket_topology_t& get_socket_topology()
        const noexcept
    {
        return topology;
    }

    [[nodiscard]] std::map<std::size_t, float> read_cpu_frequencies() const
    {
        return frequencies;
    }

    [[nodiscard]] std::map<std::size_t, backends::procfs::cpu_jiffies> read_proc_stat()
        const
    {
        return jiffies;
    }

    [[nodiscard]] backends::procfs::rusage_snapshot read_rusage() const { return rusage; }
};

[[nodiscard]] const metric_record*
find_cpu_metric(const cpu_listing_result& result, const std::string& name,
                const std::optional<std::string>& device_id)
{
    const auto itr =
        std::ranges::find_if(result.metrics, [&name, &device_id](const auto& record) {
            return record.name == name && record.device_id == device_id;
        });
    return (itr == result.metrics.end()) ? nullptr : &*itr;
}

void
expect_cpu_metric_token_table()
{
    ASSERT_EQ(backends::procfs::cpu::k_metric_tokens.size(),
              k_expected_cpu_tokens.size());
    for(std::size_t idx = 0; idx < k_expected_cpu_tokens.size(); ++idx)
    {
        const auto& token = backends::procfs::cpu::k_metric_tokens[idx];
        EXPECT_EQ(token.name, k_expected_cpu_tokens[idx]);
        EXPECT_FALSE(token.description.empty());
        EXPECT_EQ(backends::procfs::cpu::metric_selection_mask(token.kind),
                  k_expected_cpu_masks[idx]);
    }
}

TEST(avail_cpu, inventory_emits_socket_and_core_hierarchy)
{
    fake_procfs_backend backend;
    backend.topology = { { 0, { 0, 1 } }, { 1, { 2, 3 } } };

    const auto result = inventory::cpu_inventory(backend, true, false);

    ASSERT_FALSE(result.issue);
    ASSERT_EQ(result.devices.size(), 6U);
    EXPECT_EQ(result.devices[0].id, "cpu-socket-0");
    EXPECT_EQ(result.devices[0].kind, device_kind::cpu_socket);
    EXPECT_EQ(result.devices[1].id, "cpu-0");
    EXPECT_EQ(result.devices[1].kind, device_kind::cpu_core);
    EXPECT_EQ(result.devices[1].parent_id, std::optional<std::string>{ "cpu-socket-0" });
    EXPECT_EQ(result.devices[4].id, "cpu-2");
    EXPECT_EQ(result.devices[4].parent_id, std::optional<std::string>{ "cpu-socket-1" });
    EXPECT_FALSE(result.devices[1].numa_node);
    EXPECT_TRUE(result.metrics.empty());
}

TEST(avail_cpu, metrics_detect_per_core_support_and_keep_process_tokens)
{
    fake_procfs_backend backend;
    backend.topology    = { { 0, { 0, 1 } } };
    backend.frequencies = { { 0, k_cpu_frequency_mhz } };
    backend.jiffies     = { { 1, backends::procfs::cpu_jiffies{} } };

    const auto result = inventory::cpu_inventory(backend, false, true);

    ASSERT_FALSE(result.issue);
    EXPECT_TRUE(result.devices.empty());
    ASSERT_EQ(result.metrics.size(), 13U);

    const auto* frequency_cpu0 =
        find_cpu_metric(result, "frequency", std::optional<std::string>{ "cpu-0" });
    const auto* frequency_cpu1 =
        find_cpu_metric(result, "frequency", std::optional<std::string>{ "cpu-1" });
    const auto* load_cpu0 =
        find_cpu_metric(result, "load", std::optional<std::string>{ "cpu-0" });
    const auto* load_cpu1 =
        find_cpu_metric(result, "load", std::optional<std::string>{ "cpu-1" });
    ASSERT_NE(frequency_cpu0, nullptr);
    ASSERT_NE(frequency_cpu1, nullptr);
    ASSERT_NE(load_cpu0, nullptr);
    ASSERT_NE(load_cpu1, nullptr);
    EXPECT_EQ(frequency_cpu0->status, availability::available);
    EXPECT_EQ(frequency_cpu1->status, availability::unavailable);
    EXPECT_EQ(load_cpu0->status, availability::unavailable);
    EXPECT_EQ(load_cpu1->status, availability::available);

    const auto* memory = find_cpu_metric(result, "memory", std::nullopt);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(memory->status, availability::available);
    EXPECT_EQ(memory->unit, "bytes");

    backend.rusage.peak_rss  = 0;
    const auto  without_peak = inventory::cpu_inventory(backend, false, true);
    const auto* peak_rss     = find_cpu_metric(without_peak, "peak_rss", std::nullopt);
    ASSERT_NE(peak_rss, nullptr);
    EXPECT_EQ(peak_rss->status, availability::unavailable);
}

TEST(avail_cpu, token_table_matches_cpu_metrics_configuration_vocabulary)
{
    expect_cpu_metric_token_table();
    EXPECT_EQ(backends::procfs::cpu::k_default_tokens, "all");
}

}  // namespace
}  // namespace rocprofsys::avail
