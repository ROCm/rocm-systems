// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/hipfile/types.hpp"
#include <cstdint>

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <string_view>

namespace rocprofsys::pmc::collectors::hipfile::testing
{
namespace
{
TEST(HipFileMetricTable, units_match_amd_smi_conventions)
{
    // Perfetto CounterTrack::set_unit_name and RocPD pmc_info both read metric.unit.
    // An empty string is dropped by CounterTrack, which is how Perfetto lost units
    // while RocPD kept them.
    const std::map<std::string_view, std::string_view> expected{
        { "Read Bytes", "bytes" },       { "Write Bytes", "bytes" },
        { "Read Ops", "count" },         { "Write Ops", "count" },
        { "Fastpath Reads", "count" },   { "Fastpath Writes", "count" },
        { "Fallback Reads", "count" },   { "Fallback Writes", "count" },
        { "Unaligned Reads", "count" },  { "Unaligned Writes", "count" },
        { "Read Errors", "count" },      { "Write Errors", "count" },
        { "Read Bandwidth", "bytes/s" }, { "Write Bandwidth", "bytes/s" },
    };

    ASSERT_EQ(k_metric_table.size(), expected.size());

    for(const auto& metric : k_metric_table)
    {
        SCOPED_TRACE(metric.suffix);
        const auto entry_it = expected.find(metric.suffix);
        ASSERT_NE(entry_it, expected.end());
        EXPECT_EQ(std::string_view{ metric.unit }, entry_it->second);
        EXPECT_FALSE(std::string_view{ metric.unit }.empty());
    }
}
TEST(HipFileMetricTable, named_bits_match_metric_table_positions)
{
    // enabled_metrics::bits names the same 14 flags that k_metric_table numbers, but the
    // two orderings are independent declarations: inserting a metric mid-table while
    // appending to the union would silently desynchronize them.
    using setter_t = void (*)(enabled_metrics&);
    const std::map<std::string_view, setter_t> expected{
        { "Read Bytes", [](enabled_metrics& flags) { flags.bits.read_bytes     = 1; } },
        { "Write Bytes", [](enabled_metrics& flags) { flags.bits.write_bytes   = 1; } },
        { "Read Ops", [](enabled_metrics& flags) { flags.bits.read_ops         = 1; } },
        { "Write Ops", [](enabled_metrics& flags) { flags.bits.write_ops       = 1; } },
        { "Fastpath Reads",
          [](enabled_metrics& flags) { flags.bits.fastpath_reads               = 1; } },
        { "Fastpath Writes",
          [](enabled_metrics& flags) { flags.bits.fastpath_writes              = 1; } },
        { "Fallback Reads",
          [](enabled_metrics& flags) { flags.bits.fallback_reads               = 1; } },
        { "Fallback Writes",
          [](enabled_metrics& flags) { flags.bits.fallback_writes              = 1; } },
        { "Unaligned Reads",
          [](enabled_metrics& flags) { flags.bits.unaligned_reads              = 1; } },
        { "Unaligned Writes",
          [](enabled_metrics& flags) { flags.bits.unaligned_writes             = 1; } },
        { "Read Errors", [](enabled_metrics& flags) { flags.bits.read_errors   = 1; } },
        { "Write Errors", [](enabled_metrics& flags) { flags.bits.write_errors = 1; } },
        { "Read Bandwidth",
          [](enabled_metrics& flags) { flags.bits.read_bandwidth               = 1; } },
        { "Write Bandwidth",
          [](enabled_metrics& flags) { flags.bits.write_bandwidth              = 1; } },
    };

    ASSERT_EQ(k_metric_table.size(), expected.size());

    std::uint32_t covered = 0;

    for(const auto& metric : k_metric_table)
    {
        SCOPED_TRACE(metric.suffix);
        const auto entry = expected.find(metric.suffix);
        ASSERT_NE(entry, expected.end());

        // NOLINTNEXTLINE(misc-const-correctness) -- the setter writes the selected bit
        enabled_metrics probe{};
        entry->second(probe);

        // Exactly the table's bit, and nothing else.
        EXPECT_EQ(probe.value, 1U << metric.bit);

        covered |= probe.value;
    }

    // Every bit in the mask the collector advertises is claimed by exactly one name.
    EXPECT_EQ(covered, k_all_hipfile_metrics);
}
}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
