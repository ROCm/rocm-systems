// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/hipfile/types.hpp"

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
}  // namespace
}  // namespace rocprofsys::pmc::collectors::hipfile::testing
