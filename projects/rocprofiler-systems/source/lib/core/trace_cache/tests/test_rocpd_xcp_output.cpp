// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

//
// Unit tests for XCP stats output naming and gating logic.
//
// These tests replicate the XCP naming/gating logic from perfetto_processor.cpp
// and rocpd_processor.cpp using mock types, verifying correctness without heavy
// Perfetto/database/AMD-SMI dependencies.
//

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

// ──────────────────────────────────────────────────────────────────
// Mock types mirroring the production types in
//   library/pmc/collectors/gpu/types.hpp
// These are self-contained to avoid AMD SMI SDK dependency.
// ──────────────────────────────────────────────────────────────────

constexpr size_t MAX_NUM_VCN  = 4;
constexpr size_t MAX_NUM_JPEG = 32;
constexpr size_t MAX_NUM_XCP  = 8;

union mock_enabled_metrics
{
    struct
    {
        uint32_t current_socket_power : 1;  // Bit 0
        uint32_t average_socket_power : 1;  // Bit 1
        uint32_t memory_usage         : 1;  // Bit 2
        uint32_t hotspot_temperature  : 1;  // Bit 3
        uint32_t edge_temperature     : 1;  // Bit 4
        uint32_t gfx_activity         : 1;  // Bit 5
        uint32_t umc_activity         : 1;  // Bit 6
        uint32_t mm_activity          : 1;  // Bit 7
        uint32_t vcn_activity         : 1;  // Bit 8  - Device-level VCN (Radeon)
        uint32_t jpeg_activity        : 1;  // Bit 9  - Device-level JPEG (Radeon)
        uint32_t vcn_busy             : 1;  // Bit 10 - Per-XCP VCN (MI300)
        uint32_t jpeg_busy            : 1;  // Bit 11 - Per-XCP JPEG (MI300)
        uint32_t xgmi                 : 1;  // Bit 12
        uint32_t pcie                 : 1;  // Bit 13
        uint32_t sdma_usage           : 1;  // Bit 14
    } bits;
    uint32_t value = 0;
};

struct mock_xcp_metrics
{
    std::array<uint16_t, MAX_NUM_JPEG> jpeg_busy;
    std::array<uint16_t, MAX_NUM_VCN>  vcn_busy;
};

struct mock_metrics
{
    std::array<mock_xcp_metrics, MAX_NUM_XCP> xcp_stats;
    std::array<uint16_t, MAX_NUM_VCN>         vcn_activity  = {};
    std::array<uint16_t, MAX_NUM_JPEG>        jpeg_activity = {};
};

// ──────────────────────────────────────────────────────────────────
// Logic under test — replicated from production code
// ──────────────────────────────────────────────────────────────────

struct rocpd_xcp_entry
{
    std::string pmc_name;
    std::string track_name;
    double      value;
};

// Mirrors insert_xcp_metrics lambda in rocpd_processor.cpp (lines 372-387)
template <typename GetArrayFn>
std::vector<rocpd_xcp_entry>
generate_xcp_metrics(const char* base_name, const std::string& base_track,
                     bool is_enabled, const mock_metrics& m, GetArrayFn&& get_array)
{
    std::vector<rocpd_xcp_entry> entries;
    if(!is_enabled) return entries;
    for(size_t xcp = 0; xcp < m.xcp_stats.size(); ++xcp)
    {
        const auto& arr = get_array(m.xcp_stats[xcp]);
        for(size_t i = 0; i < arr.size(); ++i)
        {
            auto suffix   = "_xcp" + std::to_string(xcp) + "[" + std::to_string(i) + "]";
            auto pmc_name = std::string(base_name) + suffix;
            auto track_name = base_track + suffix;
            entries.push_back({ pmc_name, track_name, static_cast<double>(arr[i]) });
        }
    }
    return entries;
}

// Mirrors insert_device_level_metrics lambda in rocpd_processor.cpp (lines 398-411)
template <typename ArrayT>
std::vector<rocpd_xcp_entry>
generate_device_level_metrics(const std::string& base_name, bool is_enabled,
                              const ArrayT& arr)
{
    std::vector<rocpd_xcp_entry> entries;
    if(!is_enabled) return entries;
    for(size_t i = 0; i < arr.size(); ++i)
    {
        auto suffix     = "_" + std::to_string(i);
        auto pmc_name   = base_name + suffix;
        auto track_name = pmc_name;
        entries.push_back({ pmc_name, track_name, static_cast<double>(arr[i]) });
    }
    return entries;
}

// Mirrors addendum_blk lambda in perfetto_policy.hpp (lines 173-181)
std::string
format_perfetto_xcp_track(uint32_t device_id, const char* metric_name, size_t xcp_idx,
                          size_t engine_idx)
{
    return fmt::format("GPU [{}] {} XCP_{}: [{:02}] (S)", device_id, metric_name, xcp_idx,
                       engine_idx);
}

std::string
format_perfetto_device_track(uint32_t device_id, const char* metric_name,
                             size_t engine_idx)
{
    return fmt::format("GPU [{}] {} [{:02}] (S)", device_id, metric_name, engine_idx);
}

// Mirrors unique_key computation in emit_xcp_array_metrics
// (perfetto_processor.cpp:173-175)
uint64_t
compute_track_key(uint32_t device_id, std::optional<size_t> xcp_idx, size_t engine_idx)
{
    return (static_cast<uint64_t>(device_id) << 16) |
           (static_cast<uint64_t>(xcp_idx.value_or(0)) << 8) |
           static_cast<uint64_t>(engine_idx);
}

mock_metrics
make_sentinel_metrics()
{
    mock_metrics m{};
    for(auto& xcp : m.xcp_stats)
    {
        xcp.vcn_busy.fill(std::numeric_limits<uint16_t>::max());
        xcp.jpeg_busy.fill(std::numeric_limits<uint16_t>::max());
    }
    m.vcn_activity.fill(std::numeric_limits<uint16_t>::max());
    m.jpeg_activity.fill(std::numeric_limits<uint16_t>::max());
    return m;
}

}  // namespace

class xcp_output_test : public ::testing::Test
{
protected:
    void SetUp() override { m = make_sentinel_metrics(); }

    mock_metrics m;
};

// vcn_busy (bit 10) and vcn_activity (bit 8) are independent, mutually exclusive in
// practice
TEST_F(xcp_output_test, EnabledMetricsBitfieldSemantics)
{
    mock_enabled_metrics em{};

    em.bits.vcn_busy = 1;
    EXPECT_EQ(em.value & (1u << 10), (1u << 10));
    EXPECT_EQ(em.bits.vcn_activity, 0u);

    em.value             = 0;
    em.bits.vcn_activity = 1;
    EXPECT_EQ(em.value & (1u << 8), (1u << 8));
    EXPECT_EQ(em.bits.vcn_busy, 0u);

    em.bits.jpeg_busy = 1;
    EXPECT_EQ(em.value & (1u << 11), (1u << 11));
    EXPECT_EQ(em.bits.jpeg_activity, 0u);

    em.value              = 0;
    em.bits.jpeg_activity = 1;
    EXPECT_EQ(em.value & (1u << 9), (1u << 9));
    EXPECT_EQ(em.bits.jpeg_busy, 0u);
}

// With vcn_busy=1, names follow format "device_vcn_activity_xcp{N}[{IDX}]"
TEST_F(xcp_output_test, VcnBusyXcpMetricNaming)
{
    m.xcp_stats[0].vcn_busy[0] = 50;
    m.xcp_stats[0].vcn_busy[1] = 60;
    m.xcp_stats[2].vcn_busy[3] = 80;

    auto entries = generate_xcp_metrics(
        "device_vcn_activity", "device_vcn_activity", true, m,
        [](const mock_xcp_metrics& xcp) -> const auto& { return xcp.vcn_busy; });

    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(entries.size(), MAX_NUM_XCP * MAX_NUM_VCN);

    EXPECT_EQ(entries[0].pmc_name, "device_vcn_activity_xcp0[0]");
    EXPECT_DOUBLE_EQ(entries[0].value, 50.0);

    EXPECT_EQ(entries[1].pmc_name, "device_vcn_activity_xcp0[1]");
    EXPECT_DOUBLE_EQ(entries[1].value, 60.0);

    // xcp2, engine 3 → index 2*4+3 = 11
    EXPECT_EQ(entries[11].pmc_name, "device_vcn_activity_xcp2[3]");
    EXPECT_DOUBLE_EQ(entries[11].value, 80.0);
}

// With jpeg_busy=1, names follow format "device_jpeg_activity_xcp{N}[{IDX}]"
TEST_F(xcp_output_test, JpegBusyXcpMetricNaming)
{
    m.xcp_stats[1].jpeg_busy[0] = 42;

    auto entries = generate_xcp_metrics(
        "device_jpeg_activity", "device_jpeg_activity", true, m,
        [](const mock_xcp_metrics& xcp) -> const auto& { return xcp.jpeg_busy; });

    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(entries.size(), MAX_NUM_XCP * MAX_NUM_JPEG);

    // xcp1, engine 0 → index 1 * JPEG_COUNT + 0
    size_t idx = 1 * MAX_NUM_JPEG;
    EXPECT_EQ(entries[idx].pmc_name, "device_jpeg_activity_xcp1[0]");
    EXPECT_DOUBLE_EQ(entries[idx].value, 42.0);
}

// vcn_busy=0 → no per-XCP VCN metrics generated
TEST_F(xcp_output_test, DisabledVcnBusyProducesNoOutput)
{
    m.xcp_stats[0].vcn_busy[0] = 50;

    auto entries = generate_xcp_metrics(
        "device_vcn_activity", "device_vcn_activity", false, m,
        [](const mock_xcp_metrics& xcp) -> const auto& { return xcp.vcn_busy; });

    EXPECT_TRUE(entries.empty());
}

// jpeg_busy=0 → no per-XCP JPEG metrics generated
TEST_F(xcp_output_test, DisabledJpegBusyProducesNoOutput)
{
    m.xcp_stats[0].jpeg_busy[0] = 50;

    auto entries = generate_xcp_metrics(
        "device_jpeg_activity", "device_jpeg_activity", false, m,
        [](const mock_xcp_metrics& xcp) -> const auto& { return xcp.jpeg_busy; });

    EXPECT_TRUE(entries.empty());
}

// Valid VCN values across all 8 XCPs → 8*4=32 entries
TEST_F(xcp_output_test, AllXcpPartitionsWritten)
{
    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t eng = 0; eng < MAX_NUM_VCN; ++eng)
        {
            m.xcp_stats[xcp].vcn_busy[eng] = static_cast<uint16_t>(xcp * 10 + eng);
        }
    }

    auto entries = generate_xcp_metrics(
        "device_vcn_activity", "device_vcn_activity", true, m,
        [](const mock_xcp_metrics& xcp) -> const auto& { return xcp.vcn_busy; });

    EXPECT_EQ(entries.size(), MAX_NUM_XCP * MAX_NUM_VCN);

    for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        for(size_t eng = 0; eng < MAX_NUM_VCN; ++eng)
        {
            size_t idx           = xcp * MAX_NUM_VCN + eng;
            auto   expected_name = "device_vcn_activity_xcp" + std::to_string(xcp) + "[" +
                                 std::to_string(eng) + "]";
            EXPECT_EQ(entries[idx].pmc_name, expected_name)
                << "Mismatch at xcp=" << xcp << " eng=" << eng;
            EXPECT_DOUBLE_EQ(entries[idx].value, static_cast<double>(xcp * 10 + eng));
        }
    }
}

// vcn_activity=1 uses device-level array, no _xcp prefix
TEST_F(xcp_output_test, DeviceLevelVcnActivitySeparateFromXcp)
{
    m.vcn_activity[0] = 75;
    m.vcn_activity[1] = 85;

    auto entries =
        generate_device_level_metrics("device_vcn_activity", true, m.vcn_activity);

    EXPECT_EQ(entries.size(), m.vcn_activity.size());

    EXPECT_EQ(entries[0].pmc_name, "device_vcn_activity_0");
    EXPECT_DOUBLE_EQ(entries[0].value, 75.0);
    EXPECT_EQ(entries[1].pmc_name, "device_vcn_activity_1");
    EXPECT_DOUBLE_EQ(entries[1].value, 85.0);

    for(const auto& entry : entries)
    {
        EXPECT_EQ(entry.pmc_name.find("_xcp"), std::string::npos)
            << "Device-level name should not contain _xcp: " << entry.pmc_name;
    }
}

// Track names follow "GPU [{id}] VCN Busy XCP_{xcp}: [{eng:02}] (S)"
TEST_F(xcp_output_test, PerfettoXcpTrackNameFormat)
{
    uint32_t device_id = 0;

    auto vcn_name = format_perfetto_xcp_track(device_id, "VCN Busy", 3, 2);
    EXPECT_EQ(vcn_name, "GPU [0] VCN Busy XCP_3: [02] (S)");

    auto jpeg_name = format_perfetto_xcp_track(device_id, "JPEG Busy", 7, 0);
    EXPECT_EQ(jpeg_name, "GPU [0] JPEG Busy XCP_7: [00] (S)");

    auto dev_vcn = format_perfetto_device_track(device_id, "VCN Activity", 1);
    EXPECT_EQ(dev_vcn, "GPU [0] VCN Activity [01] (S)");

    auto dev_jpeg = format_perfetto_device_track(device_id, "JPEG Activity", 3);
    EXPECT_EQ(dev_jpeg, "GPU [0] JPEG Activity [03] (S)");

    auto multi_dev = format_perfetto_xcp_track(5, "VCN Busy", 0, 0);
    EXPECT_EQ(multi_dev, "GPU [5] VCN Busy XCP_0: [00] (S)");
}

// Track key uniqueness for emit_xcp_array_metrics
TEST_F(xcp_output_test, PerfettoTrackKeyUniqueness)
{
    auto key_0_0_0 = compute_track_key(0, 0, 0);
    auto key_0_0_1 = compute_track_key(0, 0, 1);
    auto key_0_1_0 = compute_track_key(0, 1, 0);
    auto key_1_0_0 = compute_track_key(1, 0, 0);

    EXPECT_NE(key_0_0_0, key_0_0_1);
    EXPECT_NE(key_0_0_0, key_0_1_0);
    EXPECT_NE(key_0_0_0, key_1_0_0);

    // Device-level (nullopt→0) and per-XCP with xcp=0 produce the same key —
    // separation is handled by using different Track type instantiations
    auto device_key = compute_track_key(0, std::nullopt, 0);
    auto xcp0_key   = compute_track_key(0, 0, 0);
    EXPECT_EQ(device_key, xcp0_key);
}

// Sentinel values (0xFFFF) are skipped in Perfetto output
TEST_F(xcp_output_test, SentinelValuesSkipped)
{
    std::vector<std::pair<std::string, double>> emitted;
    uint32_t                                    device_id = 0;

    // All sentinel by default from make_sentinel_metrics()
    for(size_t xcp = 0; xcp < m.xcp_stats.size(); ++xcp)
    {
        for(size_t i = 0; i < m.xcp_stats[xcp].vcn_busy.size(); ++i)
        {
            auto value = m.xcp_stats[xcp].vcn_busy[i];
            if(value == std::numeric_limits<uint16_t>::max()) continue;
            emitted.emplace_back(format_perfetto_xcp_track(device_id, "VCN Busy", xcp, i),
                                 static_cast<double>(value));
        }
    }
    EXPECT_TRUE(emitted.empty()) << "All sentinel values should be skipped";

    // Set one valid value
    m.xcp_stats[0].vcn_busy[0] = 42;
    emitted.clear();

    for(size_t xcp = 0; xcp < m.xcp_stats.size(); ++xcp)
    {
        for(size_t i = 0; i < m.xcp_stats[xcp].vcn_busy.size(); ++i)
        {
            auto value = m.xcp_stats[xcp].vcn_busy[i];
            if(value == std::numeric_limits<uint16_t>::max()) continue;
            emitted.emplace_back(format_perfetto_xcp_track(device_id, "VCN Busy", xcp, i),
                                 static_cast<double>(value));
        }
    }

    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_EQ(emitted[0].first, "GPU [0] VCN Busy XCP_0: [00] (S)");
    EXPECT_DOUBLE_EQ(emitted[0].second, 42.0);
}
