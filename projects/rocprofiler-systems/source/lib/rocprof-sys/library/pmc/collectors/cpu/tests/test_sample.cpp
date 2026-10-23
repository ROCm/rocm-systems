// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "core/trace_cache/archive.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace rocprofsys::trace_cache;

namespace
{
constexpr auto          k_all_metrics      = 0x1FFU;  // all 9 bits
constexpr std::int64_t  k_page_rss         = 1024;
constexpr std::int64_t  k_virt_mem         = 2048;
constexpr std::int64_t  k_peak_rss         = 4096;
constexpr std::int64_t  k_context_switches = 500;
constexpr std::int64_t  k_page_faults      = 100;
constexpr std::int64_t  k_user_mode_time   = 1000000;
constexpr std::int64_t  k_kernel_mode_time = 500000;
constexpr std::uint64_t k_timestamp        = 80000;
}  // namespace

class cpu_pmc_sample_test : public ::testing::Test
{
protected:
    void SetUp() override { buffer.fill(0); }

    std::array<std::uint8_t, 4096> buffer;
};

TEST_F(cpu_pmc_sample_test, serialize_deserialize)
{
    using namespace rocprofsys::pmc::collectors::cpu;

    enabled_metrics enabled{};
    enabled.value = k_all_metrics;

    process_metrics process{};
    process.page_rss         = k_page_rss;
    process.virt_mem         = k_virt_mem;
    process.peak_rss         = k_peak_rss;
    process.context_switches = k_context_switches;
    process.page_faults      = k_page_faults;
    process.user_mode_time   = k_user_mode_time;
    process.kernel_mode_time = k_kernel_mode_time;

    const std::vector<core_frequency> freqs_data = { { .cpu_id = 0, .value = 3200.0f },
                                                     { .cpu_id = 1, .value = 2800.5f } };
    const std::vector<core_load>      loads_data = { { .cpu_id = 0, .value = 12.5 },
                                                     { .cpu_id = 1, .value = 87.25 } };
    const cpu_pmc_sample original(enabled, 1u, k_timestamp, process, freqs_data,
                                  loads_data);

    serialize(buffer.data(), original);

    std::uint8_t* buffer_ptr   = buffer.data();
    auto const    deserialized = deserialize<cpu_pmc_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.device_id, original.device_id);
    EXPECT_EQ(deserialized.timestamp, original.timestamp);
    EXPECT_EQ(deserialized.process_data.page_rss, original.process_data.page_rss);
    EXPECT_EQ(deserialized.process_data.virt_mem, original.process_data.virt_mem);
    EXPECT_EQ(deserialized.process_data.peak_rss, original.process_data.peak_rss);
    EXPECT_EQ(deserialized.process_data.context_switches,
              original.process_data.context_switches);
    EXPECT_EQ(deserialized.process_data.page_faults, original.process_data.page_faults);
    EXPECT_EQ(deserialized.process_data.user_mode_time,
              original.process_data.user_mode_time);
    EXPECT_EQ(deserialized.process_data.kernel_mode_time,
              original.process_data.kernel_mode_time);
    EXPECT_EQ(deserialized.freqs, original.freqs);
    EXPECT_EQ(deserialized.loads, original.loads);
}

TEST_F(cpu_pmc_sample_test, get_size)
{
    using namespace rocprofsys::pmc::collectors::cpu;

    const enabled_metrics enabled{};
    const process_metrics process{};

    const std::vector<core_frequency> freqs_data = { { .cpu_id = 0, .value = 3200.0f },
                                                     { .cpu_id = 1, .value = 2800.5f } };
    const std::vector<core_load>      loads_data = { { .cpu_id = 0, .value = 12.5 },
                                                     { .cpu_id = 1, .value = 87.25 } };
    const cpu_pmc_sample record(enabled, 0u, k_timestamp, process, freqs_data,
                                loads_data);

    EXPECT_GT(get_size(record), 0u);
}

TEST_F(cpu_pmc_sample_test, type_identifier)
{
    EXPECT_EQ(cpu_pmc_sample::type_identifier, type_identifier_t::cpu_pmc_sample);
}

TEST_F(cpu_pmc_sample_test, empty_data)
{
    using namespace rocprofsys::pmc::collectors::cpu;

    const enabled_metrics             enabled{};
    const process_metrics             process{};
    const std::vector<core_frequency> empty_freqs;
    const std::vector<core_load>      empty_loads;
    const cpu_pmc_sample original(enabled, 0u, 0, process, empty_freqs, empty_loads);

    serialize(buffer.data(), original);

    std::uint8_t* buffer_ptr   = buffer.data();
    auto const    deserialized = deserialize<cpu_pmc_sample>(buffer_ptr);

    EXPECT_TRUE(deserialized.freqs.empty());
    EXPECT_TRUE(deserialized.loads.empty());
}

TEST_F(cpu_pmc_sample_test, default_constructor)
{
    const cpu_pmc_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::cpu_pmc_sample);
}
