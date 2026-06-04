// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/cpu/sample.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace rocprofsys::trace_cache;

class cpu_pmc_sample_test : public ::testing::Test
{
protected:
    void SetUp() override { buffer.fill(0); }

    std::array<std::uint8_t, 4096> buffer;
};

TEST_F(cpu_pmc_sample_test, serialize_deserialize)
{
    using namespace rocprofsys::pmc::collectors::cpu;

    enabled_metrics em{};
    em.value = 0x1FF;  // all 9 bits

    process_metrics pm{};
    pm.page_rss         = 1024;
    pm.virt_mem         = 2048;
    pm.peak_rss         = 4096;
    pm.context_switches = 500;
    pm.page_faults      = 100;
    pm.user_mode_time   = 1000000;
    pm.kernel_mode_time = 500000;

    std::vector<std::uint8_t> freqs_data = { 100, 150, 200, 180, 190, 195, 185, 170 };
    std::vector<std::uint8_t> loads_data = { 10, 20, 30, 40 };
    cpu_pmc_sample            original(em, 1u, 80000, pm, freqs_data, loads_data);

    serialize(buffer.data(), original);

    std::uint8_t* buffer_ptr   = buffer.data();
    auto          deserialized = deserialize<cpu_pmc_sample>(buffer_ptr);

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

    enabled_metrics em{};
    process_metrics pm{};

    std::vector<std::uint8_t> freqs_data = { 100, 150, 200, 180, 190, 195, 185, 170 };
    std::vector<std::uint8_t> loads_data = { 10, 20, 30, 40 };
    cpu_pmc_sample            s(em, 0u, 80000, pm, freqs_data, loads_data);

    EXPECT_GT(get_size(s), 0u);
}

TEST_F(cpu_pmc_sample_test, type_identifier)
{
    EXPECT_EQ(cpu_pmc_sample::type_identifier, type_identifier_t::cpu_pmc_sample);
}

TEST_F(cpu_pmc_sample_test, empty_data)
{
    using namespace rocprofsys::pmc::collectors::cpu;

    enabled_metrics           em{};
    process_metrics           pm{};
    std::vector<std::uint8_t> empty;
    cpu_pmc_sample            original(em, 0u, 0, pm, empty, empty);

    serialize(buffer.data(), original);

    std::uint8_t* buffer_ptr   = buffer.data();
    auto          deserialized = deserialize<cpu_pmc_sample>(buffer_ptr);

    EXPECT_TRUE(deserialized.freqs.empty());
    EXPECT_TRUE(deserialized.loads.empty());
}

TEST_F(cpu_pmc_sample_test, default_constructor)
{
    cpu_pmc_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::cpu_pmc_sample);
}
