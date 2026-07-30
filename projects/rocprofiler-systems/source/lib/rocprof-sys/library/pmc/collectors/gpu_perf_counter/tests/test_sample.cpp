// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace rocprofsys::pmc::collectors::gpu_perf_counter;
using namespace rocprofsys::trace_cache;

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{

class SdkPmcSampleTest : public ::testing::Test
{
protected:
    void SetUp() override { buffer.fill(0); }

    std::array<std::uint8_t, 8192> buffer{};
};

TEST_F(SdkPmcSampleTest, EmptySampleRoundTrip)
{
    sample original{ 0, 1000, {} };

    serialize(buffer.data(), original);

    std::uint8_t* ptr          = buffer.data();
    auto          deserialized = deserialize<sample>(ptr);

    EXPECT_EQ(deserialized.device_id, 0U);
    EXPECT_EQ(deserialized.timestamp, 1000U);
    EXPECT_TRUE(deserialized.entries.empty());
}

TEST_F(SdkPmcSampleTest, SingleEntryRoundTrip)
{
    sample original{ 2, 5000, { counter_value{ 10, 42.0 } } };

    serialize(buffer.data(), original);

    std::uint8_t* ptr          = buffer.data();
    auto          deserialized = deserialize<sample>(ptr);

    EXPECT_EQ(deserialized.device_id, 2U);
    EXPECT_EQ(deserialized.timestamp, 5000U);
    ASSERT_EQ(deserialized.entries.size(), 1U);
    EXPECT_EQ(deserialized.entries[0].counter_id, 10U);
    EXPECT_DOUBLE_EQ(deserialized.entries[0].value, 42.0);
}

TEST_F(SdkPmcSampleTest, MultiEntryRoundTrip)
{
    sample original{ 1,
                     99000,
                     { counter_value{ 100, 10.0 }, counter_value{ 101, 20.0 },
                       counter_value{ 102, 30.0 } } };

    serialize(buffer.data(), original);

    std::uint8_t* ptr          = buffer.data();
    auto          deserialized = deserialize<sample>(ptr);

    EXPECT_EQ(deserialized.device_id, 1U);
    EXPECT_EQ(deserialized.timestamp, 99000U);
    ASSERT_EQ(deserialized.entries.size(), 3U);

    EXPECT_EQ(deserialized.entries[0].counter_id, 100U);
    EXPECT_DOUBLE_EQ(deserialized.entries[0].value, 10.0);

    EXPECT_EQ(deserialized.entries[1].counter_id, 101U);
    EXPECT_DOUBLE_EQ(deserialized.entries[1].value, 20.0);

    EXPECT_EQ(deserialized.entries[2].counter_id, 102U);
    EXPECT_DOUBLE_EQ(deserialized.entries[2].value, 30.0);
}

TEST_F(SdkPmcSampleTest, GetSizeEmpty)
{
    sample test_sample{ 0, 0, {} };

    // device_id(4) + timestamp(8) + num_entries(4) = 16
    size_t expected =
        sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(std::uint32_t);
    EXPECT_EQ(get_size(test_sample), expected);
}

TEST_F(SdkPmcSampleTest, GetSizeSingleEntry)
{
    sample test_sample{ 0, 0, { counter_value{ 10, 1.0 } } };

    // header: 4 + 8 + 4 = 16
    // entry: counter_id(8) + double(8) = 16
    size_t expected = 16 + sizeof(counter_id_t) + sizeof(double);
    EXPECT_EQ(get_size(test_sample), expected);
}

TEST_F(SdkPmcSampleTest, GetSizeMatchesSerializedBytes)
{
    sample original{ 3, 42000, { counter_value{ 1, 1.0 }, counter_value{ 2, 99.5 } } };

    size_t computed_size = get_size(original);
    serialize(buffer.data(), original);

    std::uint8_t* ptr = buffer.data();
    deserialize<sample>(ptr);
    size_t bytes_consumed = static_cast<size_t>(ptr - buffer.data());

    EXPECT_EQ(bytes_consumed, computed_size);
}

TEST_F(SdkPmcSampleTest, DeserializePreservesBufferPointerAdvancement)
{
    sample first{ 0, 1000, { counter_value{ 1, 1.0 } } };
    sample second{ 1, 2000, { counter_value{ 2, 2.0 }, counter_value{ 3, 3.0 } } };

    size_t first_size = get_size(first);
    serialize(buffer.data(), first);
    serialize(buffer.data() + first_size, second);

    std::uint8_t* ptr = buffer.data();

    auto first_result = deserialize<sample>(ptr);
    EXPECT_EQ(first_result.device_id, 0U);
    ASSERT_EQ(first_result.entries.size(), 1U);
    EXPECT_EQ(first_result.entries[0].counter_id, 1U);

    auto second_result = deserialize<sample>(ptr);
    EXPECT_EQ(second_result.device_id, 1U);
    ASSERT_EQ(second_result.entries.size(), 2U);
    EXPECT_EQ(second_result.entries[0].counter_id, 2U);
    EXPECT_EQ(second_result.entries[1].counter_id, 3U);
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
