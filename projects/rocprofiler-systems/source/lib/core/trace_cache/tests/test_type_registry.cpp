// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/type_registry.hpp"
#include "mocked_types.hpp"
#include <cstdint>

#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(rocprofsys::trace_cache::detail::are_unique<1, 2, 3>());
static_assert(!rocprofsys::trace_cache::detail::are_unique<1, 2, 1>());

namespace
{
// Serializes `value`, dispatches it under `type_id` and calls `check` with the record
// the visitor received, while the serialized buffer is still alive (string_view members
// point into it). Returns whether the visitor saw a `Sample`.
template <typename Sample, typename Registry, typename Check>
bool
dispatch_round_trip(const Registry& registry, test_type_identifier_t type_id,
                    const Sample& value, Check&& check)
{
    std::vector<std::uint8_t> buffer(rocprofsys::trace_cache::get_size(value));
    rocprofsys::trace_cache::serialize(buffer.data(), value);

    auto* data     = buffer.data();
    bool  received = false;
    registry.dispatch(type_id, data, [&](const auto& sample) {
        if constexpr(std::is_same_v<Sample, std::decay_t<decltype(sample)>>)
        {
            received = true;
            check(sample);
        }
    });
    return received;
}
}  // namespace

class type_registry_test : public ::testing::Test
{
protected:
    rocprofsys::trace_cache::type_registry<test_type_identifier_t, test_sample_1,
                                           test_sample_2>
        type_registry;
};

TEST_F(type_registry_test, test_dispatch_sample_1)
{
    EXPECT_TRUE(dispatch_round_trip(type_registry, test_type_identifier_t::sample_type_1,
                                    test_sample_1{ 42, "hello" },
                                    [](const test_sample_1& sample) {
                                        EXPECT_EQ(sample.value, 42);
                                        EXPECT_EQ(sample.text, "hello");
                                    }));
}

TEST_F(type_registry_test, test_dispatch_sample_2)
{
    EXPECT_TRUE(dispatch_round_trip(type_registry, test_type_identifier_t::sample_type_2,
                                    test_sample_2{ 3.14, 123 },
                                    [](const test_sample_2& sample) {
                                        EXPECT_DOUBLE_EQ(sample.data, 3.14);
                                        EXPECT_EQ(sample.sample_id, 123);
                                    }));
}

TEST_F(type_registry_test, test_dispatch_calls_visitor_once)
{
    const test_sample_2       test_value{ 3.14, 123 };
    std::vector<std::uint8_t> buffer(rocprofsys::trace_cache::get_size(test_value));
    rocprofsys::trace_cache::serialize(buffer.data(), test_value);

    auto* data  = buffer.data();
    int   calls = 0;
    type_registry.dispatch(test_type_identifier_t::sample_type_2, data,
                           [&](const auto&) { ++calls; });

    EXPECT_EQ(calls, 1);
}

TEST_F(type_registry_test, test_dispatch_unknown_id_throws_without_calling_visitor)
{
    std::uint8_t  dummy_data = 0;
    std::uint8_t* data       = &dummy_data;
    int           calls      = 0;

    EXPECT_THROW(type_registry.dispatch(test_type_identifier_t::fragmented_space, data,
                                        [&](const auto&) { ++calls; }),
                 std::runtime_error);
    EXPECT_EQ(calls, 0);
}

TEST_F(type_registry_test, test_dispatch_multiple_calls_same_type)
{
    EXPECT_TRUE(dispatch_round_trip(type_registry, test_type_identifier_t::sample_type_1,
                                    test_sample_1{ 100, "first" },
                                    [](const test_sample_1& sample) {
                                        EXPECT_EQ(sample.value, 100);
                                        EXPECT_EQ(sample.text, "first");
                                    }));
    EXPECT_TRUE(dispatch_round_trip(type_registry, test_type_identifier_t::sample_type_1,
                                    test_sample_1{ 200, "second" },
                                    [](const test_sample_1& sample) {
                                        EXPECT_EQ(sample.value, 200);
                                        EXPECT_EQ(sample.text, "second");
                                    }));
}

class type_registry_optional_test : public ::testing::Test
{
protected:
    rocprofsys::trace_cache::type_registry<test_type_identifier_t, test_sample_1,
                                           test_sample_2, test_sample_5>
        type_registry;
};

TEST_F(type_registry_optional_test, test_dispatch_sample_5_with_value)
{
    EXPECT_TRUE(dispatch_round_trip(type_registry, test_type_identifier_t::sample_type_5,
                                    test_sample_5{ std::optional<std::uint32_t>{ 42 } },
                                    [](const test_sample_5& sample) {
                                        ASSERT_TRUE(sample.data.has_value());
                                        EXPECT_EQ(sample.data.value(), 42);
                                    }));
}

TEST_F(type_registry_optional_test, test_dispatch_sample_5_nullopt)
{
    EXPECT_TRUE(dispatch_round_trip(
        type_registry, test_type_identifier_t::sample_type_5,
        test_sample_5{ std::nullopt },
        [](const test_sample_5& sample) { EXPECT_EQ(sample.data, std::nullopt); }));
}
