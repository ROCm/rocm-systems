// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// NOLINTBEGIN(readability-magic-numbers)
// Test values in this file are arbitrary and chosen for coverage, not domain significance

#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cacheable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <string_view>
#include <vector>

namespace rocprofsys::trace_cache
{

namespace
{

enum class test_record_id : std::uint32_t
{
    scalar_test   = 1,
    string_test   = 2,
    vector_test   = 3,
    optional_test = 4,
    derived_test  = 6,
    nested_test   = 8,
};

// A record counts as serializable only when it declares its own serialize(Archive&).
struct base_with_serialize
{
    std::uint32_t first = 0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(first);
    }
};

struct derived_without_serialize : base_with_serialize
{
    std::uint32_t second = 0;
};

struct serialize_takes_archive_by_value
{
    std::uint32_t first = 0;

    template <typename Archive>
    void serialize(Archive archive)
    {
        archive(first);
    }
};

static_assert(archive_serializable<base_with_serialize, binary_size_archive>);
static_assert(archive_serializable<base_with_serialize, binary_write_archive>);
static_assert(archive_serializable<base_with_serialize, binary_read_archive>);
static_assert(!archive_serializable<derived_without_serialize, binary_size_archive>,
              "an inherited serialize would skip the fields of the derived record");
static_assert(
    !archive_serializable<serialize_takes_archive_by_value, binary_size_archive>,
    "a serialize that takes the archive by value loses the position");
static_assert(!archive_serializable<std::uint32_t, binary_size_archive>);

struct scalar_record
{
    static constexpr auto type_identifier = test_record_id::scalar_test;

    std::uint32_t val_u32 = 0;
    std::uint64_t val_u64 = 0;
    double        val_dbl = 0.0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(val_u32, val_u64, val_dbl);
    }
};

struct string_record
{
    static constexpr auto type_identifier = test_record_id::string_test;

    std::string_view text;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(text);
    }
};

struct vector_record
{
    static constexpr auto type_identifier = test_record_id::vector_test;

    std::vector<std::uint8_t> data;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(data);
    }
};

struct optional_record
{
    static constexpr auto type_identifier = test_record_id::optional_test;

    std::optional<std::int64_t> value;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(value);
    }
};

struct base_record
{
    static constexpr auto type_identifier = test_record_id::derived_test;

    std::uint32_t base_field = 0;
    double        base_value = 0.0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(base_field, base_value);
    }
};

struct derived_record : base_record
{
    static constexpr auto type_identifier = test_record_id::derived_test;

    std::uint64_t extra_field = 0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        base_record::serialize(archive);
        archive(extra_field);
    }
};

struct inner_values
{
    std::uint32_t alpha = 0;
    double        beta  = 0.0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(alpha, beta);
    }
};

struct nested_record
{
    static constexpr auto type_identifier = test_record_id::nested_test;

    std::uint64_t head = 0;
    inner_values  inner;
    std::uint16_t tail = 0;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(head, inner, tail);
    }
};

}  // namespace

constexpr std::size_t k_test_buffer_size = 4096;

TEST(archive, scalars_match_utility_encoding)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    const std::uint32_t                          val_u32 = 12345U;
    const std::uint64_t                          val_u64 = 98765432101234ULL;
    const double                                 val_dbl = 2.71828;

    scalar_record orig;
    orig.val_u32 = val_u32;
    orig.val_u64 = val_u64;
    orig.val_dbl = val_dbl;

    const std::size_t size = get_size(orig);
    serialize(buffer.data(), orig);

    std::uint8_t*       read_buffer = buffer.data();
    const scalar_record read        = deserialize<scalar_record>(read_buffer);

    EXPECT_EQ(read.val_u32, val_u32);
    EXPECT_EQ(read.val_u64, val_u64);
    EXPECT_DOUBLE_EQ(read.val_dbl, val_dbl);
    EXPECT_EQ(read_buffer - buffer.data(), static_cast<std::ptrdiff_t>(size));

    EXPECT_EQ(size, utility::get_size(val_u32) + utility::get_size(val_u64) +
                        utility::get_size(val_dbl));

    std::array<std::uint8_t, k_test_buffer_size> other_buffer{};
    std::size_t                                  utility_position = 0;
    utility::store_value(val_u32, other_buffer.data(), utility_position);
    utility::store_value(val_u64, other_buffer.data(), utility_position);
    utility::store_value(val_dbl, other_buffer.data(), utility_position);

    EXPECT_EQ(std::memcmp(buffer.data(), other_buffer.data(), size), 0);
}

TEST(archive, string_view_reads_back_as_view_into_buffer)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    const std::string_view                       orig_text = "hello world";

    string_record orig;
    orig.text = orig_text;

    const std::size_t size = get_size(orig);
    serialize(buffer.data(), orig);

    std::uint8_t*       read_buffer = buffer.data();
    const string_record read        = deserialize<string_record>(read_buffer);

    EXPECT_EQ(read.text, orig_text);
    EXPECT_EQ(read.text.data(),
              reinterpret_cast<const char*>(buffer.data() + sizeof(std::size_t)));
    EXPECT_EQ(read_buffer - buffer.data(), static_cast<std::ptrdiff_t>(size));
}

TEST(archive, vector_round_trip)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    const std::vector<std::uint8_t>              orig_data = { 10, 20, 30 };

    vector_record orig;
    orig.data = orig_data;

    const std::size_t size = get_size(orig);
    serialize(buffer.data(), orig);

    std::uint8_t*       read_buffer = buffer.data();
    const vector_record read        = deserialize<vector_record>(read_buffer);

    EXPECT_EQ(read.data, orig_data);
    EXPECT_EQ(read_buffer - buffer.data(), static_cast<std::ptrdiff_t>(size));
}

TEST(archive, optional_engaged_and_empty)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    optional_record                              orig_engaged;
    orig_engaged.value = 9876543210LL;

    const std::size_t size_engaged = get_size(orig_engaged);
    serialize(buffer.data(), orig_engaged);

    std::uint8_t*         read_buf     = buffer.data();
    const optional_record read_engaged = deserialize<optional_record>(read_buf);

    ASSERT_TRUE(read_engaged.value.has_value());
    EXPECT_EQ(read_engaged.value, orig_engaged.value);

    buffer.fill(0);
    optional_record orig_empty;
    orig_empty.value.reset();

    const std::size_t size_empty = get_size(orig_empty);
    serialize(buffer.data(), orig_empty);

    read_buf                         = buffer.data();
    const optional_record read_empty = deserialize<optional_record>(read_buf);

    EXPECT_FALSE(read_empty.value.has_value());

    EXPECT_EQ(size_engaged, size_empty + sizeof(std::int64_t));
}

TEST(archive, derived_record_serializes_base_first)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    derived_record                               orig;
    orig.base_field  = 555U;
    orig.base_value  = 3.14;
    orig.extra_field = 777ULL;

    const std::size_t base_size =
        get_size(base_record{ .base_field = 555U, .base_value = 3.14 });
    const std::size_t total_size = get_size(orig);

    serialize(buffer.data(), orig);

    std::uint8_t*        read_buffer = buffer.data();
    const derived_record read        = deserialize<derived_record>(read_buffer);

    EXPECT_EQ(read.base_field, 555U);
    EXPECT_DOUBLE_EQ(read.base_value, 3.14);
    EXPECT_EQ(read.extra_field, 777ULL);

    EXPECT_EQ(total_size, base_size + sizeof(std::uint64_t));
    EXPECT_EQ(read_buffer - buffer.data(), static_cast<std::ptrdiff_t>(total_size));
}

TEST(archive, nested_struct_serializes_itself)
{
    std::array<std::uint8_t, k_test_buffer_size> buffer{};
    nested_record                                orig;
    orig.head        = 1111ULL;
    orig.inner.alpha = 2222U;
    orig.inner.beta  = 3.33;
    orig.tail        = 4444U;

    const std::size_t size = get_size(orig);
    serialize(buffer.data(), orig);

    std::uint8_t*       read_buffer = buffer.data();
    const nested_record read        = deserialize<nested_record>(read_buffer);

    EXPECT_EQ(read.head, 1111ULL);
    EXPECT_EQ(read.inner.alpha, 2222U);
    EXPECT_DOUBLE_EQ(read.inner.beta, 3.33);
    EXPECT_EQ(read.tail, 4444U);
    EXPECT_EQ(read_buffer - buffer.data(), static_cast<std::ptrdiff_t>(size));

    EXPECT_EQ(size, utility::get_size(orig.head) + utility::get_size(orig.inner.alpha) +
                        utility::get_size(orig.inner.beta) +
                        utility::get_size(orig.tail));

    std::array<std::uint8_t, k_test_buffer_size> other_buffer{};
    std::size_t                                  nested_position = 0;
    utility::store_value(orig.head, other_buffer.data(), nested_position);
    utility::store_value(orig.inner.alpha, other_buffer.data(), nested_position);
    utility::store_value(orig.inner.beta, other_buffer.data(), nested_position);
    utility::store_value(orig.tail, other_buffer.data(), nested_position);

    EXPECT_EQ(std::memcmp(buffer.data(), other_buffer.data(), size), 0);
}

}  // namespace rocprofsys::trace_cache

// NOLINTEND(readability-magic-numbers)
