// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

namespace
{

// Round-trip a value through output_archive + input_archive and assert
// equality. Returns the byte count written so callers can also assert the
// wire size matches serialized_size().
template <typename T>
std::size_t
archive_roundtrip(const T& value, T& out)
{
    std::array<std::uint8_t, 4096> buffer{};
    std::uint8_t*                  write_cursor = buffer.data();
    {
        rocprofsys::trace_cache::output_archive oa{ write_cursor };
        oa(value);
    }
    const auto written = static_cast<std::size_t>(write_cursor - buffer.data());

    std::uint8_t* read_cursor = buffer.data();
    {
        rocprofsys::trace_cache::input_archive ia{ read_cursor };
        ia(out);
    }
    EXPECT_EQ(static_cast<std::size_t>(read_cursor - buffer.data()), written)
        << "input_archive consumed a different number of bytes than output_archive wrote";
    return written;
}

// Single-shot helper that asserts equality and wire-size consistency for
// types where operator== is well behaved (default for arithmetic / string /
// vector / optional<arithmetic-or-string-or-vector>).
template <typename T>
void
expect_archive_roundtrip(const T& value)
{
    T          out{};
    const auto written = archive_roundtrip(value, out);
    EXPECT_EQ(out, value);
    EXPECT_EQ(written, rocprofsys::trace_cache::serialized_size(value));
}

}  // namespace

class cacheable_test : public ::testing::Test
{};

// ----- arithmetic ----------------------------------------------------------

TEST_F(cacheable_test, archive_int)
{
    expect_archive_roundtrip<int>(42);
    expect_archive_roundtrip<int>(-1);
    expect_archive_roundtrip<int>(0);
}

TEST_F(cacheable_test, archive_double)
{
    expect_archive_roundtrip<double>(3.14159);
    expect_archive_roundtrip<double>(-1.0);
    expect_archive_roundtrip<double>(0.0);
}

TEST_F(cacheable_test, archive_unsigned_long)
{
    expect_archive_roundtrip<unsigned long>(123456789UL);
}

TEST_F(cacheable_test, archive_unsigned_char)
{
    expect_archive_roundtrip<unsigned char>(255);
}

TEST_F(cacheable_test, archive_uint64)
{
    expect_archive_roundtrip<std::uint64_t>(0xDEADBEEFCAFEBABEULL);
}

TEST_F(cacheable_test, archive_int64_negative)
{
    expect_archive_roundtrip<std::int64_t>(-42);
}

TEST_F(cacheable_test, archive_float) { expect_archive_roundtrip<float>(2.5F); }

// ----- string --------------------------------------------------------------

TEST_F(cacheable_test, archive_string)
{
    expect_archive_roundtrip<std::string>("Hello World");
}

TEST_F(cacheable_test, archive_empty_string)
{
    expect_archive_roundtrip<std::string>("");
}

// ----- vector --------------------------------------------------------------

TEST_F(cacheable_test, archive_vector_int)
{
    expect_archive_roundtrip<std::vector<int>>({ 1, 2, 3, 4, 5 });
}

TEST_F(cacheable_test, archive_vector_uint8)
{
    expect_archive_roundtrip<std::vector<std::uint8_t>>({ 10, 20, 30, 40 });
}

TEST_F(cacheable_test, archive_vector_uint64)
{
    expect_archive_roundtrip<std::vector<std::uint64_t>>(
        { 0x1111ULL, 0x2222ULL, 0x3333ULL });
}

TEST_F(cacheable_test, archive_vector_empty)
{
    expect_archive_roundtrip<std::vector<int>>({});
}

// ----- optional ------------------------------------------------------------

TEST_F(cacheable_test, archive_optional_double_with_value)
{
    expect_archive_roundtrip<std::optional<double>>(42.0);
}

TEST_F(cacheable_test, archive_optional_double_nullopt)
{
    expect_archive_roundtrip<std::optional<double>>(std::nullopt);
}

TEST_F(cacheable_test, archive_optional_uint32_with_value)
{
    expect_archive_roundtrip<std::optional<std::uint32_t>>(0xDEADBEEF);
}

TEST_F(cacheable_test, archive_optional_uint64_with_value)
{
    expect_archive_roundtrip<std::optional<std::uint64_t>>(0xCAFEBABEULL);
}

TEST_F(cacheable_test, archive_optional_float_with_value)
{
    expect_archive_roundtrip<std::optional<float>>(1.5F);
}

TEST_F(cacheable_test, archive_optional_int64_negative)
{
    expect_archive_roundtrip<std::optional<std::int64_t>>(-100);
}

// ----- optional<vector> ----------------------------------------------------

TEST_F(cacheable_test, archive_optional_vector_uint8_with_value)
{
    expect_archive_roundtrip<std::optional<std::vector<std::uint8_t>>>(
        std::vector<std::uint8_t>{ 10, 20, 30 });
}

TEST_F(cacheable_test, archive_optional_vector_uint8_nullopt)
{
    expect_archive_roundtrip<std::optional<std::vector<std::uint8_t>>>(std::nullopt);
}

TEST_F(cacheable_test, archive_optional_vector_uint32_with_value)
{
    expect_archive_roundtrip<std::optional<std::vector<std::uint32_t>>>(
        std::vector<std::uint32_t>{ 0xDEAD, 0xBEEF, 0xCAFE });
}

TEST_F(cacheable_test, archive_optional_vector_empty_vector)
{
    expect_archive_roundtrip<std::optional<std::vector<std::uint8_t>>>(
        std::vector<std::uint8_t>{});
}

// ----- optional<string> ----------------------------------------------------

TEST_F(cacheable_test, archive_optional_string_with_value)
{
    expect_archive_roundtrip<std::optional<std::string>>(std::string{ "hello optional" });
}

TEST_F(cacheable_test, archive_optional_string_nullopt)
{
    expect_archive_roundtrip<std::optional<std::string>>(std::nullopt);
}

TEST_F(cacheable_test, archive_optional_string_empty)
{
    expect_archive_roundtrip<std::optional<std::string>>(std::string{});
}

// ----- std::array ----------------------------------------------------------

TEST_F(cacheable_test, archive_array_uint8)
{
    expect_archive_roundtrip<std::array<std::uint8_t, 4>>({ 1, 2, 3, 4 });
}

TEST_F(cacheable_test, archive_array_double)
{
    expect_archive_roundtrip<std::array<double, 3>>({ 1.0, 2.0, 3.0 });
}

// ----- multi-field round trip ----------------------------------------------

TEST_F(cacheable_test, archive_multiple_fields)
{
    const int                        int_val    = 42;
    const double                     double_val = 3.14;
    const std::string                str_val{ "abc" };
    const std::optional<std::string> opt_val{ std::string{ "present" } };
    const std::optional<std::string> opt_empty{ std::nullopt };

    std::array<std::uint8_t, 4096> buffer{};
    std::uint8_t*                  write_cursor = buffer.data();
    {
        rocprofsys::trace_cache::output_archive oa{ write_cursor };
        oa(int_val, double_val, str_val, opt_val, opt_empty);
    }
    const auto written = static_cast<std::size_t>(write_cursor - buffer.data());

    int                        out_int{};
    double                     out_double{};
    std::string                out_str;
    std::optional<std::string> out_opt;
    std::optional<std::string> out_opt_empty;

    std::uint8_t* read_cursor = buffer.data();
    {
        rocprofsys::trace_cache::input_archive ia{ read_cursor };
        ia(out_int, out_double, out_str, out_opt, out_opt_empty);
    }

    EXPECT_EQ(out_int, int_val);
    EXPECT_DOUBLE_EQ(out_double, double_val);
    EXPECT_EQ(out_str, str_val);
    EXPECT_EQ(out_opt, opt_val);
    EXPECT_EQ(out_opt_empty, opt_empty);
    EXPECT_EQ(static_cast<std::size_t>(read_cursor - buffer.data()), written);
}

// ----- serialized_size sanity ----------------------------------------------
//
// The framing header in buffer_storage uses serialized_size() to reserve the
// payload region; if that ever drifts from the byte count actually emitted by
// output_archive the cache file becomes unparseable. Cover that invariant
// explicitly.

TEST_F(cacheable_test, serialized_size_matches_bytes_written_arithmetic)
{
    const std::uint64_t value = 0x1234567890ABCDEFULL;

    std::array<std::uint8_t, 64> buffer{};
    std::uint8_t*                cursor = buffer.data();
    {
        rocprofsys::trace_cache::output_archive oa{ cursor };
        oa(value);
    }
    EXPECT_EQ(static_cast<std::size_t>(cursor - buffer.data()),
              rocprofsys::trace_cache::serialized_size(value));
}

TEST_F(cacheable_test, serialized_size_matches_bytes_written_optional_vector)
{
    const std::optional<std::vector<std::uint32_t>> value =
        std::vector<std::uint32_t>{ 1, 2, 3, 4 };

    std::array<std::uint8_t, 256> buffer{};
    std::uint8_t*                 cursor = buffer.data();
    {
        rocprofsys::trace_cache::output_archive oa{ cursor };
        oa(value);
    }
    EXPECT_EQ(static_cast<std::size_t>(cursor - buffer.data()),
              rocprofsys::trace_cache::serialized_size(value));
}

TEST_F(cacheable_test, serialized_size_optional_nullopt_is_one_byte)
{
    const std::optional<std::uint64_t> value = std::nullopt;
    EXPECT_EQ(rocprofsys::trace_cache::serialized_size(value), sizeof(std::uint8_t));
}

// ----- store_value framing header (production caller surface) -------------
//
// utility::store_value is the only utility helper that survived the archive
// migration; buffer_storage uses it to lay down the [type_id][size] framing
// header before the archive body. Cover that contract directly.

TEST_F(cacheable_test, store_value_framing_header_layout)
{
    enum class type_id_t : std::uint16_t
    {
        kSample = 0xBEEF
    };

    std::array<std::uint8_t, 32> buffer{};
    std::size_t                  position = 0;

    const auto        type_id     = static_cast<std::uint16_t>(type_id_t::kSample);
    const std::size_t sample_size = 1234;

    rocprofsys::trace_cache::utility::store_value(type_id, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(sample_size, buffer.data(), position);

    EXPECT_EQ(position, sizeof(std::uint16_t) + sizeof(std::size_t));

    std::uint16_t read_type_id = 0;
    std::size_t   read_size    = 0;
    std::memcpy(&read_type_id, buffer.data(), sizeof(std::uint16_t));
    std::memcpy(&read_size, buffer.data() + sizeof(std::uint16_t), sizeof(std::size_t));
    EXPECT_EQ(read_type_id, type_id);
    EXPECT_EQ(read_size, sample_size);
}

// ----- type_traits public surface ----------------------------------------
//
// The is_*_v helpers in cache_type_traits.hpp are part of the public header.
// Keep light coverage so future header refactors don't silently regress them.

TEST(type_traits_test, is_vector_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_vector_v<std::vector<int>>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_vector_v<int>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_vector_v<std::optional<int>>);
}

TEST(type_traits_test, is_optional_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<int>>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<int>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<std::vector<int>>);
}

// ----- utility filename helpers ------------------------------------------

TEST(cacheable_utility_test, get_buffered_storage_filename)
{
    const int ppid = 1234;
    const int pid  = 5678;
    EXPECT_EQ(rocprofsys::trace_cache::utility::get_buffered_storage_filename(ppid, pid),
              "/tmp/buffered_storage_1234_5678.bin");
}
