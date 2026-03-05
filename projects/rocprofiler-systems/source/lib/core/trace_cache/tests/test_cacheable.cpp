// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std::string_view_literals;

class cacheable_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        buffer.fill(0);
        position = 0;
    }

    std::array<uint8_t, 1024> buffer;
    size_t                    position;
};

TEST_F(cacheable_test, store_value_int)
{
    int value = 42;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(int));
    int stored_value = *reinterpret_cast<int*>(buffer.data());
    EXPECT_EQ(stored_value, 42);
}

TEST_F(cacheable_test, store_value_double)
{
    double value = 3.14159;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(double));
    double stored_value = *reinterpret_cast<double*>(buffer.data());
    EXPECT_DOUBLE_EQ(stored_value, 3.14159);
}

TEST_F(cacheable_test, store_value_unsigned_long)
{
    unsigned long value = 123456789UL;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(unsigned long));
    unsigned long stored_value = *reinterpret_cast<unsigned long*>(buffer.data());
    EXPECT_EQ(stored_value, 123456789UL);
}

TEST_F(cacheable_test, store_value_unsigned_char)
{
    unsigned char value = 255;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(unsigned char));
    unsigned char stored_value = *reinterpret_cast<unsigned char*>(buffer.data());
    EXPECT_EQ(stored_value, 255);
}

TEST_F(cacheable_test, store_value_string_literal)
{
    auto value = "Hello World"sv;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    size_t expected_size = value.size() + sizeof(size_t);
    EXPECT_EQ(position, expected_size);

    std::string stored_value(
        reinterpret_cast<const char*>(buffer.data() + sizeof(size_t)));
    EXPECT_EQ(stored_value, "Hello World");
}

TEST_F(cacheable_test, store_value_optional)
{
    std::optional<double> value = 42.0;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    EXPECT_EQ(position, sizeof(uint8_t) + sizeof(double));
    EXPECT_EQ(buffer[0], 1);
    EXPECT_DOUBLE_EQ(*reinterpret_cast<double*>(buffer.data() + sizeof(uint8_t)),
                     value.value());
}

TEST_F(cacheable_test, store_value_optional_empty)
{
    std::optional<double> value = std::nullopt;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    EXPECT_EQ(position, sizeof(uint8_t));
    EXPECT_EQ(buffer[0], 0);
}

TEST_F(cacheable_test, store_value_empty_string)
{
    auto value = ""sv;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(size_t));
    EXPECT_EQ(buffer[0], '\0');
}

TEST_F(cacheable_test, store_value_byte_array)
{
    std::vector<uint8_t> value = { 1, 2, 3, 4, 5 };
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    size_t expected_size = value.size() + sizeof(size_t);
    EXPECT_EQ(position, expected_size);

    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, 5);

    uint8_t* data_start = buffer.data() + sizeof(size_t);
    for(size_t i = 0; i < value.size(); ++i)
    {
        EXPECT_EQ(data_start[i], value[i]);
    }
}

TEST_F(cacheable_test, store_value_empty_byte_array)
{
    std::vector<uint8_t> value;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    EXPECT_EQ(position, sizeof(size_t));
    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, 0);
}

TEST_F(cacheable_test, store_multiple_values)
{
    int    int_val    = 100;
    double double_val = 2.718;
    auto   str_val    = "test"sv;

    rocprofsys::trace_cache::utility::store_value(int_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(double_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(str_val, buffer.data(), position);

    size_t expected_total =
        sizeof(int) + sizeof(double) + str_val.size() + sizeof(size_t);
    EXPECT_EQ(position, expected_total);
}

TEST_F(cacheable_test, parse_value_int)
{
    int original_value = 987;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t* data_pos = buffer.data();
    int      parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, 987);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(int));
}

TEST_F(cacheable_test, parse_value_double)
{
    double original_value = 1.618033988;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t* data_pos = buffer.data();
    double   parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_DOUBLE_EQ(parsed_value, 1.618033988);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(double));
}

TEST_F(cacheable_test, parse_value_unsigned_long)
{
    unsigned long original_value = 0xDEADBEEF;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*      data_pos = buffer.data();
    unsigned long parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, 0xDEADBEEF);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(unsigned long));
}

TEST_F(cacheable_test, parse_value_string)
{
    auto original_value = "Parse this string"sv;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*         data_pos = buffer.data();
    std::string_view parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, "Parse this string");
    EXPECT_EQ(data_pos, buffer.data() + original_value.size() + sizeof(size_t));
}

TEST_F(cacheable_test, parse_value_empty_string)
{
    auto original_value = ""sv;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*         data_pos = buffer.data();
    std::string_view parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, "");
    EXPECT_EQ(data_pos, buffer.data() + sizeof(size_t));
}

TEST_F(cacheable_test, parse_value_optional)
{
    std::optional<double> original_value = 42.0;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*              data_pos = buffer.data();
    std::optional<double> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, 42.0);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t) + sizeof(double));
}

TEST_F(cacheable_test, parse_value_optional_empty)
{
    std::optional<double> original_value = std::nullopt;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*              data_pos = buffer.data();
    std::optional<double> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value, std::nullopt);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t));
}

TEST_F(cacheable_test, roundtrip_optional_uint32_with_value)
{
    std::optional<uint32_t> original_value = 42;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*                data_pos = buffer.data();
    std::optional<uint32_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    ASSERT_TRUE(parsed_value.has_value());
    EXPECT_EQ(parsed_value.value(), 42);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t) + sizeof(uint32_t));
}

TEST_F(cacheable_test, roundtrip_optional_uint32_nullopt)
{
    std::optional<uint32_t> original_value = std::nullopt;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*                data_pos = buffer.data();
    std::optional<uint32_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_FALSE(parsed_value.has_value());
    EXPECT_EQ(parsed_value, std::nullopt);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t));
}

TEST_F(cacheable_test, roundtrip_optional_uint64)
{
    std::optional<uint64_t> original_value = 0xDEADBEEFCAFEBABE;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*                data_pos = buffer.data();
    std::optional<uint64_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    ASSERT_TRUE(parsed_value.has_value());
    EXPECT_EQ(parsed_value.value(), 0xDEADBEEFCAFEBABE);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t) + sizeof(uint64_t));
}

TEST_F(cacheable_test, roundtrip_optional_float)
{
    std::optional<float> original_value = 3.14f;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*             data_pos = buffer.data();
    std::optional<float> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    ASSERT_TRUE(parsed_value.has_value());
    EXPECT_FLOAT_EQ(parsed_value.value(), 3.14f);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t) + sizeof(float));
}

TEST_F(cacheable_test, roundtrip_optional_int64_negative)
{
    std::optional<int64_t> original_value = -999999;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*               data_pos = buffer.data();
    std::optional<int64_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    ASSERT_TRUE(parsed_value.has_value());
    EXPECT_EQ(parsed_value.value(), -999999);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t) + sizeof(int64_t));
}

TEST_F(cacheable_test, parse_multiple_values_with_optional)
{
    int                     int_val   = 100;
    std::optional<uint32_t> opt_val   = uint32_t{ 777 };
    auto                    str_val   = "mixed"sv;
    std::optional<double>   opt_empty = std::nullopt;

    rocprofsys::trace_cache::utility::store_value(int_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(str_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_empty, buffer.data(), position);

    uint8_t*                data_pos = buffer.data();
    int                     parsed_int;
    std::optional<uint32_t> parsed_opt;
    std::string_view        parsed_str;
    std::optional<double>   parsed_opt_empty;

    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_int, parsed_opt,
                                                  parsed_str, parsed_opt_empty);

    EXPECT_EQ(parsed_int, 100);
    ASSERT_TRUE(parsed_opt.has_value());
    EXPECT_EQ(parsed_opt.value(), 777);
    EXPECT_EQ(parsed_str, "mixed");
    EXPECT_FALSE(parsed_opt_empty.has_value());
}

TEST_F(cacheable_test, roundtrip_optional_vector_uint8_with_value)
{
    std::optional<std::vector<uint8_t>> original = std::vector<uint8_t>{ 10, 20, 30 };
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    const size_t expected_size =
        sizeof(uint8_t) + sizeof(size_t) + original->size() * sizeof(uint8_t);
    EXPECT_EQ(position, expected_size);

    uint8_t*                            data_pos = buffer.data();
    std::optional<std::vector<uint8_t>> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(*parsed, *original);
    EXPECT_EQ(data_pos, buffer.data() + expected_size);
}

TEST_F(cacheable_test, roundtrip_optional_vector_uint8_nullopt)
{
    std::optional<std::vector<uint8_t>> original = std::nullopt;
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    EXPECT_EQ(position, sizeof(uint8_t));

    uint8_t*                            data_pos = buffer.data();
    std::optional<std::vector<uint8_t>> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    EXPECT_FALSE(parsed.has_value());
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t));
}

TEST_F(cacheable_test, roundtrip_optional_vector_uint32_with_value)
{
    std::optional<std::vector<uint32_t>> original =
        std::vector<uint32_t>{ 0xDEAD, 0xBEEF, 0xCAFE };
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    const size_t expected_size =
        sizeof(uint8_t) + sizeof(size_t) + original->size() * sizeof(uint32_t);
    EXPECT_EQ(position, expected_size);

    uint8_t*                             data_pos = buffer.data();
    std::optional<std::vector<uint32_t>> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(*parsed, *original);
    EXPECT_EQ(data_pos, buffer.data() + expected_size);
}

TEST_F(cacheable_test, roundtrip_optional_vector_empty_vector)
{
    std::optional<std::vector<uint8_t>> original = std::vector<uint8_t>{};
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    const size_t expected_size = sizeof(uint8_t) + sizeof(size_t);
    EXPECT_EQ(position, expected_size);

    uint8_t*                            data_pos = buffer.data();
    std::optional<std::vector<uint8_t>> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(parsed->empty());
    EXPECT_EQ(data_pos, buffer.data() + expected_size);
}

TEST_F(cacheable_test, roundtrip_optional_string_view_with_value)
{
    std::optional<std::string_view> original = "hello optional"sv;
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    const size_t expected_size = sizeof(uint8_t) + sizeof(size_t) + original->size();
    EXPECT_EQ(position, expected_size);

    uint8_t*                        data_pos = buffer.data();
    std::optional<std::string_view> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(*parsed, *original);
    EXPECT_EQ(data_pos, buffer.data() + expected_size);
}

TEST_F(cacheable_test, roundtrip_optional_string_view_nullopt)
{
    std::optional<std::string_view> original = std::nullopt;
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    EXPECT_EQ(position, sizeof(uint8_t));

    uint8_t*                        data_pos = buffer.data();
    std::optional<std::string_view> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    EXPECT_FALSE(parsed.has_value());
    EXPECT_EQ(data_pos, buffer.data() + sizeof(uint8_t));
}

TEST_F(cacheable_test, roundtrip_optional_string_view_empty_string)
{
    std::optional<std::string_view> original = ""sv;
    rocprofsys::trace_cache::utility::store_value(original, buffer.data(), position);

    const size_t expected_size = sizeof(uint8_t) + sizeof(size_t);
    EXPECT_EQ(position, expected_size);

    uint8_t*                        data_pos = buffer.data();
    std::optional<std::string_view> parsed;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed);

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(*parsed, "");
    EXPECT_EQ(data_pos, buffer.data() + expected_size);
}

TEST_F(cacheable_test, parse_multiple_values_with_optional_vector_and_string_view)
{
    int                                 int_val     = 42;
    std::optional<std::vector<uint8_t>> opt_vec     = std::vector<uint8_t>{ 1, 2, 3 };
    std::optional<std::string_view>     opt_str     = "nested"sv;
    std::optional<std::vector<uint8_t>> opt_nullvec = std::nullopt;
    std::optional<std::string_view>     opt_nullstr = std::nullopt;

    rocprofsys::trace_cache::utility::store_value(int_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_vec, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_str, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_nullvec, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(opt_nullstr, buffer.data(), position);

    uint8_t*                            data_pos = buffer.data();
    int                                 parsed_int;
    std::optional<std::vector<uint8_t>> parsed_vec;
    std::optional<std::string_view>     parsed_str;
    std::optional<std::vector<uint8_t>> parsed_nullvec;
    std::optional<std::string_view>     parsed_nullstr;

    rocprofsys::trace_cache::utility::parse_value(
        data_pos, parsed_int, parsed_vec, parsed_str, parsed_nullvec, parsed_nullstr);

    EXPECT_EQ(parsed_int, 42);
    ASSERT_TRUE(parsed_vec.has_value());
    EXPECT_EQ(*parsed_vec, (std::vector<uint8_t>{ 1, 2, 3 }));
    ASSERT_TRUE(parsed_str.has_value());
    EXPECT_EQ(*parsed_str, "nested");
    EXPECT_FALSE(parsed_nullvec.has_value());
    EXPECT_FALSE(parsed_nullstr.has_value());
}

TEST_F(cacheable_test, parse_value_byte_array)
{
    std::vector<uint8_t> original_value = { 10, 20, 30, 40, 50 };
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*             data_pos = buffer.data();
    std::vector<uint8_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), 5);
    EXPECT_EQ(parsed_value, original_value);
    EXPECT_EQ(data_pos, buffer.data() + sizeof(size_t) + original_value.size());
}

TEST_F(cacheable_test, parse_value_empty_byte_array)
{
    std::vector<uint8_t> original_value;
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*             data_pos = buffer.data();
    std::vector<uint8_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), 0);
    EXPECT_TRUE(parsed_value.empty());
    EXPECT_EQ(data_pos, buffer.data() + sizeof(size_t));
}
TEST_F(cacheable_test, parse_multiple_values)
{
    int           int_val    = 42;
    double        double_val = 3.14;
    auto          str_val    = "multi"sv;
    unsigned char uchar_val  = 128;

    rocprofsys::trace_cache::utility::store_value(int_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(double_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(str_val, buffer.data(), position);
    rocprofsys::trace_cache::utility::store_value(uchar_val, buffer.data(), position);

    uint8_t* data_pos = buffer.data();

    int              parsed_int;
    double           parsed_double;
    std::string_view parsed_string;
    unsigned char    parsed_uchar;

    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_int, parsed_double,
                                                  parsed_string, parsed_uchar);

    EXPECT_EQ(parsed_int, 42);
    EXPECT_DOUBLE_EQ(parsed_double, 3.14);
    EXPECT_EQ(parsed_string, "multi");
    EXPECT_EQ(parsed_uchar, 128);
}

TEST_F(cacheable_test, get_size_helper_int)
{
    int    value = 42;
    size_t size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, sizeof(int));
}

TEST_F(cacheable_test, get_size_helper_double)
{
    double value = 3.14;
    size_t size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, sizeof(double));
}

TEST_F(cacheable_test, get_size_helper_string_literal)
{
    auto   value = "test string"sv;
    size_t size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, value.size() + sizeof(size_t));
}

TEST_F(cacheable_test, get_size_helper_optional)
{
    std::optional<int> value = 42;
    size_t             size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, sizeof(uint8_t) + sizeof(int));
}

TEST_F(cacheable_test, get_size_helper_optional_empty)
{
    std::optional<int> value = std::nullopt;
    size_t             size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, sizeof(uint8_t));
}

TEST_F(cacheable_test, get_size_helper_byte_array)
{
    std::vector<uint8_t> value = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    size_t               size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, value.size() + sizeof(size_t));
}

TEST_F(cacheable_test, get_size_optional_vector)
{
    std::optional<std::vector<uint8_t>> val = std::vector<uint8_t>{ 1, 2, 3 };
    size_t size = rocprofsys::trace_cache::utility::get_size(val);
    EXPECT_EQ(size, sizeof(uint8_t) + sizeof(size_t) + 3 * sizeof(uint8_t));
}

TEST_F(cacheable_test, get_size_optional_vector_nullopt)
{
    std::optional<std::vector<uint8_t>> val = std::nullopt;
    size_t size = rocprofsys::trace_cache::utility::get_size(val);
    EXPECT_EQ(size, sizeof(uint8_t));
}

TEST_F(cacheable_test, get_size_optional_string_view)
{
    std::optional<std::string_view> val = "test"sv;
    size_t size                         = rocprofsys::trace_cache::utility::get_size(val);
    EXPECT_EQ(size, sizeof(uint8_t) + sizeof(size_t) + 4);
}

TEST_F(cacheable_test, get_size_optional_string_view_nullopt)
{
    std::optional<std::string_view> val = std::nullopt;
    size_t size                         = rocprofsys::trace_cache::utility::get_size(val);
    EXPECT_EQ(size, sizeof(uint8_t));
}

// Tests for size consistency of optional types. The value returned by get_size()
//  must match the actual bytes written by store_value().
TEST_F(cacheable_test, size_consistency_optional_with_value)
{
    std::optional<uint64_t> value = 0xDEADBEEF;
    // Calculate size
    size_t calculated_size = rocprofsys::trace_cache::utility::get_size(value);
    // Store and track actual bytes written
    size_t position = 0;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    // These MUST match exactly
    EXPECT_EQ(position, calculated_size)
        << "get_size() and store_value() wrote different amounts";
}
TEST_F(cacheable_test, size_consistency_optional_nullopt)
{
    std::optional<uint64_t> value = std::nullopt;
    size_t calculated_size        = rocprofsys::trace_cache::utility::get_size(value);
    size_t position               = 0;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    EXPECT_EQ(position, calculated_size)
        << "get_size() and store_value() wrote different amounts for nullopt";
}
TEST_F(cacheable_test, size_consistency_optional_vector_with_value)
{
    std::optional<std::vector<uint32_t>> value = std::vector<uint32_t>{ 1, 2, 3, 4, 5 };
    size_t calculated_size = rocprofsys::trace_cache::utility::get_size(value);
    size_t position        = 0;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    EXPECT_EQ(position, calculated_size);
}
TEST_F(cacheable_test, size_consistency_optional_vector_nullopt)
{
    std::optional<std::vector<uint32_t>> value = std::nullopt;
    size_t calculated_size = rocprofsys::trace_cache::utility::get_size(value);
    size_t position        = 0;
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);
    EXPECT_EQ(position, calculated_size);
}

TEST_F(cacheable_test, store_value_span_uint8)
{
    std::vector<uint8_t>      data = { 1, 2, 3, 4, 5 };
    rocprofsys::span<uint8_t> span_val(data);
    rocprofsys::trace_cache::utility::store_value(span_val, buffer.data(), position);

    size_t expected_size = data.size() * sizeof(uint8_t) + sizeof(size_t);
    EXPECT_EQ(position, expected_size);

    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, data.size() * sizeof(uint8_t));

    uint8_t* data_start = buffer.data() + sizeof(size_t);
    for(size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_EQ(data_start[i], data[i]);
    }
}

TEST_F(cacheable_test, store_value_span_uint32)
{
    std::vector<uint32_t>      data = { 100, 200, 300, 400, 500 };
    rocprofsys::span<uint32_t> span_val(data);
    rocprofsys::trace_cache::utility::store_value(span_val, buffer.data(), position);

    size_t expected_data_size  = data.size() * sizeof(uint32_t);
    size_t expected_total_size = expected_data_size + sizeof(size_t);
    EXPECT_EQ(position, expected_total_size);

    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, expected_data_size);

    uint32_t* data_start = reinterpret_cast<uint32_t*>(buffer.data() + sizeof(size_t));
    for(size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_EQ(data_start[i], data[i]);
    }
}

TEST_F(cacheable_test, store_value_empty_span)
{
    rocprofsys::span<uint8_t> empty_span(nullptr, 0);
    rocprofsys::trace_cache::utility::store_value(empty_span, buffer.data(), position);

    EXPECT_EQ(position, sizeof(size_t));
    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, 0);
}

TEST_F(cacheable_test, span_store_vector_parse_roundtrip)
{
    std::vector<uint8_t>      original_data = { 10, 20, 30, 40, 50 };
    rocprofsys::span<uint8_t> span_val(original_data);
    rocprofsys::trace_cache::utility::store_value(span_val, buffer.data(), position);

    uint8_t*             data_pos = buffer.data();
    std::vector<uint8_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), original_data.size());
    EXPECT_EQ(parsed_value, original_data);
}

TEST_F(cacheable_test, span_uint32_store_vector_parse_roundtrip)
{
    std::vector<uint32_t>      original_data = { 0xDEADBEEF, 0xCAFEBABE, 0x12345678 };
    rocprofsys::span<uint32_t> span_val(original_data);
    rocprofsys::trace_cache::utility::store_value(span_val, buffer.data(), position);

    uint8_t*              data_pos = buffer.data();
    std::vector<uint32_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), original_data.size());
    EXPECT_EQ(parsed_value, original_data);
}

TEST_F(cacheable_test, get_size_span_uint8)
{
    std::vector<uint8_t>      data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    rocprofsys::span<uint8_t> span_val(data);
    size_t                    size = rocprofsys::trace_cache::utility::get_size(span_val);
    EXPECT_EQ(size, data.size() * sizeof(uint8_t) + sizeof(size_t));
}

TEST_F(cacheable_test, get_size_span_uint32)
{
    std::vector<uint32_t>      data = { 100, 200, 300, 400, 500 };
    rocprofsys::span<uint32_t> span_val(data);
    size_t size = rocprofsys::trace_cache::utility::get_size(span_val);
    EXPECT_EQ(size, data.size() * sizeof(uint32_t) + sizeof(size_t));
}

TEST_F(cacheable_test, store_value_int_vector)
{
    std::vector<int> value = { -100, 0, 100, 200, -200 };
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    size_t expected_data_size = value.size() * sizeof(int);
    size_t expected_total     = expected_data_size + sizeof(size_t);
    EXPECT_EQ(position, expected_total);

    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, expected_data_size);

    int* data_start = reinterpret_cast<int*>(buffer.data() + sizeof(size_t));
    for(size_t i = 0; i < value.size(); ++i)
    {
        EXPECT_EQ(data_start[i], value[i]);
    }
}

TEST_F(cacheable_test, parse_value_int_vector)
{
    std::vector<int> original_value = { -100, 0, 100, 200, -200 };
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*         data_pos = buffer.data();
    std::vector<int> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), original_value.size());
    EXPECT_EQ(parsed_value, original_value);
    size_t expected_advance = sizeof(size_t) + original_value.size() * sizeof(int);
    EXPECT_EQ(data_pos, buffer.data() + expected_advance);
}

TEST_F(cacheable_test, store_value_uint64_vector)
{
    std::vector<uint64_t> value = { 0xFFFFFFFFFFFFFFFF, 0x0, 0x123456789ABCDEF0 };
    rocprofsys::trace_cache::utility::store_value(value, buffer.data(), position);

    size_t expected_data_size = value.size() * sizeof(uint64_t);
    size_t expected_total     = expected_data_size + sizeof(size_t);
    EXPECT_EQ(position, expected_total);

    size_t stored_size = *reinterpret_cast<size_t*>(buffer.data());
    EXPECT_EQ(stored_size, expected_data_size);
}

TEST_F(cacheable_test, parse_value_uint64_vector)
{
    std::vector<uint64_t> original_value = { 0xFFFFFFFFFFFFFFFF, 0x0,
                                             0x123456789ABCDEF0 };
    rocprofsys::trace_cache::utility::store_value(original_value, buffer.data(),
                                                  position);

    uint8_t*              data_pos = buffer.data();
    std::vector<uint64_t> parsed_value;
    rocprofsys::trace_cache::utility::parse_value(data_pos, parsed_value);

    EXPECT_EQ(parsed_value.size(), original_value.size());
    EXPECT_EQ(parsed_value, original_value);
}

TEST_F(cacheable_test, get_size_int_vector)
{
    std::vector<int> value = { 1, 2, 3, 4, 5 };
    size_t           size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, value.size() * sizeof(int) + sizeof(size_t));
}

TEST_F(cacheable_test, get_size_uint64_vector)
{
    std::vector<uint64_t> value = { 1, 2, 3, 4, 5 };
    size_t                size  = rocprofsys::trace_cache::utility::get_size(value);
    EXPECT_EQ(size, value.size() * sizeof(uint64_t) + sizeof(size_t));
}

TEST(type_traits_test, is_span_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_span_v<rocprofsys::span<int>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_span_v<rocprofsys::span<uint8_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_span_v<rocprofsys::span<double>>);

    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_span_v<int>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_span_v<std::vector<int>>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_span_v<std::string_view>);
}

TEST(type_traits_test, is_vector_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_vector_v<std::vector<int>>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_vector_v<std::vector<uint8_t>>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_vector_v<std::vector<double>>);

    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_vector_v<int>);
    EXPECT_FALSE(
        rocprofsys::trace_cache::type_traits::is_vector_v<rocprofsys::span<int>>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_vector_v<std::string_view>);
}

TEST(type_traits_test, is_optional_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<int>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<uint32_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<double>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<uint64_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_optional_v<std::optional<float>>);

    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<int>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<double>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<std::vector<int>>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_optional_v<std::string_view>);
    EXPECT_FALSE(
        rocprofsys::trace_cache::type_traits::is_optional_v<rocprofsys::span<int>>);
}

TEST(type_traits_test, is_supported_type_v)
{
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<int>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<uint64_t>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<double>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<float>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<std::string_view>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<std::vector<int>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<std::vector<uint8_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<rocprofsys::span<int>>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<
                rocprofsys::span<uint8_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<std::optional<int>>);
    EXPECT_TRUE(rocprofsys::trace_cache::type_traits::is_supported_type_v<
                std::optional<uint32_t>>);
    EXPECT_TRUE(
        rocprofsys::trace_cache::type_traits::is_supported_type_v<std::optional<double>>);

    struct custom_type
    {};
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_supported_type_v<custom_type>);
    EXPECT_FALSE(rocprofsys::trace_cache::type_traits::is_supported_type_v<std::string>);
}

TEST_F(cacheable_test, get_buffered_storage_filename)
{
    int ppid = 1234;
    int pid  = 5678;

    std::string filename =
        rocprofsys::trace_cache::utility::get_buffered_storage_filename(ppid, pid);
    std::string expected = "/tmp/buffered_storage_1234_5678.bin";

    EXPECT_EQ(filename, expected);
}
