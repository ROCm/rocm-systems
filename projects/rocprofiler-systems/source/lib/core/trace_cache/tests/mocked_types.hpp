// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cacheable.hpp"
#include <cstdint>

#include <cmath>
#include <cstddef>
#include <vector>

enum class test_type_identifier_t : std::uint32_t
{
    sample_type_1    = 1,
    sample_type_2    = 2,
    sample_type_3    = 3,
    sample_type_4    = 4,
    sample_type_5    = 5,
    fragmented_space = 0xFFFF
};
struct test_sample_1 : public rocprofsys::trace_cache::cacheable_t
{
    static constexpr test_type_identifier_t type_identifier =
        test_type_identifier_t::sample_type_1;

    test_sample_1() = default;
    test_sample_1(int v, std::string_view s)
    : value(v)
    , text(s)
    {}

    int              value = 0;
    std::string_view text;

    bool operator==(const test_sample_1& other) const
    {
        return value == other.value && text == other.text;
    }
};

struct test_sample_2 : public rocprofsys::trace_cache::cacheable_t
{
    static constexpr test_type_identifier_t type_identifier =
        test_type_identifier_t::sample_type_2;

    test_sample_2() = default;
    test_sample_2(double d, std::uint32_t id)
    : data(d)
    , sample_id(id)
    {}

    double        data      = 0.0;
    std::uint32_t sample_id = 0;

    bool operator==(const test_sample_2& other) const
    {
        if(sample_id != other.sample_id) return false;

        if(std::isnan(data) && std::isnan(other.data)) return true;

        if(std::isinf(data) && std::isinf(other.data))
            return std::signbit(data) == std::signbit(other.data);

        return std::abs(data - other.data) < 1e-9;
    }
};

struct test_sample_3 : public rocprofsys::trace_cache::cacheable_t
{
    static constexpr test_type_identifier_t type_identifier =
        test_type_identifier_t::sample_type_3;

    test_sample_3() = default;
    test_sample_3(std::vector<std::uint8_t> p)
    : payload(std::move(p))
    {}

    std::vector<std::uint8_t> payload;

    bool operator==(const test_sample_3& other) const { return payload == other.payload; }
};

struct test_sample_4 : public rocprofsys::trace_cache::cacheable_t
{
    static constexpr test_type_identifier_t type_identifier =
        test_type_identifier_t::sample_type_4;

    test_sample_4() = default;
    test_sample_4(std::vector<std::uint32_t> d)
    : data(std::move(d))
    {}

    std::vector<std::uint32_t> data;

    bool operator==(const test_sample_4& other) const { return data == other.data; }
};

struct test_sample_5 : public rocprofsys::trace_cache::cacheable_t
{
    static constexpr test_type_identifier_t type_identifier =
        test_type_identifier_t::sample_type_5;

    test_sample_5() = default;
    test_sample_5(std::optional<std::uint32_t> d)
    : data(d)
    {}

    std::optional<std::uint32_t> data;

    bool operator==(const test_sample_5& other) const { return data == other.data; }
};

template <>
inline void
rocprofsys::trace_cache::serialize(std::uint8_t* buffer, const test_sample_1& item)
{
    rocprofsys::trace_cache::utility::store_value(buffer, item.value, item.text);
}

template <>
inline test_sample_1
rocprofsys::trace_cache::deserialize(std::uint8_t*& buffer)
{
    test_sample_1 result;
    rocprofsys::trace_cache::utility::parse_value(buffer, result.value, result.text);
    return result;
}

template <>
inline size_t
rocprofsys::trace_cache::get_size(const test_sample_1& item)
{
    return rocprofsys::trace_cache::utility::get_size(item.value, item.text);
}

template <>
inline void
rocprofsys::trace_cache::serialize(std::uint8_t* buffer, const test_sample_2& item)
{
    rocprofsys::trace_cache::utility::store_value(buffer, item.data, item.sample_id);
}

template <>
inline test_sample_2
rocprofsys::trace_cache::deserialize(std::uint8_t*& buffer)
{
    test_sample_2 result;
    rocprofsys::trace_cache::utility::parse_value(buffer, result.data, result.sample_id);
    return result;
}

template <>
inline size_t
rocprofsys::trace_cache::get_size(const test_sample_2& item)
{
    return rocprofsys::trace_cache::utility::get_size(item.data, item.sample_id);
}

template <>
inline void
rocprofsys::trace_cache::serialize(std::uint8_t* buffer, const test_sample_3& item)
{
    rocprofsys::trace_cache::utility::store_value(buffer, item.payload);
}

template <>
inline test_sample_3
rocprofsys::trace_cache::deserialize(std::uint8_t*& buffer)
{
    test_sample_3 result;
    rocprofsys::trace_cache::utility::parse_value(buffer, result.payload);
    return result;
}

template <>
inline size_t
rocprofsys::trace_cache::get_size(const test_sample_3& item)
{
    return rocprofsys::trace_cache::utility::get_size(item.payload);
}

template <>
inline void
rocprofsys::trace_cache::serialize(std::uint8_t* buffer, const test_sample_4& item)
{
    rocprofsys::trace_cache::utility::store_value(buffer, item.data);
}

template <>
inline test_sample_4
rocprofsys::trace_cache::deserialize(std::uint8_t*& buffer)
{
    test_sample_4 result;
    rocprofsys::trace_cache::utility::parse_value(buffer, result.data);
    return result;
}

template <>
inline size_t
rocprofsys::trace_cache::get_size(const test_sample_4& item)
{
    return rocprofsys::trace_cache::utility::get_size(item.data);
}

template <>
inline void
rocprofsys::trace_cache::serialize(std::uint8_t* buffer, const test_sample_5& item)
{
    rocprofsys::trace_cache::utility::store_value(buffer, item.data);
}

template <>
inline test_sample_5
rocprofsys::trace_cache::deserialize(std::uint8_t*& buffer)
{
    test_sample_5 result;
    rocprofsys::trace_cache::utility::parse_value(buffer, result.data);
    return result;
}

template <>
inline size_t
rocprofsys::trace_cache::get_size(const test_sample_5& item)
{
    return rocprofsys::trace_cache::utility::get_size(item.data);
}
