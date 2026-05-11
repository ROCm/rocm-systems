// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cacheable.hpp"
#include <cstdint>

#include <cmath>
#include <cstddef>
#include <string>
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

    int         value = 0;
    std::string text;

    bool operator==(const test_sample_1& other) const
    {
        return value == other.value && text == other.text;
    }

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(value, text);
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

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(data, sample_id);
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

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(payload);
    }
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

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(data);
    }
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

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(data);
    }
};
