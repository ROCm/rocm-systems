// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace rocprofsys::trace_cache
{

static_assert(std::is_same_v<std::size_t, std::uint64_t>,
              "trace cache format assumes size_t is std::uint64_t");

/**
 * True when Type itself declares `void serialize(Archive&)`. A serialize inherited from a
 * base class, or one that takes the archive by value, does not count: either would
 * compile and silently skip the fields of the derived record or lose the position.
 */
template <typename Type, typename Archive>
concept archive_serializable =
    std::is_same_v<decltype(&Type::template serialize<Archive>),
                   void (Type::*)(Archive&)>;

struct binary_size_archive
{
    std::size_t size = 0;

    template <typename... Types>
    constexpr void operator()(const Types&... values)
    {
        (measure_value(values), ...);
    }

private:
    template <typename Type>
    constexpr void measure_value(const Type& value)
    {
        if constexpr(archive_serializable<Type, binary_size_archive>)
        {
            const_cast<Type&>(value).serialize(*this);
        }
        else
        {
            size += utility::get_size(value);
        }
    }
};

struct binary_write_archive
{
    std::uint8_t* buffer   = nullptr;
    std::size_t   position = 0;

    template <typename... Types>
    void operator()(const Types&... values)
    {
        // Locals: writes through the byte buffer may alias the members.
        std::uint8_t* const out = buffer;
        std::size_t         pos = position;
        (write_value(values, out, pos), ...);
        position = pos;
    }

private:
    template <typename Type>
    void write_value(const Type& value, std::uint8_t* out, std::size_t& pos)
    {
        if constexpr(archive_serializable<Type, binary_write_archive>)
        {
            position = pos;
            const_cast<Type&>(value).serialize(*this);
            pos = position;
        }
        else
        {
            utility::store_value(value, out, pos);
        }
    }
};

/**
 * string_view fields are read as views into the buffer. Stored strings end with a '\0'
 * that consumers rely on (for example rocpd_processor passes `.data()` as a C string).
 */
struct binary_read_archive
{
    std::uint8_t* cursor = nullptr;

    template <typename... Types>
    void operator()(Types&... values)
    {
        std::uint8_t* cur = cursor;
        (read_value(values, cur), ...);
        cursor = cur;
    }

private:
    template <typename Type>
    void read_value(Type& value, std::uint8_t*& cur)
    {
        if constexpr(archive_serializable<Type, binary_read_archive>)
        {
            cursor = cur;
            value.serialize(*this);
            cur = cursor;
        }
        else
        {
            utility::parse_value(cur, value);
        }
    }
};

template <typename T>
concept archive_record =
    std::is_default_constructible_v<T> && requires { T::type_identifier; } &&
    archive_serializable<T, binary_size_archive> &&
    archive_serializable<T, binary_write_archive> &&
    archive_serializable<T, binary_read_archive>;

// const_cast: the size and write archives only read through serialize().
// flatten, not inline (inlining drops it): one fully inlined function per record type.
template <archive_record T>
[[nodiscard, gnu::flatten]] std::size_t
get_size(const T& record)
{
    binary_size_archive archive;
    const_cast<T&>(record).serialize(archive);
    return archive.size;
}

template <archive_record T>
[[gnu::flatten]] void
serialize(std::uint8_t* buffer, const T& record)
{
    binary_write_archive archive{ .buffer = buffer, .position = 0 };
    const_cast<T&>(record).serialize(archive);
}

template <archive_record T>
[[nodiscard, gnu::flatten]] T
deserialize(std::uint8_t*& buffer)
{
    // Default-initialized like the old deserializers: zeroing it first was slower.
    T                   record;
    binary_read_archive archive{ .cursor = buffer };
    record.serialize(archive);
    buffer = archive.cursor;
    return record;
}

namespace type_traits
{
template <typename T, typename TypeIdentifierEnum>
concept cacheable = archive_record<T> && has_type_identifier<T, TypeIdentifierEnum>;
}  // namespace type_traits

}  // namespace rocprofsys::trace_cache
