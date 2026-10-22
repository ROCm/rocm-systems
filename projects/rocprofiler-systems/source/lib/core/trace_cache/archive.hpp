// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace rocprofsys::trace_cache
{

static_assert(std::is_same_v<std::size_t, std::uint64_t>,
              "trace cache format assumes size_t is std::uint64_t");

template <typename Type, typename Archive>
concept archive_serializable =
    requires(Type& value, Archive& archive) { value.serialize(archive); };

struct binary_size_archive
{
    static constexpr bool k_is_loading = false;
    std::size_t           size         = 0;

    template <typename... Types>
    void operator()(const Types&... values)
    {
        (size_one(values), ...);
    }

private:
    template <typename Type>
    void size_one(const Type& value)
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
    static constexpr bool k_is_loading = false;
    std::uint8_t*         buffer       = nullptr;
    std::size_t           position     = 0;

    template <typename... Types>
    void operator()(const Types&... values)
    {
        // Locals: writes through the byte buffer may alias the members.
        std::uint8_t* const out = buffer;
        std::size_t         pos = position;
        (write_one(values, out, pos), ...);
        position = pos;
    }

private:
    template <typename Type>
    void write_one(const Type& value, std::uint8_t* out, std::size_t& pos)
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

/** string_view and span<const std::byte> fields are read as views into the buffer. */
struct binary_read_archive
{
    static constexpr bool k_is_loading = true;
    std::uint8_t*         cursor       = nullptr;

    template <typename... Types>
    void operator()(Types&... values)
    {
        std::uint8_t* cur = cursor;
        (read_one(values, cur), ...);
        cursor = cur;
    }

private:
    template <typename Type>
    void read_one(Type& value, std::uint8_t*& cur)
    {
        if constexpr(archive_serializable<Type, binary_read_archive>)
        {
            cursor = cur;
            value.serialize(*this);
            cur = cursor;
        }
        else if constexpr(std::is_same_v<Type, std::span<const std::byte>>)
        {
            std::size_t byte_count = 0;
            std::memcpy(&byte_count, cur, sizeof(std::size_t));
            cur += sizeof(std::size_t);
            value = { reinterpret_cast<const std::byte*>(cur), byte_count };
            cur += byte_count;
        }
        else
        {
            utility::parse_value(cur, value);
        }
    }
};

template <typename T>
concept archive_record =
    std::is_default_constructible_v<T> &&
    requires(T& record, binary_size_archive& sizer, binary_write_archive& writer,
             binary_read_archive& reader) {
        { T::type_identifier };
        record.serialize(sizer);
        record.serialize(writer);
        record.serialize(reader);
    };

// const_cast: the size and write archives only read through serialize().
// flatten, not inline (inlining drops it): one fully inlined function per record type.
template <archive_record T>
[[nodiscard, gnu::flatten]] std::size_t
record_size(const T& record)
{
    binary_size_archive archive;
    const_cast<T&>(record).serialize(archive);
    return archive.size;
}

template <archive_record T>
[[gnu::flatten]] void
record_write(std::uint8_t* buffer, const T& record)
{
    binary_write_archive archive{ .buffer = buffer, .position = 0 };
    const_cast<T&>(record).serialize(archive);
}

template <archive_record T>
[[nodiscard, gnu::flatten]] T
record_read(std::uint8_t*& buffer)
{
    // Default-initialized like the old deserializers: zeroing it first was slower.
    T                   record;
    binary_read_archive archive{ .cursor = buffer };
    record.serialize(archive);
    buffer = archive.cursor;
    return record;
}

template <archive_record T>
[[nodiscard]] inline std::size_t
get_size(const T& record)
{
    return record_size(record);
}

template <archive_record T>
inline void
serialize(std::uint8_t* buffer, const T& record)
{
    record_write(buffer, record);
}

template <archive_record T>
[[nodiscard]] inline T
deserialize(std::uint8_t*& buffer)
{
    return record_read<T>(buffer);
}

namespace type_traits
{
template <typename T, typename TypeIdentifierEnum>
concept cacheable = archive_record<T> && has_type_identifier<T, TypeIdentifierEnum>;
}  // namespace type_traits

}  // namespace rocprofsys::trace_cache
