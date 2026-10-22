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

// The cache format writes size_t lengths; records pass size_t fields where the old code
// cast them to std::uint64_t. Both rely on the two being the same type.
static_assert(std::is_same_v<std::size_t, std::uint64_t>,
              "trace cache format assumes size_t is std::uint64_t");

/** Sums the encoded size of the fields a record passes to it. */
struct binary_size_archive
{
    static constexpr bool k_is_loading = false;
    std::size_t           size         = 0;

    template <typename... Types>
    void operator()(const Types&... values)
    {
        size += (std::size_t{ 0 } + ... + utility::get_size(values));
    }
};

/** Writes the fields into a buffer that already has room for record_size() bytes. */
struct binary_write_archive
{
    static constexpr bool k_is_loading = false;
    std::uint8_t*         buffer       = nullptr;
    std::size_t           position     = 0;

    template <typename... Types>
    void operator()(const Types&... values)
    {
        (utility::store_value(values, buffer, position), ...);
    }
};

/**
 * Reads the fields back. string_view and span<const std::byte> fields become views into
 * the buffer, so the buffer must outlive the record (as for the parse buffer today).
 */
struct binary_read_archive
{
    static constexpr bool k_is_loading = true;
    std::uint8_t*         cursor       = nullptr;

    template <typename... Types>
    void operator()(Types&... values)
    {
        (read_one(values), ...);
    }

private:
    template <typename Type>
    void read_one(Type& value)
    {
        if constexpr(std::is_same_v<Type, std::span<const std::byte>>)
        {
            std::size_t byte_count = 0;
            std::memcpy(&byte_count, cursor, sizeof(std::size_t));
            cursor += sizeof(std::size_t);
            value = { reinterpret_cast<const std::byte*>(cursor), byte_count };
            cursor += byte_count;
        }
        else
        {
            utility::parse_value(cursor, value);
        }
    }
};

/** A record that lists its fields once, in byte order, in serialize(Archive&). */
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

// serialize() is one non-const member shared by all archives. The size and write archives
// only read through it, so the const_cast is safe (cereal's output archives do the same).
template <archive_record T>
[[nodiscard]] std::size_t
record_size(const T& record)
{
    binary_size_archive archive;
    const_cast<T&>(record).serialize(archive);
    return archive.size;
}

template <archive_record T>
void
record_write(std::uint8_t* buffer, const T& record)
{
    binary_write_archive archive{ .buffer = buffer, .position = 0 };
    const_cast<T&>(record).serialize(archive);
}

template <archive_record T>
[[nodiscard]] T
record_read(std::uint8_t*& buffer)
{
    T                   record{};
    binary_read_archive archive{ .cursor = buffer };
    record.serialize(archive);
    buffer = archive.cursor;
    return record;
}

/** Encoded size of a record. One definition for every record type. */
template <archive_record T>
[[nodiscard]] std::size_t
get_size(const T& record)
{
    return record_size(record);
}

/** Writes a record into a buffer with room for get_size(record) bytes. */
template <archive_record T>
void
serialize(std::uint8_t* buffer, const T& record)
{
    record_write(buffer, record);
}

/** Reads a record and advances the buffer past it. */
template <archive_record T>
[[nodiscard]] T
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
