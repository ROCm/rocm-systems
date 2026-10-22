// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
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

/** A value (a record or a nested struct) that lists its own fields in serialize(). */
template <typename Type, typename Archive>
concept archive_serializable =
    requires(Type& value, Archive& archive) { value.serialize(archive); };

/** Sums the encoded size of the fields a record passes to it. */
struct binary_size_archive
{
    static constexpr bool k_is_loading = false;
    std::size_t           size         = 0;

    template <typename... Types>
    ROCPROFSYS_INLINE void operator()(const Types&... values)
    {
        (size_one(values), ...);
    }

private:
    template <typename Type>
    ROCPROFSYS_INLINE void size_one(const Type& value)
    {
        if constexpr(archive_serializable<Type, binary_size_archive>)
        {
            // Recurse into a nested struct; it is only read through serialize().
            const_cast<Type&>(value).serialize(*this);
        }
        else
        {
            size += utility::get_size(value);
        }
    }
};

/** Writes the fields into a buffer that already has room for record_size() bytes. */
struct binary_write_archive
{
    static constexpr bool k_is_loading = false;
    std::uint8_t*         buffer       = nullptr;
    std::size_t           position     = 0;

    template <typename... Types>
    ROCPROFSYS_INLINE void operator()(const Types&... values)
    {
        // Locals, not members: a write through the byte buffer may alias any member, so
        // members would be reloaded after every field and adjacent copies not merged.
        std::uint8_t* const out = buffer;
        std::size_t         pos = position;
        (write_one(values, out, pos), ...);
        position = pos;
    }

private:
    template <typename Type>
    ROCPROFSYS_INLINE void write_one(const Type& value, std::uint8_t* out,
                                     std::size_t& pos)
    {
        if constexpr(archive_serializable<Type, binary_write_archive>)
        {
            // Recurse into a nested struct; it is only read through serialize(). The
            // nested call works on the members, so hand the position over and take it
            // back.
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
 * Reads the fields back. string_view and span<const std::byte> fields become views into
 * the buffer, so the buffer must outlive the record (as for the parse buffer today).
 */
struct binary_read_archive
{
    static constexpr bool k_is_loading = true;
    std::uint8_t*         cursor       = nullptr;

    template <typename... Types>
    ROCPROFSYS_INLINE void operator()(Types&... values)
    {
        // Local cursor for the same reason as in binary_write_archive.
        std::uint8_t* cur = cursor;
        (read_one(values, cur), ...);
        cursor = cur;
    }

private:
    template <typename Type>
    ROCPROFSYS_INLINE void read_one(Type& value, std::uint8_t*& cur)
    {
        if constexpr(archive_serializable<Type, binary_read_archive>)
        {
            // The nested call works on the member, so hand the cursor over and take it
            // back.
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
//
// flatten inlines everything these functions call, including the record's own
// serialize(): without it GCC keeps that member out of line and copies the fields one by
// one through memory. They are not ROCPROFSYS_INLINE on purpose: flatten is lost once the
// function is itself inlined, so each record type gets one fully inlined function, as the
// hand-written specializations had before.
template <archive_record T>
[[nodiscard]] __attribute__((flatten)) std::size_t
                                       record_size(const T& record)
{
    binary_size_archive archive;
    const_cast<T&>(record).serialize(archive);
    return archive.size;
}

template <archive_record T>
__attribute__((flatten)) void
record_write(std::uint8_t* buffer, const T& record)
{
    binary_write_archive archive{ .buffer = buffer, .position = 0 };
    const_cast<T&>(record).serialize(archive);
}

template <archive_record T>
[[nodiscard]] __attribute__((flatten)) T
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
[[nodiscard]] ROCPROFSYS_INLINE std::size_t
                                get_size(const T& record)
{
    return record_size(record);
}

/** Writes a record into a buffer with room for get_size(record) bytes. */
template <archive_record T>
ROCPROFSYS_INLINE void
serialize(std::uint8_t* buffer, const T& record)
{
    record_write(buffer, record);
}

/** Reads a record and advances the buffer past it. */
template <archive_record T>
[[nodiscard]] ROCPROFSYS_INLINE T
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
