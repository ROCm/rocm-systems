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

template <typename Type, typename Archive>
concept archive_serializable_vector =
    type_traits::is_vector_v<Type> &&
    archive_serializable<typename Type::value_type, Archive>;

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
        size += utility::get_size(value);
    }

    template <archive_serializable<binary_size_archive> Type>
    void size_one(const Type& value)
    {
        const_cast<Type&>(value).serialize(*this);
    }

    template <archive_serializable_vector<binary_size_archive> Type>
    void size_one(const Type& value)
    {
        size += sizeof(std::size_t);
        for(const auto& element : value)
        {
            size_one(element);
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
        utility::store_value(value, out, pos);
    }

    template <archive_serializable<binary_write_archive> Type>
    void write_one(const Type& value, [[maybe_unused]] std::uint8_t* out,
                   std::size_t& pos)
    {
        position = pos;
        const_cast<Type&>(value).serialize(*this);
        pos = position;
    }

    template <archive_serializable_vector<binary_write_archive> Type>
    void write_one(const Type& value, std::uint8_t* out, std::size_t& pos)
    {
        // Byte count first, as for vectors of arithmetic values; known once the elements
        // are written.
        const std::size_t count_position = pos;
        pos += sizeof(std::size_t);
        for(const auto& element : value)
        {
            write_one(element, out, pos);
        }
        const std::size_t byte_count = pos - count_position - sizeof(std::size_t);
        std::memcpy(out + count_position, &byte_count, sizeof(std::size_t));
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
        utility::parse_value(cur, value);
    }

    template <archive_serializable<binary_read_archive> Type>
    void read_one(Type& value, std::uint8_t*& cur)
    {
        cursor = cur;
        value.serialize(*this);
        cur = cursor;
    }

    template <archive_serializable_vector<binary_read_archive> Type>
    void read_one(Type& value, std::uint8_t*& cur)
    {
        std::size_t byte_count = 0;
        std::memcpy(&byte_count, cur, sizeof(std::size_t));
        cur += sizeof(std::size_t);
        const std::uint8_t* const end = cur + byte_count;
        value.clear();
        while(cur < end)
        {
            read_one(value.emplace_back(), cur);
        }
    }

    static void read_one(std::span<const std::byte>& value, std::uint8_t*& cur)
    {
        std::size_t byte_count = 0;
        std::memcpy(&byte_count, cur, sizeof(std::size_t));
        cur += sizeof(std::size_t);
        value = { reinterpret_cast<const std::byte*>(cur), byte_count };
        cur += byte_count;
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
