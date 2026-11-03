// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/units/data_size.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace std::chrono_literals;

namespace rocprofsys::trace_cache
{

using namespace common::units::literals;

constexpr size_t k_buffer_size              = static_cast<size_t>((100_mib).to_bytes());
constexpr size_t k_flush_threshold          = static_cast<size_t>((80_mib).to_bytes());
constexpr auto   k_cache_file_flush_timeout = 10ms;

constexpr auto k_absolute   = "ABS";
constexpr auto k_percentage = "%";

using buffer_array_t = std::array<std::uint8_t, k_buffer_size>;

const auto k_tmp_directory = std::string{ "/tmp/" };

namespace utility
{

const auto get_buffered_storage_filename = [](const int& ppid, const int& pid) {
    return std::string{ k_tmp_directory + "buffered_storage_" + std::to_string(ppid) +
                        "_" + std::to_string(pid) + ".bin" };
};

const auto get_metadata_filepath = [](const int& ppid, const int& pid) {
    return std::string{ k_tmp_directory + "metadata_" + std::to_string(ppid) + "_" +
                        std::to_string(pid) + ".json" };
};

template <typename Type>
    requires type_traits::supported_cache_type<Type>
__attribute__((always_inline)) constexpr size_t
get_size(Type&& val)
{
    using DecayedType = std::decay_t<Type>;

    if constexpr(type_traits::string_like<DecayedType>)
    {
        return val.size() + sizeof(char) + sizeof(size_t);
    }
    else if constexpr(type_traits::is_vector_v<DecayedType> ||
                      type_traits::is_span_v<DecayedType>)
    {
        using container_type = std::decay_t<decltype(val)>;
        static_assert(
            type_traits::is_raw_cacheable<typename container_type::value_type>::value,
            "vectors and spans are cached as raw bytes: the element must be a "
            "number, a scoped enum or a struct marked explicitly_raw");
        const size_t item_size  = sizeof(typename container_type::value_type);
        const size_t item_count = val.size();
        const size_t total_size = item_count * item_size;

        return total_size + sizeof(size_t);
    }
    else if constexpr(type_traits::is_optional_v<DecayedType>)
    {
        static_assert(!type_traits::is_optional_v<typename DecayedType::value_type>,
                      "Nested std::optional is not supported");
        return sizeof(std::uint8_t) + (val.has_value() ? get_size(val.value()) : 0);
    }
    else
    {
        static_assert(
            type_traits::is_raw_cacheable<DecayedType>::value,
            "field type is copied as raw bytes: give it serialize(Archive&) or, "
            "if it is plain data, mark it explicitly_raw");
        return sizeof(DecayedType);
    }
}

template <typename Type>
    requires type_traits::supported_cache_type<Type>
__attribute__((always_inline)) inline void
store_value(const Type& value, std::uint8_t* buffer, size_t& position)
{
    using DecayedType = std::decay_t<Type>;

    auto* dest = buffer + position;

    if constexpr(type_traits::string_like<DecayedType>)
    {
        const size_t data_size  = value.size();
        const size_t entry_size = sizeof(size_t) + data_size + sizeof(char);
        std::memcpy(dest, &data_size, sizeof(size_t));
        if(data_size != 0)
        {
            std::memcpy(dest + sizeof(size_t), value.data(), data_size);
        }
        dest[entry_size - sizeof(char)] =
            '\0';  // part of the format, see binary_read_archive
        position += entry_size;
    }
    else if constexpr(type_traits::is_vector_v<DecayedType> ||
                      type_traits::is_span_v<DecayedType>)
    {
        static_assert(
            type_traits::is_raw_cacheable<typename DecayedType::value_type>::value,
            "vectors and spans are cached as raw bytes: the element must be a "
            "number, a scoped enum or a struct marked explicitly_raw");
        const size_t total_size  = get_size(value);
        const size_t header_size = sizeof(size_t);
        const size_t data_size   = total_size - header_size;
        std::memcpy(dest, &data_size, sizeof(size_t));
        if(data_size != 0)
        {
            std::memcpy(dest + sizeof(size_t), value.data(), data_size);
        }
        position += total_size;
    }
    else if constexpr(type_traits::is_optional_v<DecayedType>)
    {
        static_assert(!type_traits::is_optional_v<typename DecayedType::value_type>,
                      "Nested std::optional is not supported");
        buffer[position++] = value.has_value() ? 1 : 0;

        if(value.has_value())
        {
            store_value(*value, buffer, position);
        }
    }
    else
    {
        static_assert(
            type_traits::is_raw_cacheable<DecayedType>::value,
            "field type is copied as raw bytes: give it serialize(Archive&) or, "
            "if it is plain data, mark it explicitly_raw");
        std::memcpy(dest, &value, sizeof(DecayedType));
        position += sizeof(DecayedType);
    }
}

template <typename Type>
    requires type_traits::supported_cache_type<Type>
__attribute__((always_inline)) inline static void
parse_value(std::uint8_t*& data_pos, Type& arg)
{
    using DecayedType = std::decay_t<Type>;

    if constexpr(type_traits::string_like<DecayedType>)
    {
        size_t string_size = 0;
        std::memcpy(&string_size, data_pos, sizeof(size_t));
        data_pos += sizeof(size_t);
        arg = std::string_view{ reinterpret_cast<const char*>(data_pos), string_size };
        data_pos += string_size + sizeof(char);
    }
    else if constexpr(type_traits::is_vector_v<DecayedType> ||
                      type_traits::is_span_v<DecayedType>)
    {
        using item_type = std::decay_t<decltype(arg)>::value_type;
        static_assert(type_traits::is_raw_cacheable<item_type>::value,
                      "vectors and spans are cached as raw bytes: the element must be a "
                      "number, a scoped enum or a struct marked explicitly_raw");
        size_t total_size = 0;
        std::memcpy(&total_size, data_pos, sizeof(size_t));
        data_pos += sizeof(size_t);
        // Only whole elements are read; a stray tail is skipped, never copied.
        const size_t item_count = total_size / sizeof(item_type);
        arg.resize(item_count);
        if(item_count != 0)
        {
            std::memcpy(arg.data(), data_pos, item_count * sizeof(item_type));
        }
        data_pos += total_size;
    }
    else if constexpr(type_traits::is_optional_v<DecayedType>)
    {
        static_assert(!type_traits::is_optional_v<typename DecayedType::value_type>,
                      "Nested std::optional is not supported");
        const bool has_value = *data_pos++;
        if(has_value)
        {
            arg.emplace();
            parse_value<typename DecayedType::value_type>(data_pos, *arg);
        }
        else
        {
            arg.reset();
        }
    }
    else
    {
        static_assert(
            type_traits::is_raw_cacheable<DecayedType>::value,
            "field type is copied as raw bytes: give it serialize(Archive&) or, "
            "if it is plain data, mark it explicitly_raw");
        std::memcpy(&arg, data_pos, sizeof(DecayedType));
        data_pos += sizeof(DecayedType);
    }
}

}  // namespace utility
}  // namespace rocprofsys::trace_cache
