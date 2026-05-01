// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/cache_type_traits.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

using namespace std::chrono_literals;

namespace rocprofsys
{

namespace trace_cache
{

struct cacheable_t
{
    cacheable_t() = default;
};

constexpr size_t MByte                    = 1024 * 1024;
constexpr size_t buffer_size              = 100 * MByte;
constexpr size_t flush_threshold          = 80 * MByte;
constexpr auto   CACHE_FILE_FLUSH_TIMEOUT = 10ms;

constexpr auto ABSOLUTE   = "ABS";
constexpr auto PERCENTAGE = "%";

template <typename TypeIdentifierEnum>
constexpr size_t header_size = sizeof(TypeIdentifierEnum) + sizeof(size_t);
using buffer_array_t         = std::array<uint8_t, buffer_size>;

const auto tmp_directory = std::string{ "/tmp/" };

namespace utility
{

const auto get_buffered_storage_filename = [](const int& ppid, const int& pid) {
    return std::string{ tmp_directory + "buffered_storage_" + std::to_string(ppid) + "_" +
                        std::to_string(pid) + ".bin" };
};

const auto get_metadata_filepath = [](const int& ppid, const int& pid) {
    return std::string{ tmp_directory + "metadata_" + std::to_string(ppid) + "_" +
                        std::to_string(pid) + ".json" };
};

template <typename Type>
__attribute__((always_inline)) inline constexpr size_t
get_size(Type&& val)
{
    using DecayedType = std::decay_t<Type>;
    static_assert(type_traits::is_supported_type_v<DecayedType>,
                  "Unsupported type in get_size");

    if constexpr(type_traits::is_string_view_v<DecayedType> ||
                 type_traits::is_vector_v<DecayedType> ||
                 type_traits::is_span_v<DecayedType>)
    {
        using ContainerType     = std::decay_t<decltype(val)>;
        const size_t item_size  = sizeof(typename ContainerType::value_type);
        const size_t item_count = val.size();
        const size_t total_size = item_count * item_size;

        return total_size + sizeof(size_t);
    }
    else if constexpr(type_traits::is_optional_v<DecayedType>)
    {
        static_assert(!type_traits::is_optional_v<typename DecayedType::value_type>,
                      "Nested std::optional is not supported");
        return sizeof(uint8_t) + (val.has_value() ? get_size(val.value()) : 0);
    }
    else
    {
        return sizeof(DecayedType);
    }
}

template <typename Type, typename... Types>
__attribute__((always_inline)) inline constexpr size_t
get_size(Type&& val, Types&&... vals)
{
    return get_size(std::forward<Type>(val)) + get_size(std::forward<Types>(vals)...);
}

template <typename Type>
__attribute__((always_inline)) inline void
store_value(const Type& value, uint8_t* buffer, size_t& position)
{
    using DecayedType = std::decay_t<Type>;
    static_assert(type_traits::is_supported_type_v<DecayedType>,
                  "Unsupported type in store_value");

    auto* dest = buffer + position;

    if constexpr(type_traits::is_string_view_v<DecayedType> ||
                 type_traits::is_vector_v<DecayedType> ||
                 type_traits::is_span_v<DecayedType>)
    {
        const size_t total_size  = get_size(value);
        const size_t header_size = sizeof(size_t);
        const size_t data_size   = total_size - header_size;
        std::memcpy(dest, &data_size, sizeof(size_t));
        std::memcpy(dest + sizeof(size_t), value.data(), data_size);
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
        std::memcpy(dest, &value, sizeof(DecayedType));
        position += sizeof(DecayedType);
    }
}

template <typename... Types>
__attribute__((always_inline)) inline void
store_value(uint8_t* buffer, const Types&... values)
{
    size_t position = 0;
    (store_value(values, buffer, position), ...);
}

template <typename Type>
__attribute__((always_inline)) inline static void
parse_value(uint8_t*& data_pos, Type& arg)
{
    using DecayedType = std::decay_t<Type>;
    static_assert(type_traits::is_supported_type_v<DecayedType>,
                  "Unsupported type in parse_value");

    if constexpr(type_traits::is_string_view_v<DecayedType>)
    {
        size_t string_size = 0;
        std::memcpy(&string_size, data_pos, sizeof(size_t));
        data_pos += sizeof(size_t);
        arg = std::string_view{ reinterpret_cast<const char*>(data_pos), string_size };
        data_pos += string_size;
    }
    else if constexpr(type_traits::is_vector_v<DecayedType> ||
                      type_traits::is_span_v<DecayedType>)
    {
        using ContainerType     = std::decay_t<decltype(arg)>;
        using ItemType          = typename ContainerType::value_type;
        const size_t item_size  = sizeof(ItemType);
        size_t       total_size = 0;
        std::memcpy(&total_size, data_pos, sizeof(size_t));
        data_pos += sizeof(size_t);
        const size_t item_count = total_size / item_size;
        arg.reserve(item_count);
        for(size_t i = 0; i < item_count; ++i)
        {
            ItemType item;
            std::memcpy(&item, data_pos + i * item_size, item_size);
            arg.push_back(std::move(item));
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
        std::memcpy(&arg, data_pos, sizeof(DecayedType));
        data_pos += sizeof(DecayedType);
    }
}

template <typename Type, typename... Types>
__attribute__((always_inline)) inline static void
parse_value(uint8_t*& data_pos, Type& arg, Types&... args)
{
    parse_value(data_pos, arg);
    parse_value(data_pos, args...);
}

}  // namespace utility
}  // namespace trace_cache
}  // namespace rocprofsys
