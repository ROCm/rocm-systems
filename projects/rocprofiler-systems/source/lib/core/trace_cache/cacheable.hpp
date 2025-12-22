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
        const size_t total_size          = get_size(value);
        const size_t header_size         = sizeof(size_t);
        const size_t data_size           = total_size - header_size;
        *reinterpret_cast<size_t*>(dest) = data_size;
        std::memcpy(dest + sizeof(size_t), value.data(), data_size);
        position += total_size;
    }
    else
    {
        *reinterpret_cast<DecayedType*>(dest) = value;
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
        const size_t string_size = *reinterpret_cast<const size_t*>(data_pos);
        data_pos += sizeof(size_t);
        arg = std::string_view{ reinterpret_cast<const char*>(data_pos), string_size };
        data_pos += string_size;
    }
    else if constexpr(type_traits::is_vector_v<DecayedType> ||
                      type_traits::is_span_v<DecayedType>)
    {
        using ContainerType     = std::decay_t<decltype(arg)>;
        const size_t item_size  = sizeof(typename ContainerType::value_type);
        const size_t total_size = *reinterpret_cast<const size_t*>(data_pos);
        data_pos += sizeof(size_t);
        arg.reserve(total_size / item_size);
        std::copy_n(reinterpret_cast<const typename ContainerType::value_type*>(data_pos),
                    total_size / item_size, std::back_inserter(arg));
        data_pos += total_size;
    }
    else
    {
        arg = *reinterpret_cast<const DecayedType*>(data_pos);
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
