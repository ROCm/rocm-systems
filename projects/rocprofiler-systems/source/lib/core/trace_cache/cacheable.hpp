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
#include "cache_type_traits.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <timemory/units.hpp>
#include <type_traits>

namespace rocprofsys
{
namespace trace_cache
{

using namespace std::chrono_literals;

/**
 * @brief Base interface for cacheable types in the trace cache system
 *
 * This struct serves as a common base type for all objects that can be stored
 * in the buffered storage system. Types that inherit from cacheable_t can be
 * serialized, stored in memory buffers, and later deserialized.
 */
struct cacheable_t
{
    cacheable_t() = default;
};

constexpr auto ABSOLUTE   = "ABS";
constexpr auto PERCENTAGE = "%";

constexpr size_t buffer_size     = 100 * tim::units::megabyte;
constexpr size_t flush_threshold = 80 * tim::units::megabyte;

constexpr auto CACHE_FILE_FLUSH_TIMEOUT = 10ms;

/**
 * @brief Calculate the header size for a given type identifier enum
 *
 * The header consists of the type identifier and the payload size.
 *
 * @tparam TypeIdentifierEnum The enum type used for type identification
 * @return constexpr size_t Size in bytes of the header (enum + size_t)
 */
template <typename TypeIdentifierEnum>
constexpr size_t header_size = sizeof(TypeIdentifierEnum) + sizeof(size_t);
using buffer_array_t         = std::array<uint8_t, buffer_size>;

const auto tmp_directory = std::string{ "/tmp/" };

namespace utility
{

/**
 * @brief Generate a unique filename for buffered storage files
 *
 * Creates a filename in the temporary directory based on parent and child process IDs
 * to ensure uniqueness across multiple processes.
 *
 * @param ppid Parent process ID
 * @param pid Current process ID
 * @return std::string Unique filename for the buffered storage file
 */
const auto get_buffered_storage_filename = [](const int& ppid, const int& pid) {
    return std::string{ tmp_directory + "buffered_storage_" + std::to_string(ppid) + "_" +
                        std::to_string(pid) + ".bin" };
};

/**
 * @brief Generate a unique filename for metadata files
 *
 * Creates a filename for JSON metadata files based on parent and child process IDs.
 *
 * @param ppid Parent process ID
 * @param pid Current process ID
 * @return std::string Unique filename for the metadata file
 */
const auto get_metadata_filepath = [](const int& ppid, const int& pid) {
    return std::string{ tmp_directory + "metadata_" + std::to_string(ppid) + "_" +
                        std::to_string(pid) + ".json" };
};

/**
 * @brief Calculate the serialized size of a value
 *
 * Computes the number of bytes required to serialize a value of the given type.
 * Handles different types including string literals, vectors, and fundamental types.
 *
 * @tparam T Type of the value to size (must be a supported type)
 * @param val The value to calculate size for
 * @return constexpr size_t Number of bytes required for serialization
 *
 * @note Supported types include: const char*, char*, unsigned long, unsigned int,
 *       long, unsigned char, std::vector<unsigned char>, double, and int
 *
 * @note For string literals, includes the null terminator
 * @note For std::vector<uint8_t>, includes space for size information
 */
template <typename T>
constexpr size_t
get_size_helper(T&& val)
{
    constexpr bool is_supported_type = type_traits::supported_types::is_supported<T>;
    static_assert(is_supported_type,
                  "Supported types are const char*, char*, "
                  "unsigned long, unsigned int, long, unsigned "
                  "char, std::vector<unsigned char>, double, and int.");

    if constexpr(type_traits::is_string_literal_v<T>)
    {
        size_t count = 0;
        while(val[count] != '\0')
        {
            ++count;
        }
        return ++count;
    }
    else if constexpr(std::is_same_v<std::decay_t<T>, std::vector<uint8_t>>)
    {
        return val.size() + sizeof(size_t);
    }
    else
    {
        return sizeof(T);
    }
}

/**
 * @brief Store a value in a binary buffer at the specified position
 *
 * Serializes a value into a binary buffer, handling different types appropriately.
 * The position is advanced by the number of bytes written.
 *
 * @tparam Type Type of the value to store (must be a supported type)
 * @param value The value to serialize and store
 * @param buffer Pointer to the destination buffer
 * @param position Reference to the current position in the buffer (will be updated)
 *
 * @note For string literals, copies the entire string including null terminator
 * @note For std::vector<uint8_t>, stores the size first, then the data
 * @note For fundamental types, stores the value directly using memcpy semantics
 * @note The position parameter is modified to point after the stored data
 */
template <typename Type>
void
store_value(const Type& value, uint8_t* buffer, size_t& position)
{
    constexpr bool is_supported_type = type_traits::supported_types::is_supported<Type>;
    static_assert(is_supported_type,
                  "Supported types are const char*, char*, "
                  "unsigned long, unsigned int, long, unsigned "
                  "char, std::vector<unsigned char>, double, and int.");

    size_t len  = 0;
    auto*  dest = buffer + position;
    if constexpr(type_traits::is_string_literal_v<Type>)
    {
        len = get_size_helper(value);
        std::memcpy(dest, value, len);  // will include \0
    }
    else if constexpr(std::is_same_v<std::decay_t<Type>, std::vector<uint8_t>>)
    {
        size_t elem_count = value.size();
        len               = elem_count + sizeof(size_t);
        std::memcpy(dest, &elem_count, sizeof(size_t));
        std::memcpy(dest + sizeof(size_t), value.data(), value.size());
    }
    else
    {
        using ClearType                     = std::decay_t<decltype(value)>;
        len                                 = sizeof(ClearType);
        *reinterpret_cast<ClearType*>(dest) = value;
    }
    position += len;
};

/**
 * @brief Parse and deserialize a value from binary data
 *
 * Reads a serialized value from binary data and advances the data pointer.
 * Handles different types including strings, vectors, and fundamental types.
 *
 * @tparam T Type of the value to parse
 * @param arg Reference to the output variable where the parsed value will be stored
 * @param data_pos Reference to pointer to the binary data (will be advanced)
 *
 * @note For std::string, reads a null-terminated C string
 * @note For std::vector<uint8_t>, first reads the size, then the data
 * @note For fundamental types, reads the value directly using reinterpret_cast
 * @note The data_pos pointer is advanced past the parsed data
 */
template <typename T>
static void
parse_value(T& arg, uint8_t*& data_pos)
{
    using Type = std::decay_t<std::remove_reference_t<T>>;
    if constexpr(std::is_same_v<Type, std::string>)
    {
        arg = std::string((const char*) data_pos);
        data_pos += get_size_helper((const char*) data_pos);
    }
    else if constexpr(std::is_same_v<Type, std::vector<uint8_t>>)
    {
        size_t vector_size = *reinterpret_cast<const size_t*>(data_pos);
        data_pos += sizeof(size_t);
        arg.reserve(vector_size);
        std::copy_n(data_pos, vector_size, std::back_inserter(arg));
        data_pos += vector_size;
    }
    else
    {
        arg = *reinterpret_cast<const Type*>(data_pos);
        data_pos += sizeof(Type);
    }
}

}  // namespace utility
}  // namespace trace_cache
}  // namespace rocprofsys
