// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
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
using buffer_array_t         = std::array<std::uint8_t, buffer_size>;

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

// store_value writes the [type_id][size] framing header that precedes every
// archive-serialized record in buffer_storage. Only arithmetic / enum payloads
// are needed for that header, so this overload is the only remaining caller
// surface; the previous variadic / container / optional branches were dropped
// once the rest of the wire body migrated to the archive in archive.hpp.
template <typename Type>
__attribute__((always_inline)) inline void
store_value(const Type& value, std::uint8_t* buffer, size_t& position)
{
    using DecayedType = std::decay_t<Type>;
    static_assert(std::is_arithmetic_v<DecayedType> || std::is_enum_v<DecayedType>,
                  "store_value supports only arithmetic / enum payloads "
                  "(framing header). Use the archive in archive.hpp for "
                  "everything else.");

    std::memcpy(buffer + position, &value, sizeof(DecayedType));
    position += sizeof(DecayedType);
}

}  // namespace utility
}  // namespace trace_cache
}  // namespace rocprofsys
