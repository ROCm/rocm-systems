// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/archive.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofsys::trace_cache
{

/** Frame in front of every record in the cache: the record type and its payload size. */
template <typename TypeIdentifierEnum>
struct record_header
{
    TypeIdentifierEnum type{};
    std::size_t        size = 0;

    template <typename Archive>
    constexpr void serialize(Archive& archive)
    {
        archive(type, size);
    }
};

template <typename TypeIdentifierEnum>
constexpr std::size_t k_header_size = [] {
    binary_size_archive archive;
    record_header<TypeIdentifierEnum>{}.serialize(archive);
    return archive.size;
}();

// One fully inlined function per record type, like serialize().
template <typename TypeIdentifierEnum, archive_record T>
[[gnu::flatten]] void
write_record(std::uint8_t* buffer, record_header<TypeIdentifierEnum> header,
             const T& record)
{
    binary_write_archive archive{ .buffer = buffer, .position = 0 };
    header.serialize(archive);
    const_cast<T&>(record).serialize(archive);
}

}  // namespace rocprofsys::trace_cache
