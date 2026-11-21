/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Bounded views of memory that arrives as a raw pointer
 *
 * Clang's -Wunsafe-buffer-usage flags pointer arithmetic, indexing
 * through a raw pointer, the C library's unbounded memory and string
 * functions, and building a std::span or std::string_view from a pointer
 * and a separate length, since nothing ties the length to the memory.
 * Code should work with containers and views instead, which (with
 * _GLIBCXX_ASSERTIONS) check their bounds.
 *
 * Memory that comes through a C interface (the public API, libmount,
 * main()'s argv) only comes as a pointer and a length, though. The
 * functions here turn it into a view, and are the only places that
 * suppress the warning. The caller vouches that the pointer and length
 * describe valid memory, so convert once, where the pointer enters, and
 * use the view from then on.
 */

#pragma once

#include "hipfile-warnings.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

namespace hipFile {

HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF

/// @brief View count objects starting at ptr
/// @param ptr The first object
/// @param count The number of objects
/// @return A span of the objects
template <typename T>
constexpr std::span<T>
spanFromPointer(T *ptr, size_t count) noexcept
{
    return {ptr, count};
}

/// @brief View the NUL-terminated string at str, without its NUL,
///        scanning at most maxLen bytes
///
/// A string with no NUL in its first maxLen bytes is cut to maxLen bytes,
/// so to reject a string longer than some limit, pass the limit plus one
/// and compare the length with the limit.
///
/// str may be null when maxLen is zero, as it is for an empty span.
///
/// @param str The string
/// @param maxLen The most bytes to scan for the NUL
/// @return A view of the string
inline std::string_view
cStringView(const char *str, size_t maxLen) noexcept
{
    /* strnlen()'s pointer must not be null, even with a zero limit */
    return {str, maxLen == 0 ? 0 : strnlen(str, maxLen)};
}

/// @brief View a span of characters as a string
///
/// A span's size always matches its memory, but newer versions of Clang
/// flag this anyway.
///
/// @param chars The characters
/// @return A view of the same characters
constexpr std::string_view
asStringView(std::span<const char> chars) noexcept
{
    return {chars.data(), chars.size()};
}

HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON

/// @brief View the NUL-terminated string in chars, without its NUL, or
///        all of chars if it has no NUL
///
/// Pass it a char array without decaying the array to a pointer.
///
/// @param chars The characters
/// @return A view of the string
inline std::string_view
cStringView(std::span<const char> chars) noexcept
{
    return cStringView(chars.data(), chars.size());
}

/// @brief View main()'s arguments
/// @param argc main()'s argc
/// @param argv main()'s argv
/// @return A span of the argc arguments, without argv's terminating NULL,
///         or an empty span if argc isn't positive
inline std::span<char *>
argSpan(int argc, char **argv) noexcept
{
    return spanFromPointer(argv, argc > 0 ? static_cast<size_t>(argc) : 0);
}

/// @brief Set every byte of obj to zero, padding included
///
/// Use it on a struct that's handed to the kernel or the HIP runtime,
/// where T x{} isn't enough, since it doesn't have to zero the padding.
/// Zero the object that's handed over, not one it's copied from: copying
/// or returning a struct needn't keep its padding.
///
/// @param obj The object to zero
template <typename T>
    requires std::is_trivially_copyable_v<T>
void
zeroObject(T &obj) noexcept
{
    std::ranges::fill(std::as_writable_bytes(std::span<T, 1>{std::addressof(obj), 1}), std::byte{0});
}

}
