// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/cxx/details/mpl.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
// A helper class which computes a 64-bit hash of the input data.
// The algorithm used is FNV-1a as it is fast and easy to implement and has
// relatively few collisions.
// WARNING: This hash function should not be used for any cryptographic purpose.
struct fnv1a_hasher
{
    // Creates an empty hash object
    fnv1a_hasher()  = default;
    ~fnv1a_hasher() = default;

    fnv1a_hasher(const fnv1a_hasher&)     = default;
    fnv1a_hasher(fnv1a_hasher&&) noexcept = default;

    fnv1a_hasher& operator=(const fnv1a_hasher&) = default;
    fnv1a_hasher& operator=(fnv1a_hasher&&) noexcept = default;

    // Hashes a numeric value.
    template <typename Tp, typename std::enable_if_t<std::is_arithmetic<Tp>::value, bool> = true>
    void update(Tp data);

    // Using the loop instead of "update(str, strlen(str))" to avoid looping twice
    void update(const char* str);

    // Hashes a byte array.
    void update(const char* data, size_t size);

    void update(std::string_view s) { update(s.data(), s.size()); }

    void update(const std::string& s) { update(s.data(), s.size()); }

    // Usage: uint64_t hashed_value = Hash::combine(33, false, "ABC", 458L, 3u, 'x');
    template <typename Tp, typename... TailT>
    static uint64_t combine(Tp&& arg, TailT&&... args);

    // fnv1a_hasher.update_all(33, false, "ABC")` is shorthand for calling fnv1a_hasher.update(...)
    // for each value in the same order
    template <typename Tp, typename... TailT>
    void update_all(Tp&& arg, TailT&&... args);

    uint64_t digest() const { return m_result; }

private:
    static constexpr uint64_t kFnv1a64OffsetBasis = 0xcbf29ce484222325;
    static constexpr uint64_t kFnv1a64Prime       = 0x100000001b3;

    uint64_t m_result = kFnv1a64OffsetBasis;
};

template <typename Tp, typename std::enable_if_t<std::is_arithmetic<Tp>::value, bool>>
void
fnv1a_hasher::update(Tp data)
{
    update(reinterpret_cast<const char*>(&data), sizeof(data));
}

inline void
fnv1a_hasher::update(const char* str)
{
    constexpr auto max_n = 4096;

    size_t n = 0;
    for(const auto* p = str; *p != 0; ++p)
    {
        update(*p);
        if(++n >= max_n) break;  // prevent infinite loop
    }
}

inline void
fnv1a_hasher::update(const char* data, size_t size)
{
    for(size_t i = 0; i < size; ++i)
    {
        m_result ^= static_cast<uint8_t>(data[i]);
        // Note: Arithmetic overflow of unsigned integers is well defined in C++ standard unlike
        // signed integers.
        m_result *= kFnv1a64Prime;
    }
}

template <typename Tp, typename... TailT>
uint64_t
fnv1a_hasher::combine(Tp&& arg, TailT&&... args)
{
    auto _hasher = fnv1a_hasher{};
    _hasher.update_all(std::forward<Tp>(arg), std::forward<TailT>(args)...);
    return _hasher.digest();
}

template <typename Tp, typename... TailT>
void
fnv1a_hasher::update_all(Tp&& arg, TailT&&... args)
{
    update(arg);
    if constexpr(sizeof...(TailT) > 0) update_all(std::forward<TailT>(args)...);
}
}  // namespace common
}  // namespace rocprofiler
