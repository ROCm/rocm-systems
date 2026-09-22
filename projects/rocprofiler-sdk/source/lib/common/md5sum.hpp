// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "lib/common/mpl.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
// helper function
std::string
compute_md5sum(std::string_view inp);

// helper function for array of string, string_view, or const char*
template <template <typename, typename...> class ContainerT, typename Tp, typename... TailT>
std::string
compute_md5sum(const ContainerT<Tp, TailT...>& inp,
               std::enable_if_t<mpl::is_string_type<Tp>::value, int> = 0);

// a small class for calculating MD5 hashes of strings or byte arrays
//
// usage:
//      1) feed it blocks of uchars with update(...)
//      2) finalize()
//      3) get hexdigest() string
//  or
//      md5sum{...}.hexdigest()
//
// assumes that char is 8 bit and int is 32 bit
class md5sum
{
public:
    using size_type                = uint32_t;  // must be 32bit
    using raw_digest_t             = std::array<uint8_t, 16>;
    static constexpr int blocksize = 64;

    template <typename Tp, typename... Args>
    explicit md5sum(Tp&& arg, Args&&... args);

    md5sum()              = default;
    ~md5sum()             = default;
    md5sum(const md5sum&) = default;
    md5sum(md5sum&&)      = default;

    md5sum& operator=(const md5sum&) = default;
    md5sum& operator=(md5sum&&) = default;

    md5sum&      update(std::string_view inp);
    md5sum&      update(const unsigned char* buf, size_type length);
    md5sum&      update(const char* buf, size_type length);
    md5sum&      finalize();
    std::string  hexdigest() const;
    std::string  hexliteral() const;
    raw_digest_t rawdigest() const { return digest; }

    template <typename Tp, typename Up = std::enable_if_t<std::is_arithmetic<Tp>::value, int>>
    md5sum& update(Tp inp);

    friend std::ostream& operator<<(std::ostream&, md5sum md5);

private:
    void transform(const uint8_t block[blocksize]);

    bool finalized = false;
    // 64bit counter for number of bits (lo, hi)
    std::array<uint32_t, 2>        count = {0, 0};
    std::array<uint8_t, blocksize> buffer{};  // overflow bytes from last 64 byte chunk
    // digest so far, initialized to magic initialization constants.
    std::array<uint32_t, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    std::array<uint8_t, 16> digest{};  // result
};

template <typename Tp, typename... Args>
md5sum::md5sum(Tp&& arg, Args&&... args)
{
    auto _update = [&](auto&& _val) {
        using value_type = common::mpl::unqualified_type_t<decltype(_val)>;
        static_assert(!std::is_pointer<value_type>::value,
                      "constructor cannot be called with pointer argument");
        update(std::forward<decltype(_val)>(_val));
    };

    _update(std::forward<Tp>(arg));
    (_update(std::forward<Args>(args)), ...);
    finalize();
}

template <typename Tp, typename Up>
md5sum&
md5sum::update(Tp inp)
{
    static_assert(std::is_arithmetic<Tp>::value, "expected arithmetic type");
    return update(reinterpret_cast<const char*>(&inp), sizeof(Tp));
}

template <template <typename, typename...> class ContainerT, typename Tp, typename... TailT>
std::string
compute_md5sum(const ContainerT<Tp, TailT...>& inp,
               std::enable_if_t<mpl::is_string_type<Tp>::value, int>)
{
    auto _val = md5sum{};
    for(const auto& itr : inp)
        _val.update(std::string_view{inp});
    _val.finalize();
    return _val.hexdigest();
}
}  // namespace common
}  // namespace rocprofiler
