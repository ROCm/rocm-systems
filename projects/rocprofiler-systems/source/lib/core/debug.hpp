// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "defines.hpp"
#include "exception.hpp"
#include "locking.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/signals/signal_handlers.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/locking.hpp>
#include <timemory/utility/utility.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <utility>

namespace rocprofsys
{
inline namespace config
{
bool
get_debug() ROCPROFSYS_HOT;

int
get_verbose() ROCPROFSYS_HOT;

bool
get_debug_env() ROCPROFSYS_HOT;

int
get_verbose_env() ROCPROFSYS_HOT;

bool
get_is_continuous_integration() ROCPROFSYS_HOT;

bool
get_debug_tid() ROCPROFSYS_HOT;

bool
get_debug_pid() ROCPROFSYS_HOT;
}  // namespace config

namespace debug
{
struct source_location
{
    std::string_view function = {};
    std::string_view file     = {};
    int              line     = 0;
};
//
void
set_source_location(source_location&&);
//
FILE*
get_file();
//
void
close_file();
//
int64_t
get_tid();
//
inline void
flush()
{
    fprintf(stdout, "%s", ::tim::log::color::end());
    fflush(stdout);
    std::cout << ::tim::log::color::end() << std::flush;
    fprintf(::rocprofsys::debug::get_file(), "%s", ::tim::log::color::end());
    fflush(::rocprofsys::debug::get_file());
    std::cerr << ::tim::log::color::end() << std::flush;
}
//
struct lock
{
    lock();
    ~lock();

private:
    locking::atomic_lock m_lk;
};
//
template <typename Arg, typename... Args>
bool
is_bracket(Arg&& _arg, Args&&...)
{
    if constexpr(::tim::concepts::is_string_type<Arg>::value)
        return (::std::string_view{ _arg }.empty()) ? false : _arg[0] == '[';
    else
        return false;
}
//
namespace
{
template <typename T, size_t... Idx>
auto
get_chars(T&& _c, std::index_sequence<Idx...>)
{
    return std::array<const char, sizeof...(Idx) + 1>{ std::forward<T>(_c)[Idx]...,
                                                       '\0' };
}
}  // namespace
}  // namespace debug

namespace binary
{
struct address_range;
}

using address_range_t = binary::address_range;

template <typename Tp>
std::string
as_hex(Tp, size_t _wdith = 16);

template <>
std::string as_hex<address_range_t>(address_range_t, size_t);

extern template std::string as_hex<int32_t>(int32_t, size_t);
extern template std::string as_hex<uint32_t>(uint32_t, size_t);
extern template std::string as_hex<int64_t>(int64_t, size_t);
extern template std::string as_hex<uint64_t>(uint64_t, size_t);
extern template std::string
as_hex<void*>(void*, size_t);
}  // namespace rocprofsys
