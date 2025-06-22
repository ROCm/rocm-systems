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

#include "core/binary/address_range.hpp"
#include "core/binary/fwd.hpp"

#include <timemory/unwind/bfd.hpp>

#include <cstdint>
#include <deque>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace binary
{
struct inlined_symbol
{
    unsigned int line = 0;
    std::string  file = {};
    std::string  func = {};

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);
};

struct symbol : private tim::unwind::bfd_file::symbol
{
    using base_type = tim::unwind::bfd_file::symbol;

    symbol() = default;
    symbol(const base_type& _v);

    ~symbol()                 = default;
    symbol(const symbol&)     = default;
    symbol(symbol&&) noexcept = default;

    symbol& operator=(const symbol&)     = default;
    symbol& operator=(symbol&&) noexcept = default;

    bool     operator==(const symbol&) const;
    bool     operator<(const symbol&) const;
    bool     operator()(const std::vector<scope_filter>&) const;
    symbol&  operator+=(const symbol&);
    explicit operator bool() const;

    bool          read_bfd_line_info(bfd_file&);
    size_t        read_dwarf_entries(const std::deque<dwarf_entry>&);
    size_t        read_dwarf_breakpoints(const std::vector<uintptr_t>&);
    address_range ipaddr() const { return address + load_address; }
    symbol        clone() const;

    template <typename Tp = std::deque<symbol>>
    Tp get_inline_symbols(const std::vector<scope_filter>&) const;

    template <typename Tp = std::deque<symbol>>
    Tp get_debug_line_info(const std::vector<scope_filter>&) const;

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);

    using base_type::binding;
    using base_type::section;
    using base_type::visibility;

    unsigned int                line         = 0;
    uintptr_t                   load_address = 0;
    address_range               address      = {};
    std::string                 func         = {};
    std::string                 file         = {};
    std::vector<uintptr_t>      breakpoints  = {};
    std::vector<inlined_symbol> inlines      = {};
    std::vector<dwarf_entry>    dwarf_info   = {};
};
}  // namespace binary
}  // namespace rocprofsys
