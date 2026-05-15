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
#include "core/utility.hpp"
#include "dwarf_entry.hpp"
#include "symbol.hpp"

#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace binary
{
struct binary_info
{
    std::shared_ptr<bfd_file>                bfd         = {};
    std::vector<procfs::maps>                mappings    = {};
    std::deque<symbol>                       symbols     = {};
    std::deque<dwarf_entry>                  debug_info  = {};
    std::vector<address_range>               ranges      = {};
    std::vector<uintptr_t>                   breakpoints = {};
    std::unordered_map<address_range, void*> sections    = {};

    void        sort();
    std::string filename() const;

    template <typename RetT = void>
    RetT* find_section(uintptr_t) const;
};

inline void
binary_info::sort()
{
    utility::filter_sort_unique(mappings);
    utility::filter_sort_unique(symbols);
    utility::filter_sort_unique(ranges);
    utility::filter_sort_unique(debug_info);
    utility::filter_sort_unique(breakpoints);
}

template <typename RetT>
inline RetT*
binary_info::find_section(uintptr_t _addr) const
{
    for(const auto& sitr : sections)
    {
        if(sitr.first.contains(_addr)) return static_cast<RetT*>(sitr.second);
    }
    return nullptr;
}

inline std::string
binary_info::filename() const
{
    return (bfd) ? std::string{ bfd->name } : std::string{};
}
}  // namespace binary
}  // namespace rocprofsys
