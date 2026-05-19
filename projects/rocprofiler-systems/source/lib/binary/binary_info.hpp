// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
