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

#include "core/config.hpp"

#if !defined(TIMEMORY_USE_BFD)
#    error "BFD support not enabled"
#endif

#define PACKAGE "rocprofiler-systems"

#include <bfd.h>

#include "analysis.hpp"
#include "binary_info.hpp"
#include "core/binary/address_range.hpp"
#include "core/binary/fwd.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "dwarf_entry.hpp"
#include "link_map.hpp"
#include "scope_filter.hpp"
#include "symbol.hpp"

#include <timemory/log/macros.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/types.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <regex>
#include <set>
#include <stdexcept>

namespace rocprofsys
{
namespace binary
{
namespace
{
binary_info
parse_line_info(const std::string& _name, bool _process_dwarf, bool _process_bfd,
                bool _include_all)
{
    auto _info = binary_info{};

    auto& _bfd = _info.bfd;
    _bfd       = std::make_shared<bfd_file>(_name);

    ROCPROFSYS_BASIC_VERBOSE(0, "[binary] Reading line info for '%s'...\n",
                             _name.c_str());

    if(_bfd && _bfd->is_good())
    {
        auto& _section_map = _info.sections;
        auto  _section_set = std::set<asection*>{};
        auto  _processed   = std::set<uintptr_t>{};
        for(auto&& itr : _bfd->get_symbols())
        {
            if(!_include_all && itr.symsize == 0) continue;
            auto& _sym = _info.symbols.emplace_back(symbol{ itr });
            // if(itr.symsize == 0) continue;
            auto* _section = static_cast<asection*>(itr.section);
            _section_set.emplace(_section);
            _processed.emplace(itr.address);
            _info.ranges.emplace_back(
                address_range{ itr.address, itr.address + itr.symsize });
            if(_process_bfd) _sym.read_bfd_line_info(*_bfd);
        }

        for(auto* itr : _section_set)
        {
            auto*         _section     = const_cast<asection*>(itr);
            bfd_vma       _section_vma = bfd_section_vma(_section);
            bfd_size_type _section_len = bfd_section_size(_section);
            auto          _section_range =
                address_range{ _section_vma, _section_vma + _section_len };
            _section_map[_section_range] = _section;
        }

        TIMEMORY_REQUIRE(_include_all || _section_set.size() == _section_map.size())
            << "section set size (" << _section_set.size() << ") != section map size ("
            << _section_map.size() << ")\n";

        if(_process_dwarf)
        {
            std::tie(_info.debug_info, _info.ranges, _info.breakpoints) =
                dwarf_entry::process_dwarf(_bfd->fd);
        }

        for(auto& itr : _info.symbols)
        {
            itr.read_dwarf_entries(_info.debug_info);
            itr.read_dwarf_breakpoints(_info.breakpoints);
        }

        _info.sort();
    }

    ROCPROFSYS_BASIC_VERBOSE(1, "[binary] Reading line info for '%s'... %zu entries\n",
                             _bfd->name.c_str(), _info.symbols.size());

    return _info;
}
}  // namespace

std::vector<binary_info>
get_binary_info(const std::vector<std::string>&  _files,
                const std::vector<scope_filter>& _filters, bool _process_dwarf,
                bool _process_bfd, bool _include_all)
{
    auto _satisfies_filter = [&_filters](auto _scope, const std::string& _value) {
        for(const auto& itr : _filters)  // NOLINT
        {
            // if the filter is for the specified scope and itr does not satisfy the
            // include/exclude mode, return false
            if((itr.scope & _scope) == _scope && !itr(_value)) return false;
        }
        return true;
    };

    auto _satisfies_binary_filter = [&_satisfies_filter](const std::string& _value) {
        return _satisfies_filter(scope_filter::BINARY_FILTER, _value);
    };

    // filter function used by procfs::get_contiguous_maps
    // ensures that we do not process rocprof-sys/gotcha/libunwind libraries
    // and do not process the libraries outside of the binary scope
    auto _filter = [&_satisfies_binary_filter](const procfs::maps& _v) {
        if(_v.pathname.empty()) return false;
        auto _path = filepath::realpath(_v.pathname, nullptr, false);
        return (filepath::exists(_path) && _satisfies_binary_filter(_path));
    };

    auto _data = std::vector<binary_info>{};
    _data.reserve(_files.size());
    {
        auto _exists = std::set<std::string>{};
        for(const auto& itr : _files)
        {
            auto _filename = filepath::realpath(itr, nullptr, false);
            if(filepath::exists(_filename) && _satisfies_binary_filter(_filename) &&
               _exists.find(_filename) == _exists.end())
            {
                _data.emplace_back(parse_line_info(_filename, _process_dwarf,
                                                   _process_bfd, _include_all));
                _exists.emplace(_filename);
            }
        }
    }

    // get the memory maps
    auto _maps = procfs::get_contiguous_maps(process::get_id(), _filter, false);

    for(auto& itr : _data)
    {
        for(const auto& mitr : _maps)
            if(itr.bfd->name == mitr.pathname) itr.mappings.emplace_back(mitr);
    }

    for(auto& itr : _data)
    {
        for(const auto& mitr : itr.mappings)
        {
            auto mrange = address_range{ mitr.load_address, mitr.last_address };
            for(auto& sitr : itr.symbols)
            {
                auto _addr = sitr.address + mitr.load_address;
                if(mrange.contains(_addr)) sitr.load_address = mitr.load_address;
            }
        }
    }

    for(auto& itr : _data)
        itr.sort();

    return _data;
}

template <bool ExcludeInternal>
std::optional<tim::unwind::processed_entry>
lookup_ipaddr_entry(uintptr_t _addr, unw_context_t* _context_p,
                    tim::unwind::cache* _cache_p)
{
    static auto _mutex     = locking::atomic_mutex{};
    static auto _cache_v   = tim::unwind::cache{ true };
    static auto _context_v = []() {
        auto _v = unw_context_t{};
        unw_getcontext(&_v);
        return _v;
    }();

    if constexpr(ExcludeInternal)
    {
        static auto _exclude_range = []() {
            auto _maps                 = ::tim::procfs::maps::iterate_program_headers();
            auto _exclude_range_v      = std::set<address_range_t>{};
            auto _insert_exclude_range = [&_maps,
                                          &_exclude_range_v](const std::string& _v) {
                auto _base_v = std::string_view{ filepath::basename(_v) };
                auto _real_v = filepath::realpath(_v);
                for(const auto& mitr : _maps)
                {
                    if(std::string_view{ filepath::basename(mitr.pathname) } == _base_v ||
                       _real_v == _v)
                    {
                        _exclude_range_v.emplace(
                            address_range_t{ mitr.load_address, mitr.last_address });
                    }
                }
            };

            for(const auto& itr : binary::get_link_map("librocprof-sys.so", "", ""))
                _insert_exclude_range(itr.real());

            for(const auto& itr : binary::get_link_map("librocprof-sys-dl.so", "", ""))
                _insert_exclude_range(itr.real());

            return _exclude_range_v;
        }();

        for(auto itr : _exclude_range)
            if(itr.contains(_addr)) return std::optional<tim::unwind::processed_entry>{};
    }

    // NOLINTNEXTLINE(readability-misleading-indentation)
    if(_addr == 0) return std::optional<tim::unwind::processed_entry>{};

    auto _lk = locking::atomic_lock{ _mutex, std::defer_lock };

    if(!_context_p) _context_p = &_context_v;
    if(!_cache_p)
    {
        _cache_p = &_cache_v;
        // prevent concurrent access to cache
        _lk.lock();
    }

    auto _entry = tim::unwind::entry{ _addr };

    auto citr = _cache_p->entries.find(_entry);
    if(citr != _cache_p->entries.end())
    {
        if(citr->second.error == 0) return citr->second;
        return std::optional<tim::unwind::processed_entry>{};
    }

    auto _v    = tim::unwind::processed_entry{};
    _v.address = _entry.address();
    _v.name    = _entry.template get_name<4096, true>(*_context_p, &_v.offset, &_v.error);

    tim::unwind::processed_entry::construct(_v, &_cache_p->files);

    if(_v.error != 0 && _v.lineinfo)
    {
        auto _lineinfo = _v.lineinfo.get();
        if(_lineinfo)
        {
            _v.name  = _lineinfo.name;
            _v.error = 0;
        }
    }
    else if(_v.info && _v.info.symbol)
    {
        _v.name  = _v.info.symbol.name;
        _v.error = 0;
    }

    _cache_p->entries.emplace(_entry, _v);

    return (_v.error == 0) ? std::optional<tim::unwind::processed_entry>{ _v }
                           : std::optional<tim::unwind::processed_entry>{};
}

template std::optional<tim::unwind::processed_entry>
lookup_ipaddr_entry<true>(uintptr_t, unw_context_t*, tim::unwind::cache*);

template std::optional<tim::unwind::processed_entry>
lookup_ipaddr_entry<false>(uintptr_t, unw_context_t*, tim::unwind::cache*);
}  // namespace binary
}  // namespace rocprofsys
